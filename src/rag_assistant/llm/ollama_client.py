import json
import logging
import threading
from collections.abc import Iterator

import httpx

from rag_assistant.config import settings

logger = logging.getLogger(__name__)

# Fraction of num_ctx above which a prompt is treated as at risk of truncation.
# Ollama does NOT error when a prompt exceeds the context window - it silently
# drops tokens. This project's prompts put the system instructions and evidence
# item [1] at the START, which is exactly the end that gets dropped, so an
# overflow is indistinguishable from "the model ignored its instructions"
# unless it is detected here. CLAUDE.md §7 item 21.
_CONTEXT_WARN_FRACTION = 0.9

# Per-turn token accounting. threading.local rather than a module global
# because FastAPI runs its plain `def` routes in a threadpool - one request per
# thread, but several requests potentially in flight - so a shared list would
# interleave two turns' calls. One graph run executes entirely on its calling
# thread, which makes thread-local exactly the right scope.
_call_stats = threading.local()


def reset_call_stats() -> None:
    """Starts a fresh token-accounting window for the current thread."""
    _call_stats.calls = []


def get_call_stats() -> list[dict]:
    """Every LLM call recorded since the last reset_call_stats() on this thread."""
    return list(getattr(_call_stats, "calls", []))


def _record_call(label: str, data: dict) -> None:
    """Captures Ollama's own token counts for one call.

    `prompt_eval_count` (input tokens) and `eval_count` (generated tokens) are
    the field names on the installed Ollama 0.34.2 - confirmed against a real
    response before this was written, not taken from memory (CLAUDE.md §7
    item 4). They are absent on some error shapes, hence the None handling.
    """
    prompt_tokens = data.get("prompt_eval_count")
    eval_tokens = data.get("eval_count")
    num_ctx = settings.llm_num_ctx

    calls = getattr(_call_stats, "calls", None)
    if calls is None:
        calls = []
        _call_stats.calls = calls
    calls.append(
        {
            "label": label,
            "prompt_eval_count": prompt_tokens,
            "eval_count": eval_tokens,
            "num_ctx": num_ctx,
        }
    )

    if prompt_tokens is None:
        return
    logger.info(
        "llm[%s] prompt_eval_count=%s eval_count=%s num_ctx=%s",
        label or "?",
        prompt_tokens,
        eval_tokens,
        num_ctx,
    )
    if prompt_tokens >= num_ctx * _CONTEXT_WARN_FRACTION:
        logger.warning(
            "llm[%s] prompt is %s tokens against num_ctx=%s - Ollama truncates silently, "
            "dropping the START of the prompt (system instructions and evidence item [1])",
            label or "?",
            prompt_tokens,
            num_ctx,
        )


def chat_structured(
    messages: list[dict],
    json_schema: dict,
    label: str = "",
    think: bool | None = None,
) -> dict:
    """Ollama's native `format` field, confirmed to do genuine grammar-
    constrained decoding against a JSON schema (CLAUDE.md §13/§21) - not
    prompt-engineering-and-hope. Used for Prompts 1/2/3/6, never for
    streaming generation (§21 - the two aren't combinable on Ollama).

    `label` names the calling prompt so per-turn token accounting can say
    WHICH call overflowed, not just that one did."""
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "format": json_schema,
        "stream": False,
        "options": {"num_ctx": settings.llm_num_ctx, "temperature": settings.llm_temperature},
    }
    # qwen3 is a thinking model and spends 220-380 tokens reasoning before
    # filling a six-field JSON. Disabling that is measured at 12.5s -> 4.3s.
    # It is NOT safe everywhere: for GENERATION it destroyed citations (0/3
    # runs vs 2/3), so this is opt-in per call site, never global (§7 item 53).
    if think is not None:
        payload["think"] = think
    resp = httpx.post(
        f"{settings.ollama_host}/api/chat", json=payload, timeout=settings.ollama_timeout_seconds
    )
    resp.raise_for_status()
    data = resp.json()
    _record_call(label or "structured", data)
    return json.loads(data["message"]["content"])


def chat(messages: list[dict], stream: bool = False, label: str = "") -> str | Iterator[str]:
    """Calls Ollama's *native* /api/chat endpoint (CLAUDE.md §13 - not the
    OpenAI-compat endpoint, which doesn't yet reliably enforce JSON schemas).
    num_ctx is explicitly bounded (measured to matter for RAM on this
    machine, §29) rather than left at the model's much larger default.

    temperature is deliberately low (0.2, not Ollama's default ~0.7-0.8):
    confirmed by direct testing that at default temperature, qwen3:0.6b's
    citation-marker inclusion is non-deterministic enough to occasionally
    fail even in scenarios that pass most of the time (CLAUDE.md §13) - this
    is a grounded-QA task where consistency matters more than creative
    variation, so a low (not necessarily zero - full determinism isn't
    guaranteed cross-platform anyway) temperature is the right default here.
    """
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": stream,
        "options": {"num_ctx": settings.llm_num_ctx, "temperature": settings.llm_temperature},
    }
    url = f"{settings.ollama_host}/api/chat"

    if not stream:
        resp = httpx.post(url, json=payload, timeout=settings.ollama_timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        _record_call(label or "chat", data)
        return data["message"]["content"]

    def _iter() -> Iterator[str]:
        with httpx.stream(
            "POST", url, json=payload, timeout=settings.ollama_timeout_seconds
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    # Only the final frame carries the token counts.
                    _record_call(label or "chat-stream", data)
                    break

    return _iter()
