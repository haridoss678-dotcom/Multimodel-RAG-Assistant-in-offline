import json
import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_assistant.api import progress
from rag_assistant.config import settings
from rag_assistant.graph.build_graph import ask
from rag_assistant.llm.ollama_client import get_call_stats, reset_call_stats
from rag_assistant.memory import conversation_memory as cm
from rag_assistant.storage.db import get_connection, write_trace_record

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    source_id: str | None = None
    # Scope the answer to several sources at once. One answer must never
    # silently blend two clients' material, and "all sources" has to be a
    # deliberate choice rather than the only option (§7 item 26).
    source_ids: list[str] | None = None
    conversation_id: str | None = None
    # Client-generated, so the UI can poll GET /chat/progress/{id} while it
    # waits out a ~107s turn instead of showing a spinner that is
    # indistinguishable from a hang (§7 item 17's lesson, applied to chat).
    request_id: str | None = None


def _scoped_filenames(source_ids: list[str]) -> list[str]:
    conn = get_connection()
    placeholders = ", ".join("?" for _ in source_ids)
    rows = conn.execute(
        f"SELECT source_id, filename FROM sources WHERE source_id IN ({placeholders})",
        source_ids,
    ).fetchall()
    conn.close()
    by_id = {r["source_id"]: r["filename"] for r in rows}
    return [by_id[s] for s in source_ids if s in by_id]


def _reject_unready_scope(source_ids: list[str]) -> None:
    """A scoped source that is not READY cannot answer anything yet.

    Nothing checked this, so asking a question while ingestion was still
    running searched a HALF-INDEXED corpus: EMBEDDING and INDEXING write
    vectors incrementally, so retrieval returned whatever chunks happened to
    exist at that instant. The answer came back thin or uncited with no
    explanation, and the same question minutes later - once indexing had
    finished - answered correctly. That is indistinguishable from the app
    being unreliable, which is exactly what a real user reported.

    A 409 with the actual state is the honest response: the request is not
    malformed, the corpus is simply not ready yet, and the UI already polls
    GET /jobs/{id} to know when it will be.
    """
    if not source_ids:
        return
    conn = get_connection()
    placeholders = ", ".join("?" for _ in source_ids)
    rows = conn.execute(
        f"SELECT source_id, filename, state FROM sources WHERE source_id IN ({placeholders})",
        source_ids,
    ).fetchall()
    conn.close()
    unready = [r for r in rows if r["state"] != "READY"]
    if not unready:
        return
    names = ", ".join(f"{r['filename']} ({r['state']})" for r in unready)
    raise HTTPException(
        status_code=409,
        detail={
            "code": "SOURCE_NOT_READY",
            "message": (
                f"Still processing: {names}. Its text is only partly indexed, so an "
                "answer now would be based on incomplete evidence. Wait for it to "
                "reach READY, or change the scope."
            ),
        },
    )


def _derive_title(message: str, source_ids: list[str]) -> str:
    """The label the conversation list shows.

    The document name comes first because that is what a user is actually
    looking for when scanning past conversations - "Maxcret meeting.m4a"
    identifies a conversation, "2026-02-14 09:12" does not. The question is
    only the fallback, for a conversation with no scope to name.
    """
    names = _scoped_filenames(source_ids)
    if len(names) == 1:
        return names[0][: cm.MAX_TITLE_CHARS]
    if len(names) > 1:
        return f"{names[0][: cm.MAX_TITLE_CHARS - 10]} +{len(names) - 1} more"
    return cm.heading_from_question(message)


def _diagnostics(result: dict, token_stats: list[dict], elapsed_ms: int) -> dict:
    """Everything the graph already computed, surfaced instead of discarded.

    retrieval_strategy, candidate/reranked counts, the deterministic citation
    check, the grounding verdict and retry count all existed in GraphState and
    reached nothing but a trace row nobody could read. This is what turns the
    app from a chat box into something debuggable (§30).

    Fields the request genuinely cannot determine are None, never a plausible
    fabrication - §30's own rule.
    """
    classification = result.get("classification") or {}
    verification = result.get("verification") or {}
    retrieved = result.get("retrieved") or []
    max_prompt, total_eval = _summarise_tokens(token_stats)
    return {
        "retrieval_strategy": result.get("retrieval_strategy"),
        "candidate_count": result.get("candidate_count"),
        "reranked_count": result.get("reranked_count"),
        "evidence_items": len(retrieved),
        "evidence_chars": sum(len(c.get("text", "")) for c in retrieved),
        "citation_check": result.get("citation_check") or {},
        "grounding_verdict": verification.get("verdict"),
        "retries_used": verification.get("retries", 0),
        # Summarization collapses a batch onto its first chunk's metadata, so
        # knowing it fired is the difference between a page-exact citation and
        # a first-chunk-of-batch-exact one (§7 item 28).
        "summarized": any(c.get("label") == "summarize" for c in token_stats),
        "intent": classification.get("intent"),
        "response_depth": classification.get("response_depth"),
        "language": classification.get("language"),
        "rewritten_query": result.get("standalone_query"),
        "llm_calls": [
            {
                "node": c["label"],
                "prompt_tokens": c["prompt_eval_count"],
                "eval_tokens": c["eval_count"],
            }
            for c in token_stats
        ],
        "max_prompt_tokens": max_prompt,
        "total_eval_tokens": total_eval,
        "num_ctx": settings.llm_num_ctx,
        "elapsed_ms": elapsed_ms,
    }


def _summarise_tokens(token_stats: list[dict]) -> tuple[int | None, int | None]:
    """Collapses a turn's several LLM calls into the two trace columns.

    A turn makes 3-5 calls, so neither column can be "the" count. The choices
    are deliberate: prompt_eval_count is the MAXIMUM, because the question that
    column has to answer is "did any call in this turn approach num_ctx and get
    silently truncated" - an average would hide exactly the call that did.
    eval_count is the SUM, because total tokens generated is genuine per-turn
    work. Both are None when no call reported counts, so the row stays honest
    rather than recording a fabricated 0 (CLAUDE.md §30).
    """
    prompts = [c["prompt_eval_count"] for c in token_stats if c.get("prompt_eval_count")]
    evals = [c["eval_count"] for c in token_stats if c.get("eval_count")]
    return (max(prompts) if prompts else None, sum(evals) if evals else None)


def _record_trace(
    trace_id: str,
    conversation_id: str,
    question: str,
    result: dict,
    elapsed_ms: int,
    token_stats: list[dict],
) -> None:
    """CLAUDE.md §30 - one trace row per request so a bad answer is
    attributable to a stage (bad rewrite / bad retrieval / mis-citation /
    genuinely-missing evidence) rather than defaulting to "the LLM was wrong".

    Only per-request total latency is recorded; a per-node breakdown would
    need instrumentation inside every graph node, so that field carries just
    the total rather than a fabricated split.
    """
    classification = result.get("classification") or {}
    citation_check = result.get("citation_check") or {}
    verification = result.get("verification") or {}
    max_prompt_tokens, total_eval_tokens = _summarise_tokens(token_stats)
    try:
        write_trace_record(
            {
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "raw_query": question,
                "rewritten_query": result.get("standalone_query"),
                "intent": classification.get("intent"),
                "retrieval_strategy": result.get("retrieval_strategy"),
                "candidate_count": result.get("candidate_count"),
                "reranked_count": result.get("reranked_count"),
                "selected_chunk_ids_json": json.dumps(
                    [c.get("chunk_id") for c in result.get("retrieved", [])]
                ),
                "llm_model": settings.llm_model,
                "stage_latency_ms_json": json.dumps({"total_ms": elapsed_ms}),
                "citation_check_result": json.dumps(citation_check),
                "grounding_verdict": verification.get("verdict"),
                "final_answer": result.get("formatted_answer") or result.get("answer", ""),
                "prompt_eval_count": max_prompt_tokens,
                "eval_count": total_eval_tokens,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
    except Exception:
        # Telemetry must never take down the request it is describing.
        logger.exception("failed to write trace record %s", trace_id)


@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    """V3: conversation memory (CLAUDE.md §4.3) - a new conversation is
    created if none is given, so callers can always start chatting without
    a separate setup step."""
    scope = req.source_ids or ([req.source_id] if req.source_id else [])
    _reject_unready_scope(scope)
    conversation_id = req.conversation_id
    if not conversation_id or not cm.conversation_exists(conversation_id):
        conversation_id = cm.create_conversation(title=_derive_title(req.message, scope))

    context = cm.load_recent_context(conversation_id)
    cm.save_turn(conversation_id, "user", req.message)

    trace_id = str(uuid.uuid4())
    started = time.monotonic()
    # Starts this thread's token-accounting window. The graph runs entirely on
    # the calling thread, so everything ask() triggers is attributed to this turn.
    reset_call_stats()
    progress.start(req.request_id or trace_id)
    try:
        result = ask(
            req.message,
            source_id=req.source_id,
            source_ids=req.source_ids,
            conversation_context=context,
            conversation_id=conversation_id,
            request_id=req.request_id or trace_id,
        )
    except Exception as exc:
        # A bare 500 with no detail (FastAPI's default) is undiagnosable -
        # confirmed the hard way: a real "Internal Server Error" during
        # manual testing turned out to be a stale leftover server process
        # holding Qdrant's local-mode lock (only one process may access it
        # at a time). Logging + a real error code makes the *next* one
        # diagnosable without needing to reproduce it separately.
        logger.exception("chat_endpoint failed for conversation %s", conversation_id)
        detail = {"code": "INTERNAL_ERROR", "message": str(exc), "trace_id": trace_id}
        raise HTTPException(status_code=500, detail=detail) from exc
    finally:
        # Always dropped, so a failed turn cannot leave a stale entry that the
        # UI would poll forever.
        progress.finish(req.request_id or trace_id)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    answer = result.get("formatted_answer") or result.get("answer", "")
    citations = result.get("citations", [])

    token_stats = get_call_stats()
    cm.save_turn(
        conversation_id,
        "assistant",
        answer,
        cited_sources=citations,
        language=(result.get("classification") or {}).get("language"),
        grounded=bool(result.get("grounded", False)),
    )
    _record_trace(trace_id, conversation_id, req.message, result, elapsed_ms, token_stats)

    needs_summary, older = cm.needs_summarization(conversation_id)
    if needs_summary:
        _summarize_older_messages(conversation_id, older)

    return {
        "answer": answer,
        "sources": citations,
        "grounded": result.get("grounded", False),
        "conversation_id": conversation_id,
        "trace_id": trace_id,
        "diagnostics": _diagnostics(result, token_stats, elapsed_ms),
    }


@router.get("/chat/progress/{request_id}")
def chat_progress(request_id: str):
    """Where a turn currently is. Polled once a second while /chat is open.

    404 means "not tracked" - either the turn finished, or it has not started
    yet. The client treats both as "keep waiting" rather than as an error,
    because a 404 here is never a failure of the answer itself."""
    entry = progress.get(request_id)
    if entry is None:
        raise HTTPException(
            status_code=404, detail={"code": "NOT_FOUND", "message": "no such in-flight request"}
        )
    return entry


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """The stored trace row for a past answer (§30).

    /chat returns a trace_id precisely so a specific bad answer can be looked up
    afterwards; until now nothing could look one up."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM trace_records WHERE trace_id = ?", (trace_id,)).fetchone()
    conn.close()
    if row is None:
        detail = {"code": "NOT_FOUND", "message": "no such trace"}
        raise HTTPException(status_code=404, detail=detail)
    return dict(row)


def _summarize_older_messages(conversation_id: str, older: list[dict]) -> None:
    from rag_assistant.llm.ollama_client import chat
    from rag_assistant.prompts.summarizer import build_messages

    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    try:
        summary = chat(build_messages(transcript), stream=False, label="conversation_summary")
        cm.save_summary(conversation_id, older[-1]["message_id"], summary)
    except Exception:
        pass  # summarization is a memory-size optimization, not correctness-critical
