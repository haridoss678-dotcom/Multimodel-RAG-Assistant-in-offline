from rag_assistant.config import settings
from rag_assistant.graph.state import GraphState
from rag_assistant.llm.ollama_client import chat
from rag_assistant.prompts.summarizer import build_messages

# Fixed cost of the generation prompt itself (role, constraints, depth and
# language instructions, the question) with an EMPTY evidence list. Measured
# directly on the installed model rather than estimated, and RE-measured after
# generation-v7 added three citation-notation lines: 416 tokens for
# short/English, rising to 493 for the longest combination (comprehensive depth
# + Tamil instruction). Budget against the worst case, not the typical one.
# Re-measure this whenever Prompt 4's wording changes - it drifted from 393 to
# 493 in a single edit, which would silently eat 100 tokens of evidence budget.
PROMPT_OVERHEAD_TOKENS = 500

# Room for the answer. num_ctx covers the prompt AND the generation, so evidence
# that fills the window leaves the model nothing to answer with. The largest
# eval_count observed across the golden set and the real corpus was 542, on a
# comprehensive answer.
ANSWER_RESERVE_TOKENS = 600

# Measured on this corpus WITH the evidence block's own per-item
# '[n] source=... time=...:' metadata included: 2.87 chars/token at 1k chars,
# 3.48 at 2.5k, 3.57 at 5k, and 3.27-3.47 across four real pipeline turns.
# The LOW end is the safe one to budget with - overestimating tokens costs a
# little evidence breadth, underestimating costs a silently truncated prompt.
CHARS_PER_TOKEN = 2.9

BATCH_SIZE = 6


def max_evidence_chars() -> int:
    """Evidence budget in characters, DERIVED from num_ctx - never hardcoded.

    CLAUDE.md §7 item 21. The bug this replaces: MAX_EVIDENCE_CHARS was a
    hardcoded 12_000 commented as "~3,000 tokens" at an assumed 4 chars/token,
    while num_ctx was a separately configurable 2048. The two could never agree
    - on this corpus 12,000 chars is ~4,100 tokens, roughly twice the entire
    window - so any evidence between ~5,000 and 12,000 chars passed the budget
    check and then silently overflowed the model. Ollama does not error on
    overflow; it drops the START of the prompt, which is where the citation
    instructions and evidence item [1] live, so the failure surfaced as
    "the model ignored its instructions" rather than as an error.

    Deriving the character budget from num_ctx makes that class of drift
    impossible: change the window and the budget follows.
    """
    budget_tokens = settings.llm_num_ctx - PROMPT_OVERHEAD_TOKENS - ANSWER_RESERVE_TOKENS
    return max(0, int(budget_tokens * CHARS_PER_TOKEN))


def _total_chars(chunks: list[dict]) -> int:
    return sum(len(c.get("text", "")) for c in chunks)


def summarize_node(state: GraphState) -> dict:
    """Map-reduce over oversized evidence (CLAUDE.md §20.3/§21's Prompt 5).
    Each batch collapses to one merged entry, citing the batch's first chunk
    (its real page/timestamp/source_id) - an honest, disclosed simplification:
    fine-grained per-fact attribution through a summarization pass is not
    attempted, but every citation produced still resolves to a real source,
    never an invented one (CLAUDE.md §22.3 still holds)."""
    chunks = state.get("retrieved", [])
    if _total_chars(chunks) <= max_evidence_chars():
        return {}

    merged = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        combined_text = "\n\n".join(c["text"] for c in batch)
        try:
            summary_text = chat(build_messages(combined_text), stream=False, label="summarize")
        except Exception:
            summary_text = combined_text[:2000]  # degrade gracefully, don't fail the request
        merged.append({**batch[0], "text": summary_text})

    return {"retrieved": merged}
