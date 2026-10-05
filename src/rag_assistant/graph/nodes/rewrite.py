import re

from rag_assistant.graph.state import GraphState
from rag_assistant.llm.ollama_client import chat_structured
from rag_assistant.prompts.rewriter import Rewrite, build_messages, schema

# Openers that only make sense as a continuation of something already said.
_CONTINUATION_RE = re.compile(
    r"^\s*(and\b|also\b|what about\b|how about\b|explain\b|tell me more\b|more\b|"
    r"why\b|innum\b|adhukku\b|adhula\b|apdina\b)",
    re.IGNORECASE,
)
# Words that point at something outside the question itself (English + Tanglish).
_DEICTIC_RE = re.compile(
    r"\b(that|those|these|this|it|its|they|them|same|above|previous|earlier|"
    r"adhu|andha|adhula|adhan|idhu|indha|avanga)\b",
    re.IGNORECASE,
)


def looks_like_followup(question: str) -> bool:
    """Deterministic follow-up detection, used to back up the classifier's
    `needs_rewrite` flag rather than replace it.

    CLAUDE.md §13 records `needs_rewrite` as measurably unreliable on
    qwen3:0.6b - it missed a plain "explain that in more detail" right after a
    decorator question. When it misses, retrieval runs on a question full of
    unresolved pronouns and finds the wrong thing. This is the same design rule
    §20.2 already applies to temporal routing: route on the signal that is
    simple enough to be reliable.

    Kept deliberately conservative - a false positive costs a real 15-35s LLM
    call on this hardware (§7 item 15), so it fires only on an explicit
    continuation opener, or on a short question containing a pointing word.
    """
    text = question.strip()
    if not text:
        return False
    if _CONTINUATION_RE.match(text):
        return True
    return len(text.split()) <= 6 and bool(_DEICTIC_RE.search(text))


def rewrite_node(state: GraphState) -> dict:
    """Prompt 2 (CLAUDE.md §21) - only reached when the classifier set
    needs_rewrite=True (§4.2's conditional edge). Falls back to the original
    question unchanged if the LLM call fails twice, rather than crashing."""
    context = state.get("conversation_context", "")
    messages = build_messages(state["question"], context)

    for _attempt in range(2):
        try:
            raw = chat_structured(messages, schema(), label="rewrite")
            result = Rewrite(**raw)
            return {"standalone_query": result.standalone_query}
        except Exception:
            continue

    return {"standalone_query": state["question"]}
