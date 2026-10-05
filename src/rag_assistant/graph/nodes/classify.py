import re

from rag_assistant.config import settings
from rag_assistant.graph.state import GraphState
from rag_assistant.llm.ollama_client import chat_structured
from rag_assistant.prompts.classifier import SAFE_DEFAULT, Classification, build_messages, schema

# Questions that ask for a SET of things cannot be answered in one or two
# sentences, however the classifier labelled them.
#
# Measured (CLAUDE.md §7 item 51): "what are the skills it have in that pdf"
# came back response_depth="short", so generation was instructed to answer in
# one or two sentences and returned a partial list of a resume's skills. The
# depth label is a judgement call, and §13 already records this model's
# judgement as the unreliable part; the shape of the question is not a
# judgement call. Same rule §20.2 applies to temporal routing and §7 item 16 to
# follow-up detection: route on the signal simple enough to be reliable.
_ENUMERATION_RE = re.compile(
    r"\b(what\s+(are|were)|which\s+\w+\s+(are|were)|list\s|all\s+(the\s+)?\w+|"
    r"every\s+\w+|everything|overview|summar(y|ise|ize)|tell\s+me\s+about)\b",
    re.IGNORECASE,
)


# Questions that ask to be WALKED THROUGH something. Measured cause, not a
# guess (CLAUDE.md §7 item 58): "explain the code they showed for the post API"
# was classified response_depth="short", which truncated 20 reranked candidates
# to 3 - 1,070 characters against an 8,688-character budget - and then told the
# prompt to answer in one or two sentences. The segments it dropped were the
# ones naming app.post and the Pydantic model, i.e. exactly the code asked
# about. An explanation is inherently multi-part; a one-sentence answer to
# "explain X" is a wrong answer, not a concise one.
#
# Tanglish forms are included because §9 makes Tamil first-class and a real
# user types "explain pannu" / "sollu" / "kaatu", not "explain". Kept to
# unambiguous request verbs - a single word like "enna" would fire on almost
# any question.
_WALKTHROUGH_RE = re.compile(
    r"\b(explain|describe|walk\s+me\s+through|how\s+(do|does|did|to)|"
    r"why\s+(do|does|did|is|are)|show\s+me|what\s+happens|step\s+by\s+step|"
    r"sollu|solunga|sollunga|kaat(u|tu|unga)|vilakku)\b",
    re.IGNORECASE,
)


def looks_like_enumeration(question: str) -> bool:
    """True when the question asks for a set, not a single fact."""
    return bool(_ENUMERATION_RE.search(question or ""))


def looks_like_walkthrough(question: str) -> bool:
    """True when the question asks to be taken through something, so a
    one-or-two-sentence answer cannot satisfy it."""
    return bool(_WALKTHROUGH_RE.search(question or ""))


def _apply_deterministic_overrides(result: Classification, question: str) -> dict:
    data = result.model_dump()
    # Only ever widens, never narrows: a model that says "comprehensive" is
    # trusted, a model that says "short" to a question that cannot be answered
    # in one sentence is not. Same rule §20.2 uses for temporal routing - route
    # on the signal simple enough to be reliable, and keep the 0.6B model's
    # categorical judgement as one input rather than the gate.
    if data.get("response_depth") == "short" and (
        looks_like_enumeration(question) or looks_like_walkthrough(question)
    ):
        data["response_depth"] = "detailed"
    return data


def classify_node(state: GraphState) -> dict:
    """Prompt 1 (CLAUDE.md §21) - grammar-constrained JSON, retry once on
    failure, then fall back to a safe default rather than crashing the graph
    (§21's reliability policy)."""
    question = state["question"]
    context = state.get("conversation_context", "")
    messages = build_messages(question, context)

    for _attempt in range(2):
        try:
            raw = chat_structured(
                messages, schema(), label="classify", think=settings.classify_think
            )
            result = Classification(**raw)
            return {"classification": _apply_deterministic_overrides(result, question)}
        except Exception:
            continue

    return {"classification": _apply_deterministic_overrides(SAFE_DEFAULT, question)}
