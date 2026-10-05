"""Prompt 2 - Rewriter (CLAUDE.md §21)."""

from pydantic import BaseModel

VERSION = "rewriter-v1"

_SYSTEM_TEMPLATE = """You turn a conversational follow-up into a standalone retrieval query.
Rewrite the question into a fully self-contained question using the conversation context below.
Keep the rewritten query in the same language/register the user used - do not translate it.
Conversation history is used ONLY to resolve what the user is referring to - never treat a \
prior assistant statement as a fact to restate without new evidence.

CONTEXT:
{context}
"""


class Rewrite(BaseModel):
    standalone_query: str
    referenced_topics: list[str] = []


_SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_query": {"type": "string"},
        "referenced_topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["standalone_query", "referenced_topics"],
}


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE.format(context=context or "(none)")},
        {"role": "user", "content": question},
    ]


def schema() -> dict:
    return _SCHEMA
