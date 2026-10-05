"""Prompt 1 - Classifier (CLAUDE.md §21). Structured JSON output via Ollama's
native `format` field (§13) - never parsed from free text."""

from pydantic import BaseModel

VERSION = "classifier-v1"

_SYSTEM_TEMPLATE = """You classify questions for a bilingual (English/Tamil) document-and-media \
Q&A assistant.
Given recent conversation context and the current question, determine language, intent, \
response depth, whether a time reference is present, and whether retrieval needs rewriting \
or decomposition.

CONTEXT:
{context}
"""


class Classification(BaseModel):
    language: str  # "en" | "ta" | "mixed"
    intent: str  # definition|explanation|comparison|code_lookup|table_lookup|fact_lookup|
    # timeline|speaker_specific|comprehensive|follow_up|multi_hop
    response_depth: str  # short|normal|detailed|comprehensive
    time_reference: str | None = None
    needs_rewrite: bool
    needs_decomposition: bool


SAFE_DEFAULT = Classification(
    language="en",
    intent="explanation",
    response_depth="normal",
    time_reference=None,
    needs_rewrite=False,
    needs_decomposition=False,
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["en", "ta", "mixed"]},
        "intent": {
            "type": "string",
            "enum": [
                "definition",
                "explanation",
                "comparison",
                "code_lookup",
                "table_lookup",
                "fact_lookup",
                "timeline",
                "speaker_specific",
                "comprehensive",
                "follow_up",
                "multi_hop",
            ],
        },
        "response_depth": {
            "type": "string",
            "enum": ["short", "normal", "detailed", "comprehensive"],
        },
        "time_reference": {"type": ["string", "null"]},
        "needs_rewrite": {"type": "boolean"},
        "needs_decomposition": {"type": "boolean"},
    },
    "required": [
        "language",
        "intent",
        "response_depth",
        "time_reference",
        "needs_rewrite",
        "needs_decomposition",
    ],
}


def build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE.format(context=context or "(none)")},
        {"role": "user", "content": question},
    ]


def schema() -> dict:
    return _SCHEMA
