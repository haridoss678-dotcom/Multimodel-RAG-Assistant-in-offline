"""Prompt 3 - Decomposer (CLAUDE.md §21)."""

_SYSTEM_TEMPLATE = """You break a compound question into independently retrievable sub-questions.
Split the question into an ordered list of sub-queries; mark dependencies between them by order \
index (0-based). Most questions need only 2-3 sub-queries."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "query": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["order", "query", "depends_on"],
            },
        }
    },
    "required": ["sub_queries"],
}


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE},
        {"role": "user", "content": question},
    ]


def schema() -> dict:
    return _SCHEMA
