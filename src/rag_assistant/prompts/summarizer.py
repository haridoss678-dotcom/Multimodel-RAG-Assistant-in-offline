"""Prompt 5 - Summarizer / map-reduce (CLAUDE.md §21), used both for
token-budget overflow and broad-scope (§20.3) evidence."""

_SYSTEM_TEMPLATE = """You condense retrieved evidence for a broad question without losing \
citation-relevant facts. Summarize the following passages into a short, dense paragraph. \
Keep concrete facts and terms; drop repetition and filler."""


def build_messages(combined_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE},
        {"role": "user", "content": combined_text},
    ]
