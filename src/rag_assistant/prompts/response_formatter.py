"""Prompt 7 - Response formatter (CLAUDE.md §21). Only invoked when the user
asked for a specific shape ("just the code", "3 bullets")."""

_SYSTEM_TEMPLATE = """You reshape an already-verified answer into the exact format requested.
Add no new claims - only reformat what is already there.

REQUESTED FORMAT: {requested_format}
"""


def build_messages(verified_answer: str, requested_format: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE.format(requested_format=requested_format)},
        {"role": "user", "content": verified_answer},
    ]
