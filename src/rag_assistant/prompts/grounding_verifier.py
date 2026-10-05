"""Prompt 6 - Grounding verifier (CLAUDE.md §21/§22.2). Runs AFTER the
deterministic citation check (§22.1) - this is the semantic layer, judging
whether cited content actually supports the claim, not just whether the
citation index exists."""

_SYSTEM_TEMPLATE = """You audit whether an answer's claims are actually supported by the cited \
evidence. For each sentence with a [n] marker, check it against evidence item [n]. A claim is \
supported only if the evidence text actually states it - not just related to it.

EVIDENCE:
{evidence_block}
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citation": {"type": "integer"},
                    "supported": {"type": "boolean"},
                },
                "required": ["text", "citation", "supported"],
            },
        },
        "overall_verdict": {
            "type": "string",
            "enum": ["grounded", "partially_grounded", "unsupported"],
        },
    },
    "required": ["claims", "overall_verdict"],
}


def build_messages(answer: str, evidence_block: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_TEMPLATE.format(evidence_block=evidence_block)},
        {"role": "user", "content": answer},
    ]


def schema() -> dict:
    return _SCHEMA
