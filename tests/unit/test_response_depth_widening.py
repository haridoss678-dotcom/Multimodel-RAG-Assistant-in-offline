"""CLAUDE.md §7 item 58 - the measured cause of "explain the code" answers that
were incomplete and different every time.

The classifier labelled "explain the code they showed for the post API"
response_depth="short". That did two things: capped the reranked evidence at 3
of 20 candidates - 1,070 characters against a measured 8,688-character budget,
12% of what was available for free - and told the generation prompt to answer in
one or two sentences. The dropped segments were the ones naming `app.post` and
the Pydantic model, i.e. exactly the code being asked about.
"""

from rag_assistant.graph.nodes.classify import (
    _apply_deterministic_overrides,
    looks_like_enumeration,
    looks_like_walkthrough,
)
from rag_assistant.prompts.classifier import Classification


def _classify(depth: str) -> Classification:
    return Classification(
        language="en",
        intent="explanation",
        response_depth=depth,
        time_reference=None,
        needs_rewrite=False,
        needs_decomposition=False,
    )


def test_explain_question_is_widened_off_short():
    """An explanation is inherently multi-part; one sentence is a wrong answer
    to "explain X", not a concise one."""
    out = _apply_deterministic_overrides(_classify("short"), "explain the code for the post API")
    assert out["response_depth"] == "detailed"


def test_tanglish_walkthrough_requests_are_widened_too():
    """§9 makes Tamil first-class - a real user types "explain pannunga" or
    "kaatunga", and the widening has to see those as the same request."""
    for question in ("andha code explain pannunga", "code kaatunga", "konjam sollunga"):
        out = _apply_deterministic_overrides(_classify("short"), question)
        assert out["response_depth"] == "detailed", question


def test_a_genuine_one_fact_question_stays_short():
    """The widening must not fire on everything, or it costs real wall-clock on
    every lookup - a short question measured 28-36s precisely because it stays
    short (§7 item 56)."""
    for question in ("what is the candidate's email address", "who is speaker 1", "DC enna"):
        out = _apply_deterministic_overrides(_classify("short"), question)
        assert out["response_depth"] == "short", question


def test_widening_never_narrows():
    """Only ever widens: a model that says comprehensive is trusted."""
    out = _apply_deterministic_overrides(_classify("comprehensive"), "what is the email address")
    assert out["response_depth"] == "comprehensive"


def test_enumeration_and_walkthrough_are_independent_signals():
    """Two different shapes, both unanswerable in one sentence, and neither
    subsumes the other."""
    assert looks_like_enumeration("what are the skills in that pdf")
    assert not looks_like_walkthrough("what are the skills in that pdf")
    assert looks_like_walkthrough("how does the post API work")
    assert not looks_like_enumeration("how does the post API work")
