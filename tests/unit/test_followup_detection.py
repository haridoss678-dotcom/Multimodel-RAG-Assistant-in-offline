import pytest

from rag_assistant.graph.nodes.rewrite import looks_like_followup


@pytest.mark.parametrize(
    "question",
    [
        "explain that in more detail",
        "tell me more",
        "what about the second one",
        "and the other one?",
        "why?",
        "adhu enna?",
        "andha function epdi work agum?",
        "more details",
    ],
)
def test_follow_ups_are_detected(question):
    assert looks_like_followup(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is a decorator?",
        "Show me the decorator example from the Python notes",
        "What did the speaker say about embeddings around 32 minutes?",
        "decorator na enna? konjam explain pannunga",
        "",
    ],
)
def test_standalone_questions_are_not_flagged(question):
    """A false positive costs a real 15-35s LLM call on this hardware, so the
    check must stay conservative - a self-contained question with a concrete
    subject is never a follow-up, even when it is long or code-mixed."""
    assert looks_like_followup(question) is False
