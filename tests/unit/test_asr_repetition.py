from rag_assistant.ingestion.asr import (
    collapse_repeated_ngrams,
    collapse_repeated_segments,
)
from rag_assistant.ingestion.models import ContentElement


def _segment(text: str, start: float, end: float) -> ContentElement:
    return ContentElement(
        element_type="transcript_segment", text=text, start_time=start, end_time=end
    )


def test_consecutive_identical_segments_collapse_to_one():
    """The real failure, verbatim from a client recording: four separate
    segments all reading the same sentence. Collapsing only inside a segment
    would have left every one of them in place."""
    elements = [
        _segment("It is not making impact.", 901.3, 903.3),
        _segment("It is not making impact.", 903.3, 905.3),
        _segment("It is not making impact.", 905.3, 907.3),
        _segment("It is not making impact.", 907.3, 909.3),
        _segment("So, it is done.", 915.3, 917.3),
    ]

    cleaned = collapse_repeated_segments(elements)

    assert len(cleaned) == 2
    # Flagged as metadata; the text itself stays clean so the marker is never
    # embedded into the vectors (§15).
    assert cleaned[0].text == "It is not making impact."
    assert cleaned[0].low_confidence is True
    # The collapsed run still spans the audio it came from, so the citation
    # timestamp stays truthful.
    assert cleaned[0].start_time == 901.3
    assert cleaned[0].end_time == 909.3
    assert cleaned[1].text == "So, it is done."
    assert cleaned[1].low_confidence is False


def test_two_repeats_are_left_alone():
    """Saying something twice is normal speech, not a decoder loop."""
    elements = [
        _segment("Correct.", 10.0, 11.0),
        _segment("Correct.", 11.0, 12.0),
    ]

    cleaned = collapse_repeated_segments(elements)

    assert len(cleaned) == 2
    assert all(e.low_confidence is False for e in cleaned)


def test_repetition_inside_one_segment_collapses():
    text = "DC is the same item DC is the same item DC is the same item"

    collapsed, changed = collapse_repeated_ngrams(text)

    assert changed is True
    assert collapsed == "DC is the same item"


def test_longest_repeating_unit_wins():
    """ "the cost" x3 must collapse as a phrase, not be read as single words."""
    collapsed, changed = collapse_repeated_ngrams("the cost the cost the cost")

    assert changed is True
    assert collapsed == "the cost"


def test_normal_text_is_untouched():
    text = "We are doing all this in inventory."

    collapsed, changed = collapse_repeated_ngrams(text)

    assert changed is False
    assert collapsed == text


def test_a_word_repeated_twice_is_not_collapsed():
    collapsed, changed = collapse_repeated_ngrams("very very good")

    assert changed is False
    assert collapsed == "very very good"
