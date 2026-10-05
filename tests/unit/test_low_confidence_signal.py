"""The confidence signal must survive as METADATA end to end, never as text.

An earlier implementation wrote "[LOW CONFIDENCE - review]" into the segment's
own text, so BGE-M3 embedded those literal words into every flagged chunk's
dense and sparse vectors, and nothing in retrieval, reranking or grounding
could read the signal at all (CLAUDE.md §15/§17/§20).
"""

from rag_assistant.graph.nodes.citation import resolve_citations_node
from rag_assistant.graph.nodes.reranking import LOW_CONFIDENCE_PENALTY
from rag_assistant.ingestion.asr import REVIEW_ANNOTATION, collapse_repeated_segments
from rag_assistant.ingestion.chunking import chunk_elements
from rag_assistant.ingestion.models import ContentElement


def _segment(text: str, start: float, end: float) -> ContentElement:
    return ContentElement(
        element_type="transcript_segment", text=text, start_time=start, end_time=end
    )


def test_collapsed_loop_is_flagged_without_touching_the_text():
    elements = [
        _segment("It is not making impact.", 901.3, 903.3),
        _segment("It is not making impact.", 903.3, 905.3),
        _segment("It is not making impact.", 905.3, 907.3),
    ]

    cleaned = collapse_repeated_segments(elements)

    assert len(cleaned) == 1
    assert cleaned[0].low_confidence is True
    # The marker must NOT be in the content that gets embedded.
    assert cleaned[0].text == "It is not making impact."
    assert REVIEW_ANNOTATION not in cleaned[0].text


def test_clean_speech_is_not_flagged():
    cleaned = collapse_repeated_segments([_segment("We are doing this in inventory.", 0.0, 3.0)])

    assert cleaned[0].low_confidence is False
    assert REVIEW_ANNOTATION not in cleaned[0].text


def test_flag_survives_chunking_into_the_qdrant_payload():
    """§17 indexes low_confidence, so chunking must carry it through - without
    this the signal would die between ContentElement and Chunk."""
    element = _segment("It is not making impact.", 10.0, 12.0)
    element.low_confidence = True
    element.confidence_score = -1.8

    chunks = chunk_elements([element], source_id="src1")

    assert len(chunks) == 1
    payload = chunks[0].model_dump()
    assert payload["low_confidence"] is True
    assert payload["confidence_score"] == -1.8
    assert REVIEW_ANNOTATION not in payload["text"]


def test_citation_from_a_flagged_segment_carries_a_visible_warning(isolated_storage):
    """Policy (b), §20: flagged evidence stays usable but the citation says so.
    Neither grounding check (§22) asks whether the cited text is itself
    trustworthy, so this is the only thing that tells the reader."""
    from rag_assistant.storage.db import get_connection

    conn = get_connection()
    conn.execute(
        "INSERT INTO sources (source_id, filename, source_type, state, created_at, updated_at) "
        "VALUES ('src1', 'meeting.m4a', 'audio', 'READY', datetime('now'), datetime('now'))"
    )
    conn.commit()
    conn.close()

    result = resolve_citations_node(
        {
            "answer": "They discussed the batch costing approach [1].",
            "retrieved": [
                {
                    "source_id": "src1",
                    "start_time": 120.0,
                    "end_time": 124.0,
                    "text": "They discussed the batch costing approach",
                    "low_confidence": True,
                }
            ],
        }
    )

    assert result["citations"][0]["low_confidence"] is True


def test_clean_citation_has_no_warning_key(isolated_storage):
    from rag_assistant.storage.db import get_connection

    conn = get_connection()
    conn.execute(
        "INSERT INTO sources (source_id, filename, source_type, state, created_at, updated_at) "
        "VALUES ('src1', 'meeting.m4a', 'audio', 'READY', datetime('now'), datetime('now'))"
    )
    conn.commit()
    conn.close()

    result = resolve_citations_node(
        {
            "answer": "They discussed the batch costing approach [1].",
            "retrieved": [
                {
                    "source_id": "src1",
                    "start_time": 120.0,
                    "text": "They discussed the batch costing approach",
                }
            ],
        }
    )

    assert "low_confidence" not in result["citations"][0]


def test_flagged_chunks_are_down_weighted_not_excluded():
    """A flagged chunk loses to any comparable clean one, but is still
    reachable when nothing better exists - hard exclusion would silently
    answer "not found" for a question whose evidence sits in a noisy passage."""
    assert 0 < LOW_CONFIDENCE_PENALTY < 1
