import logging

from rag_assistant.storage.db import get_connection, write_trace_record


def test_trace_record_round_trips_and_leaves_unknown_fields_null(isolated_storage):
    """CLAUDE.md §30 - the trace table existed but nothing ever wrote to it.
    Fields the caller can't determine must stay NULL rather than be invented
    (§9's accuracy-honesty rule applied to telemetry)."""
    write_trace_record(
        {
            "trace_id": "t1",
            "raw_query": "What is a decorator?",
            "intent": "definition",
            "retrieval_strategy": "hybrid",
            "candidate_count": 20,
            "reranked_count": 5,
            "grounding_verdict": "grounded",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )

    conn = get_connection()
    row = conn.execute("SELECT * FROM trace_records WHERE trace_id = 't1'").fetchone()
    conn.close()

    assert row["raw_query"] == "What is a decorator?"
    assert row["retrieval_strategy"] == "hybrid"
    assert row["candidate_count"] == 20
    assert row["reranked_count"] == 5
    assert row["grounding_verdict"] == "grounded"
    assert row["llm_model"] is None
    assert row["rewritten_query"] is None


def test_anonymous_hub_advisory_is_filtered_but_real_errors_are_not():
    """The HF "unauthenticated requests" line is advisory and reprints on every
    model load; suppressing it must not also hide genuine Hub failures."""
    from rag_assistant.config import _AnonymousHubWarningFilter

    hub_filter = _AnonymousHubWarningFilter()

    def record(message: str) -> logging.LogRecord:
        return logging.LogRecord("x", logging.WARNING, __file__, 1, message, None, None)

    advisory = "You are sending unauthenticated requests to the HF Hub."
    assert hub_filter.filter(record(advisory)) is False
    assert hub_filter.filter(record("401 Client Error: Unauthorized for url")) is True
    assert hub_filter.filter(record("429 Too Many Requests")) is True
