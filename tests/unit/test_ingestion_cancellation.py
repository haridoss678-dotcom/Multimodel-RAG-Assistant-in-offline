"""Deleting a source mid-ingestion must leave nothing behind.

CLAUDE.md §7 item 35. Two defects, fixed together because they are one bug:
  - DELETE /sources/{id} did not signal the running background job, which kept
    going and re-created the job row and vectors AFTER the cascade removed them;
  - state_machine defined a CANCELLED state that nothing ever set - dead code by
    §8 rule 1.
"""

import pytest

from rag_assistant.ingestion import pipeline as p
from rag_assistant.ingestion import state_machine as sm
from rag_assistant.storage.db import get_connection


def test_is_cancelled_covers_both_deleted_and_cancelled(isolated_storage):
    sm.register_source("s1", "a.pdf", "pdf")
    assert sm.is_cancelled("s1") is False

    sm.request_cancel("s1")
    assert sm.is_cancelled("s1") is True
    assert sm.get_existing_state("s1") == sm.CANCELLED

    # A source that was deleted outright is also "cancelled" from a worker's
    # point of view - that is the case that actually produced orphans.
    assert sm.is_cancelled("never-existed") is True


def test_update_job_does_not_resurrect_a_deleted_sources_row(isolated_storage):
    """The orphan mechanism: _update_job is INSERT ... ON CONFLICT DO UPDATE, so
    a worker that outlived its source re-created the job row every stage."""
    sm.register_source("s1", "a.pdf", "pdf")
    p._update_job("s1", sm.CHUNKING, 0.6)

    conn = get_connection()
    conn.execute("DELETE FROM jobs WHERE source_id = 's1'")
    conn.execute("DELETE FROM sources WHERE source_id = 's1'")
    conn.commit()
    conn.close()

    p._update_job("s1", sm.EMBEDDING, 0.8)  # the still-running worker

    conn = get_connection()
    rows = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE source_id = 's1'").fetchone()["n"]
    conn.close()
    assert rows == 0, "deleted source's job row was re-created by the worker"


def test_pipeline_aborts_at_a_stage_boundary_when_cancelled(isolated_storage):
    sm.register_source("s1", "a.pdf", "pdf")
    sm.request_cancel("s1")

    with pytest.raises(p.IngestionCancelled):
        p._abort_if_cancelled("s1")


def test_cancelled_source_is_not_marked_failed(isolated_storage, monkeypatch, sample_pdf_bytes):
    """A cancelled ingestion is not a failure. Marking it FAILED would both
    mislead the UI and re-create the row the cascade removed."""
    source_id = sm.compute_source_id(sample_pdf_bytes)

    real_chunk_stage = p._run_extract_through_chunk

    def cancel_then_continue(*args, **kwargs):
        # Simulate the user pressing Delete while extraction is running.
        sm.request_cancel(source_id)
        return real_chunk_stage(*args, **kwargs)

    monkeypatch.setattr(p, "_run_extract_through_chunk", cancel_then_continue)

    with pytest.raises(p.IngestionCancelled):
        p.ingest_file(sample_pdf_bytes, "a.pdf")

    assert sm.get_existing_state(source_id) == sm.CANCELLED
