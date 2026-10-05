"""The ingestion-resume test CLAUDE.md §31 requires before Phase 1 counts as
complete: interrupt after CHUNKING, restart, confirm it resumes from the
persisted artifact instead of re-parsing/re-chunking."""

import pytest

from rag_assistant.ingestion import pipeline
from rag_assistant.ingestion.state_machine import FAILED, READY, get_existing_state


def test_resume_skips_completed_stages_after_a_mid_pipeline_failure(
    isolated_storage, sample_pdf_bytes, monkeypatch
):
    parse_calls = []
    chunk_calls = []
    embed_calls = []

    real_parse = pipeline.parse_pdf
    real_chunk = pipeline.chunk_elements

    def tracking_parse(*args, **kwargs):
        parse_calls.append(1)
        return real_parse(*args, **kwargs)

    def tracking_chunk(*args, **kwargs):
        chunk_calls.append(1)
        return real_chunk(*args, **kwargs)

    def failing_embed(chunks):
        embed_calls.append(1)
        if len(embed_calls) == 1:
            raise RuntimeError("simulated crash during EMBEDDING")
        return None  # second call succeeds

    monkeypatch.setattr(pipeline, "parse_pdf", tracking_parse)
    monkeypatch.setattr(pipeline, "chunk_elements", tracking_chunk)
    monkeypatch.setattr(pipeline, "embed_and_index", failing_embed)

    # First run: crashes during EMBEDDING, after EXTRACTING/CHUNKING already
    # persisted their artifacts.
    with pytest.raises(RuntimeError):
        pipeline.ingest_pdf(sample_pdf_bytes, "notes.pdf")

    source_id = pipeline.sm.compute_source_id(sample_pdf_bytes)
    assert get_existing_state(source_id) == FAILED
    assert len(parse_calls) == 1
    assert len(chunk_calls) == 1

    # Second run (resume): same bytes -> same source_id. EXTRACTING/CHUNKING
    # artifacts already exist, so parse/chunk must NOT run again - only the
    # failed EMBEDDING stage (and beyond) re-runs.
    result_id = pipeline.ingest_pdf(sample_pdf_bytes, "notes.pdf")

    assert result_id == source_id
    assert get_existing_state(source_id) == READY
    assert len(parse_calls) == 1, "parse_pdf must not be re-run on resume"
    assert len(chunk_calls) == 1, "chunk_elements must not be re-run on resume"
    assert len(embed_calls) == 2, "embed_and_index should be retried since it failed"
