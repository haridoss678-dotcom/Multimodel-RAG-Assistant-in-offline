"""The exact 'Phase 1 done' test from CLAUDE.md §33: ingest one real PDF,
ask one real question, assert the answer cites the correct page."""

import httpx
import pytest

from rag_assistant.config import settings
from rag_assistant.graph.build_graph import ask
from rag_assistant.ingestion.pipeline import ingest_pdf


def _ollama_available() -> bool:
    try:
        httpx.get(f"{settings.ollama_host}/api/tags", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running on this machine")
def test_v1_end_to_end_citation_is_correct(isolated_storage, sample_pdf_bytes):
    source_id = ingest_pdf(sample_pdf_bytes, "python_notes.pdf")

    result = ask("What is a decorator?", source_id=source_id)

    assert result["citations"], "expected at least one resolved citation"
    assert result["grounded"] is True
    # The decorator fact lives on page 2 of the fixture PDF (conftest.py) -
    # this is the concrete "correct page citation" check, not just "an answer came back".
    assert any(c["page"] == 2 for c in result["citations"])
