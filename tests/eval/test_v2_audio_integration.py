"""V2's audio-path equivalent of the V1 PDF citation test: ingest a real
short speech clip, ask a question, assert a timestamp citation (not a page)."""

from pathlib import Path

import httpx
import pytest

from rag_assistant.config import settings
from rag_assistant.graph.build_graph import ask
from rag_assistant.ingestion.pipeline import ingest_file

_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample_speech.wav"


def _ollama_available() -> bool:
    try:
        httpx.get(f"{settings.ollama_host}/api/tags", timeout=3)
        return True
    except Exception:
        return False


@pytest.fixture
def sample_speech_wav():
    """A real synthesized speech clip (Windows SAPI), saying "Decorators are
    functions that wrap other functions to extend their behavior." - not a
    mock. Generated once via `tests/fixtures/generate_sample_speech.ps1`."""
    return _FIXTURE_PATH.read_bytes()


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running on this machine")
def test_v2_audio_end_to_end_timestamp_citation(isolated_storage, sample_speech_wav):
    source_id = ingest_file(sample_speech_wav, "lecture.wav")

    result = ask("What is a decorator?", source_id=source_id)

    assert result["citations"], "expected at least one resolved citation"
    assert result["grounded"] is True
    citation = result["citations"][0]
    assert "start_time" in citation, "audio citations must carry a timestamp, not a page"
    assert citation.get("page") is None
