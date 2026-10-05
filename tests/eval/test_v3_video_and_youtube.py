"""Live tests for the two ingestion paths that had code but no test at all
before this pass: video (§8 - audio-track-only) and YouTube (§13's
captions-first, ASR-fallback path)."""

from pathlib import Path

import httpx
import pytest

from rag_assistant.config import settings
from rag_assistant.graph.build_graph import ask
from rag_assistant.ingestion.pipeline import ingest_file, ingest_youtube

_VIDEO_PATH = Path(__file__).parent.parent / "fixtures" / "sample_video.mp4"

# The first video ever uploaded to YouTube ("Me at the zoo", ~19s, public,
# universally known test case) - used here purely to verify the ingestion
# pipeline mechanically works end-to-end, not as a real user document.
_TEST_YOUTUBE_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def _ollama_available() -> bool:
    try:
        httpx.get(f"{settings.ollama_host}/api/tags", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _VIDEO_PATH.exists(), reason="sample_video.mp4 fixture not generated")
@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running on this machine")
def test_video_ingest_and_timestamp_citation(isolated_storage):
    source_id = ingest_file(_VIDEO_PATH.read_bytes(), "clip.mp4")
    result = ask("What is a decorator?", source_id=source_id)

    assert result["citations"], "expected at least one resolved citation"
    citation = result["citations"][0]
    assert "start_time" in citation, "video citations should carry a timestamp, like audio"


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running on this machine")
def test_youtube_ingest_produces_real_transcript_chunks(isolated_storage):
    source_id = ingest_youtube(_TEST_YOUTUBE_URL)

    from rag_assistant.storage.vector_store import get_chunks_by_source

    chunks = get_chunks_by_source(source_id)
    assert len(chunks) > 0, "expected at least one transcript chunk from the YouTube video"
    assert all(c["element_type"] == "transcript_segment" for c in chunks)
