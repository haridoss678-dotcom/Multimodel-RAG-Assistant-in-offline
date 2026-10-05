"""A type-narrowed search must not EXCLUDE the source the question is about.

CLAUDE.md §7 item 58. `code_lookup` ran the hybrid search with
element_type="code" and, when that matched anything, returned it INSTEAD of the
unfiltered result. The victim is specific and real: a video whose code is only
ever explained aloud has no element_type="code" chunk at all - every transcript
chunk is a transcript_segment - so as soon as the corpus also held a PDF with a
fenced code block, "explain the code from the video" matched the PDF's code,
`typed` came back non-empty, and the video was excluded outright. The documented
empty-result fallback never fired, because the filter DID find something - just
not from the source being asked about.
"""

from rag_assistant.graph.nodes import retrieval as r
from rag_assistant.ingestion.chunking import chunk_elements
from rag_assistant.ingestion.models import ContentElement
from rag_assistant.storage.vector_store import upsert_chunks

_DENSE = [0.1] * 1024
_SPARSE = {"indices": [1], "values": [1.0]}


def _index(source_id: str, elements: list[ContentElement]) -> None:
    chunks = chunk_elements(elements, source_id)
    upsert_chunks([c.model_dump() for c in chunks], [_DENSE] * len(chunks), [_SPARSE] * len(chunks))


def _state(intent: str, question: str) -> dict:
    return {
        "question": question,
        "classification": {"intent": intent, "response_depth": "normal"},
    }


def test_a_pdfs_code_block_does_not_hide_a_videos_spoken_explanation(isolated_storage, monkeypatch):
    monkeypatch.setattr(r, "embed_query", lambda q: (_DENSE, _SPARSE))
    _index("pdfsrc", [ContentElement(element_type="paragraph", text="```\nprint(1)\n```", page=1)])
    _index(
        "vidsrc",
        [
            ContentElement(
                element_type="transcript_segment",
                text="the post request is at app.post and we import the Pydantic base model",
                start_time=700.0,
                end_time=730.0,
            )
        ],
    )

    out = r.retrieve_node(_state("code_lookup", "explain the code for the post API"))
    sources = {c["source_id"] for c in out["retrieved"]}

    assert "pdfsrc" in sources, "a real fenced code block must still be retrieved"
    assert "vidsrc" in sources, "the spoken explanation must not be filtered away"
    assert out["retrieval_strategy"] == "hybrid+code"


def test_the_code_block_still_outranks_narration_into_the_reranker(isolated_storage, monkeypatch):
    """Merged, not shuffled: typed hits go first so a genuine code block is
    still the reranker's leading candidate."""
    monkeypatch.setattr(r, "embed_query", lambda q: (_DENSE, _SPARSE))
    _index("pdfsrc", [ContentElement(element_type="paragraph", text="```\nprint(1)\n```", page=1)])
    _index(
        "vidsrc",
        [
            ContentElement(
                element_type="transcript_segment",
                text="we talked about printing things",
                start_time=1.0,
                end_time=9.0,
            )
        ],
    )

    out = r.retrieve_node(_state("code_lookup", "show me the code"))
    assert out["retrieved"][0]["element_type"] == "code"


def test_a_corpus_with_no_code_at_all_still_degrades_to_plain_hybrid(isolated_storage, monkeypatch):
    """The original fallback, preserved: a filter artifact must never be
    reported as "not found"."""
    monkeypatch.setattr(r, "embed_query", lambda q: (_DENSE, _SPARSE))
    _index(
        "vidsrc",
        [
            ContentElement(
                element_type="transcript_segment",
                text="we import the Pydantic base model here",
                start_time=1.0,
                end_time=9.0,
            )
        ],
    )

    out = r.retrieve_node(_state("code_lookup", "show me the code"))
    assert out["retrieval_strategy"] == "hybrid"
    assert out["retrieved"], "narration must still be retrievable when no code chunk exists"
