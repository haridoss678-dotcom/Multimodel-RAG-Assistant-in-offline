"""Parent expansion must add context without destroying citation precision.

CLAUDE.md §7 items 27 and 28. Two defects lived here, each harmless-looking on
its own:
  - a reassembled PDF page came back in uuid order, not document order;
  - expanding all five reranked hits routinely pushed evidence past the
    summarizer threshold, and the summarizer collapses a batch onto its FIRST
    chunk's page - so the feature that creates page-exact citations was what
    triggered the step that destroys them.
"""

from rag_assistant.graph.nodes import context_expansion as ce
from rag_assistant.graph.nodes.citation import resolve_citations_node
from rag_assistant.ingestion.chunking import CHUNK_OVERLAP


def _chunk(cid: str, text: str, index: int, page: int = 7, parent: str = "src:page:7") -> dict:
    return {
        "chunk_id": cid,
        "source_id": "src",
        "text": text,
        "page": page,
        "parent_id": parent,
        "chunk_index": index,
        "element_type": "paragraph",
    }


def test_page_is_reassembled_in_document_order_not_uuid_order(monkeypatch):
    """chunk_index is the ordering key. Before it existed, every chunk on a page
    had start_time None -> 0 and the tiebreak fell through to chunk_id, a uuid4,
    so the model was handed a shuffled page."""
    siblings = [
        _chunk("zzz-uuid-sorts-last", "First sentence. ", 0),
        _chunk("aaa-uuid-sorts-first", "Second sentence. ", 1),
        _chunk("mmm-uuid-middle", "Third sentence.", 2),
    ]
    monkeypatch.setattr(ce, "get_by_parent_id", lambda _pid: siblings)

    out = ce.expand_context_node({"retrieved": [siblings[1]]})

    # Non-overlapping chunks are newline-separated; what matters is the ORDER.
    assert out["retrieved"][0]["text"].split("\n") == [
        "First sentence. ",
        "Second sentence. ",
        "Third sentence.",
    ]


def test_chunk_overlap_is_not_duplicated_in_the_merged_parent(monkeypatch):
    """_split_prose repeats CHUNK_OVERLAP characters between adjacent chunks.
    Joining them naively put every boundary in twice, spending context budget on
    text the model has already read."""
    tail = "X" * CHUNK_OVERLAP
    siblings = [
        _chunk("c0", "AAAA" + tail, 0),
        _chunk("c1", tail + "BBBB", 1),
    ]
    monkeypatch.setattr(ce, "get_by_parent_id", lambda _pid: siblings)

    merged = ce.expand_context_node({"retrieved": [siblings[0]]})["retrieved"][0]["text"]

    assert merged == "AAAA" + tail + "BBBB"
    assert merged.count(tail) == 1


def test_expansion_stops_at_the_budget_and_keeps_the_precise_match(monkeypatch):
    """Over budget, the matched chunk keeps its own text rather than being
    expanded into an overflow that the summarizer would then collapse."""
    huge = [_chunk(f"c{i}", "z" * 5000, i) for i in range(10)]
    matched = _chunk("match", "the precise sentence that matched", 0)
    monkeypatch.setattr(ce, "get_by_parent_id", lambda _pid: huge)
    monkeypatch.setattr(ce, "max_evidence_chars", lambda: 2000)

    out = ce.expand_context_node({"retrieved": [matched]})

    assert out["retrieved"][0]["text"] == "the precise sentence that matched"


def test_expansion_never_moves_the_citation_off_the_matched_chunk(monkeypatch):
    """The T1.5 acceptance property: for an ordinary question the citation must
    resolve to the page of the chunk that actually matched, not to the parent's
    first chunk and not to a summarized batch's first chunk."""
    page_5_match = _chunk(
        "hit", "landed cost is apportioned per batch", 3, page=5, parent="src:page:5"
    )
    siblings = [
        _chunk("s0", "unrelated opening text ", 0, page=5, parent="src:page:5"),
        page_5_match,
    ]
    monkeypatch.setattr(ce, "get_by_parent_id", lambda _pid: siblings)

    expanded = ce.expand_context_node({"retrieved": [page_5_match]})["retrieved"]

    # Text grew to the whole page...
    assert "unrelated opening text" in expanded[0]["text"]
    # ...but the citation metadata did not move.
    assert expanded[0]["page"] == 5
    assert expanded[0]["chunk_id"] == "hit"

    resolved = resolve_citations_node(
        {"answer": "Landed cost is apportioned per batch [1].", "retrieved": expanded}
    )
    assert resolved["citations"][0]["page"] == 5
    assert resolved["grounded"] is True
