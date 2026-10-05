"""Temporal retrieval must find segments that CONTAIN the asked-about moment.

CLAUDE.md §7 items 31 and 32. Both defects here were invisible on the short test
fixture and only bite on real recordings, where batched ASR produces ~25s
segments (measured up to 31s).
"""

from rag_assistant.graph.nodes import retrieval as r
from rag_assistant.ingestion.chunking import chunk_elements
from rag_assistant.ingestion.models import ContentElement
from rag_assistant.storage.vector_store import search_temporal, upsert_chunks

# Temporal lookup is a pure metadata scroll - it never reads a vector. Indexing
# with placeholders instead of real BGE-M3 output keeps these in tests/unit
# ("seconds") rather than dragging a 1.2GB model load into the fast suite.
_DUMMY_DENSE = [0.0] * 1024
_DUMMY_SPARSE = {"indices": [1], "values": [1.0]}


def _index_transcript(source_id: str, spans: list[tuple[float, float]]) -> None:
    elements = [
        ContentElement(
            element_type="transcript_segment",
            text=f"speech from {start:.0f} to {end:.0f} seconds",
            start_time=start,
            end_time=end,
        )
        for start, end in spans
    ]
    chunks = chunk_elements(elements, source_id)
    upsert_chunks(
        [c.model_dump() for c in chunks],
        [_DUMMY_DENSE] * len(chunks),
        [_DUMMY_SPARSE] * len(chunks),
    )


def test_segment_starting_before_the_window_but_containing_the_moment_is_found(
    isolated_storage,
):
    """The real bug: filtering on start_time alone. A segment spanning
    1500-1800s contains second 1740, but its START is outside a [1560, 1920]
    window, so it was excluded - exactly the segment the user asked about."""
    _index_transcript("src", [(1500.0, 1800.0), (60.0, 85.0)])

    found = search_temporal(["src"], center_seconds=1740.0, window_seconds=180)

    spans = {(f["start_time"], f["end_time"]) for f in found}
    assert (1500.0, 1800.0) in spans, "segment containing the target moment was missed"
    assert (60.0, 85.0) not in spans, "unrelated distant segment should not match"


def test_results_are_ordered_nearest_first(isolated_storage):
    """scroll() returns an arbitrary page, so on a long recording the results
    were not the closest ones to the asked-about time."""
    _index_transcript(
        "src",
        [(900.0, 925.0), (1730.0, 1755.0), (1600.0, 1625.0), (1000.0, 1025.0)],
    )

    found = search_temporal(["src"], center_seconds=1740.0, window_seconds=900, limit=3)

    starts = [f["start_time"] for f in found]
    assert starts[0] == 1730.0, f"nearest segment should come first, got {starts}"
    assert starts == sorted(starts, key=lambda s: abs(s - 1740.0))


def test_limit_is_respected(isolated_storage):
    _index_transcript("src", [(float(i * 20), float(i * 20 + 20)) for i in range(40)])

    found = search_temporal(["src"], center_seconds=400.0, window_seconds=10_000, limit=5)

    assert len(found) == 5


def test_multi_hop_shares_the_candidate_budget_instead_of_multiplying_it(monkeypatch):
    """Reranking is the most expensive step in a turn; fan-out must not scale
    its cost with however many sub-queries the decomposer produced."""
    requested: list[int] = []

    def fake_hybrid(query, top_k, source_id, element_type=None):
        requested.append(top_k)
        return [{"chunk_id": f"{query}-{top_k}", "text": "t"}]

    monkeypatch.setattr(r, "_hybrid", fake_hybrid)

    r.retrieve_node(
        {
            "question": "q",
            "classification": {},
            "sub_queries": [
                {"order": 0, "query": "a", "depends_on": []},
                {"order": 1, "query": "b", "depends_on": []},
                {"order": 2, "query": "c", "depends_on": []},
                {"order": 3, "query": "d", "depends_on": []},
                {"order": 4, "query": "e", "depends_on": []},
            ],
        }
    )

    # Capped at MAX_SUB_QUERIES, and the budget is divided, not multiplied.
    assert len(requested) == r.MAX_SUB_QUERIES
    assert sum(requested) <= r.TOP_K
