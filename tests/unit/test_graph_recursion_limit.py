"""The most degraded graph path must still return an answer, not crash.

CLAUDE.md §7 item 24. recursion_limit was hardcoded to 15 while the worst-case
real path is 17 super-steps (rewrite + decompose + summarize all firing, then
generate/resolve/verify three times as the retry budget is spent, then
format_response). So a question that exercised the retry loop fully - exactly
the case §22.2 designed the loop for - raised GraphRecursionError instead of
returning the answer flagged "not fully verified".
"""

from rag_assistant.graph import build_graph
from rag_assistant.graph.nodes.verify_grounding import MAX_RETRIES


def test_worst_case_step_count_is_derived_from_the_retry_budget():
    """The limit must follow MAX_RETRIES, not be a number someone has to
    remember to bump."""
    assert build_graph._WORST_CASE_STEPS == 7 + 3 * (1 + MAX_RETRIES) + 1
    assert build_graph.RECURSION_LIMIT > build_graph._WORST_CASE_STEPS


def test_fully_degraded_path_completes_instead_of_raising(isolated_storage, monkeypatch):
    """Drives every optional node AND exhausts the retry budget.

    The nodes are stubbed because this test is about the graph's wiring and its
    step budget, not about retrieval or generation quality - it asserts the
    traversal completes and reports grounded=False rather than raising.
    """
    generate_calls: list[int] = []

    def fake_classify(_state):
        return {
            "classification": {
                "language": "en",
                "intent": "multi_hop",
                "response_depth": "comprehensive",
                "time_reference": None,
                "needs_rewrite": True,
                "needs_decomposition": True,
            }
        }

    def fake_retrieve(_state):
        # Large enough to push _after_expand down the summarize branch.
        big = "word " * 4000
        return {
            "retrieved": [{"chunk_id": "c1", "source_id": "s1", "text": big, "parent_id": None}],
            "retrieval_strategy": "multi_hop",
            "candidate_count": 1,
        }

    def fake_generate(_state):
        generate_calls.append(1)
        return {"answer": "An answer with an out-of-range marker [9]."}

    def fake_verify(state):
        retries = (state.get("verification") or {}).get("retries", 0)
        if retries < MAX_RETRIES:
            return {
                "verification": {"verdict": "unsupported", "retries": retries + 1, "retry": True}
            }
        return {
            "verification": {"verdict": "unsupported", "retries": retries, "retry": False},
            "grounded": False,
        }

    monkeypatch.setattr(build_graph, "classify_node", fake_classify)
    monkeypatch.setattr(build_graph, "rewrite_node", lambda s: {"standalone_query": "q"})
    monkeypatch.setattr(
        build_graph,
        "decompose_node",
        lambda s: {"sub_queries": [{"order": 0, "query": "q", "depends_on": []}]},
    )
    monkeypatch.setattr(build_graph, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(build_graph, "rerank_node", lambda s: {"reranked_count": 1})
    monkeypatch.setattr(build_graph, "expand_context_node", lambda s: {})
    monkeypatch.setattr(build_graph, "summarize_node", lambda s: {})
    monkeypatch.setattr(build_graph, "generate_node", fake_generate)
    monkeypatch.setattr(
        build_graph,
        "resolve_citations_node",
        lambda s: {
            "citations": [],
            "grounded": False,
            "citation_check": {"markers_found": True, "out_of_range": True, "repointed": 0},
        },
    )
    monkeypatch.setattr(build_graph, "verify_grounding_node", fake_verify)
    monkeypatch.setattr(build_graph, "format_response_node", lambda s: {"formatted_answer": None})

    result = build_graph.ask("a compound question about two things")

    # 1 initial attempt + MAX_RETRIES regenerations - the full degraded path.
    assert len(generate_calls) == 1 + MAX_RETRIES
    assert result["grounded"] is False
    assert result["verification"]["retry"] is False
