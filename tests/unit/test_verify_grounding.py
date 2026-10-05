from rag_assistant.graph.nodes import verify_grounding as vg


def test_true_abstention_with_no_evidence_skips_the_llm_check(monkeypatch):
    """Nothing retrieved means generate_node returned its canned "couldn't find
    anything" line - the correct outcome per CLAUDE.md §3. Verifying and
    regenerating it cost ~43s of a measured 112s turn and could never change the
    verdict, so no LLM call should be made and no retry attempted."""
    called = []
    monkeypatch.setattr(vg, "chat_structured", lambda *a, **k: called.append(1) or {})

    result = vg.verify_grounding_node(
        {
            "answer": "I couldn't find anything about DC in your documents.",
            "retrieved": [],
            "citation_check": {"markers_found": False, "out_of_range": False, "repointed": 0},
        }
    )

    assert called == []
    assert result["grounded"] is False
    assert result["verification"]["retry"] is False


def test_missing_markers_after_failed_derivation_is_not_retried(monkeypatch):
    called = []
    monkeypatch.setattr(vg, "chat_structured", lambda *a, **k: called.append(1) or {})

    result = vg.verify_grounding_node(
        {
            "answer": "Something the evidence does not support.",
            "retrieved": [{"source_id": "s1", "text": "entirely unrelated text"}],
            "citation_check": {
                "markers_found": False,
                "out_of_range": False,
                "repointed": 0,
                "derived": 0,
                "unmatched_sentences": 1,
            },
        }
    )

    assert called == [], "no LLM call - there is nothing to verify"
    assert result["verification"]["retry"] is False
    assert result["grounded"] is False


def test_a_clean_deterministic_result_skips_the_semantic_check(monkeypatch):
    """The second opinion costs 30-60s of EVERY turn and comes from a model
    §22.2 already records passing a mis-numbered citation as grounded - the one
    failure it existed to catch. When the deterministic layer found nothing to
    doubt, that call is the unnecessary one §9 says to skip."""
    called = []
    monkeypatch.setattr(vg, "chat_structured", lambda *a, **k: called.append(1) or {})

    result = vg.verify_grounding_node(
        {
            "answer": "Skills include Python and SQL [1].",
            "retrieved": [{"source_id": "s1", "text": "Skills include Python and SQL"}],
            "grounded": True,
            "citation_check": {
                "markers_found": True,
                "out_of_range": False,
                "repointed": 0,
                "derived": 1,
                "unmatched_sentences": 0,
                "derived_coverage": 0.95,
            },
        }
    )

    assert called == []
    assert result["verification"]["verdict"] == "grounded"
    assert result["verification"]["verified_by"] == "deterministic-attribution"
    assert result["grounded"] is True


def test_doubt_in_the_deterministic_result_still_gets_the_semantic_check(monkeypatch):
    """An unmatched sentence means part of the answer went beyond the
    evidence - that is precisely when the second opinion is worth its cost."""
    called = []

    def fake(*_a, **_k):
        called.append(1)
        return {"claims": [{"text": "x", "citation": 1, "supported": True}]}

    monkeypatch.setattr(vg, "chat_structured", fake)

    vg.verify_grounding_node(
        {
            "answer": "Supported bit [1]. Unsupported bit.",
            "retrieved": [{"source_id": "s1", "text": "Supported bit"}],
            "citation_check": {
                "markers_found": True,
                "out_of_range": False,
                "repointed": 0,
                "derived": 1,
                "unmatched_sentences": 1,
                "derived_coverage": 0.9,
            },
        }
    )

    assert called == [1], "a partly-unmatched answer must still be verified"


def test_unsupported_verdict_gives_up_once_retries_are_exhausted(monkeypatch):
    """The retry budget still bounds the SEMANTIC path - an answer the verifier
    keeps rejecting is returned flagged, not looped on forever."""
    monkeypatch.setattr(
        vg,
        "chat_structured",
        lambda *a, **k: {"claims": [{"text": "x", "citation": 1, "supported": False}]},
    )

    result = vg.verify_grounding_node(
        {
            "answer": "Inventory is tracked by batch [1].",
            "retrieved": [{"source_id": "s1", "text": "items are tracked based on batches"}],
            "citation_check": {"markers_found": True, "out_of_range": False, "repointed": 0},
            "verification": {"retries": vg.MAX_RETRIES},
        }
    )

    assert result["verification"]["retry"] is False
    assert result["grounded"] is False


def test_invalid_marker_still_gets_the_full_loop(monkeypatch):
    """Markers that exist but point out of range are a different failure: a
    regeneration can genuinely fix those, so they must not be short-circuited."""
    monkeypatch.setattr(
        vg,
        "chat_structured",
        lambda *a, **k: {"claims": [{"text": "x", "citation": 9, "supported": False}]},
    )

    result = vg.verify_grounding_node(
        {
            "answer": "Something claimed here [9].",
            "retrieved": [{"source_id": "s1", "text": "evidence"}],
            "citation_check": {"markers_found": True, "out_of_range": True, "repointed": 0},
        }
    )

    assert result["verification"]["retry"] is True
    assert result["verification"]["retries"] == 1


def test_supported_claims_resolve_to_grounded(monkeypatch):
    monkeypatch.setattr(
        vg,
        "chat_structured",
        lambda *a, **k: {
            "claims": [{"text": "x", "citation": 1, "supported": True}],
            "overall_verdict": "unsupported",  # model's own field - deliberately ignored
        },
    )

    result = vg.verify_grounding_node(
        {
            "answer": "A decorator wraps a function [1].",
            "retrieved": [{"source_id": "s1", "text": "a decorator wraps a function"}],
            "citation_check": {"markers_found": True, "out_of_range": False, "repointed": 0},
            "grounded": True,
        }
    )

    assert result["verification"]["verdict"] == "grounded"
    assert result["grounded"] is True


def test_a_repointed_marker_does_not_trigger_the_semantic_check(monkeypatch):
    """A re-point is a problem code ALREADY SOLVED.

    The citation now points at the item that actually supports the sentence -
    more trustworthy than what the model wrote, not less. Treating a completed
    correction as unresolved doubt sent every bulleted answer (where
    re-pointing is routine) down the verify-and-retry path, measured at
    273-300s per turn (CLAUDE.md §7 item 55)."""
    called = []
    monkeypatch.setattr(vg, "chat_structured", lambda *a, **k: called.append(1) or {})

    result = vg.verify_grounding_node(
        {
            "answer": "A claim [1].",
            "retrieved": [{"source_id": "s1", "text": "some evidence"}],
            "grounded": True,
            "citation_check": {
                "markers_found": True,
                "out_of_range": False,
                "repointed": 1,
                "derived": 0,
                "unmatched_sentences": 0,
            },
        }
    )

    assert called == [], "a corrected citation is not a reason to pay for a second opinion"
    assert result["verification"]["verdict"] == "grounded"


def test_an_out_of_range_marker_still_gets_the_semantic_check(monkeypatch):
    called = []

    def fake(*_a, **_k):
        called.append(1)
        return {"claims": [{"text": "x", "citation": 1, "supported": False}]}

    monkeypatch.setattr(vg, "chat_structured", fake)

    vg.verify_grounding_node(
        {
            "answer": "A claim [9].",
            "retrieved": [{"source_id": "s1", "text": "some evidence"}],
            "citation_check": {
                "markers_found": True,
                "out_of_range": True,
                "repointed": 0,
                "derived": 0,
                "unmatched_sentences": 0,
            },
        }
    )

    assert called == [1]
