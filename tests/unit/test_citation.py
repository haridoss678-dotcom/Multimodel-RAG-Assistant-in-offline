from rag_assistant.graph.nodes.citation import resolve_citations_node


def test_valid_citation_resolves(isolated_storage):
    from rag_assistant.storage.db import get_connection

    conn = get_connection()
    conn.execute(
        "INSERT INTO sources (source_id, filename, source_type, state, created_at, updated_at) "
        "VALUES ('src1', 'notes.pdf', 'pdf', 'READY', datetime('now'), datetime('now'))"
    )
    conn.commit()
    conn.close()

    state = {
        "answer": "Decorators wrap functions [1].",
        "retrieved": [{"source_id": "src1", "page": 2, "text": "..."}],
    }
    result = resolve_citations_node(state)
    assert result["grounded"] is True
    assert result["citations"] == [
        {"marker": 1, "source_id": "src1", "filename": "notes.pdf", "page": 2}
    ]


def test_invented_citation_index_is_rejected(isolated_storage):
    state = {
        "answer": "This claims something [7].",
        "retrieved": [{"source_id": "src1", "page": 1, "text": "..."}],
    }
    result = resolve_citations_node(state)
    assert result["grounded"] is False
    assert result["citations"] == []


def test_no_citation_markers_is_not_grounded(isolated_storage):
    state = {
        "answer": "An answer with no markers at all.",
        "retrieved": [{"source_id": "src1", "page": 1, "text": "x"}],
    }
    result = resolve_citations_node(state)
    assert result["grounded"] is False


def _register_source(source_id: str = "src1", filename: str = "notes.pdf") -> None:
    from rag_assistant.storage.db import get_connection

    conn = get_connection()
    conn.execute(
        "INSERT INTO sources (source_id, filename, source_type, state, created_at, updated_at) "
        "VALUES (?, ?, 'pdf', 'READY', datetime('now'), datetime('now'))",
        (source_id, filename),
    )
    conn.commit()
    conn.close()


def test_mis_numbered_marker_resolves_to_the_chunk_the_text_came_from(isolated_storage):
    """Regression for a measured failure, not a hypothetical one: on the same
    question, qwen3:0.6b produced the correct sentence from evidence [1] but
    wrote "[2]" after it on roughly 40% of runs, yielding a real-but-wrong page
    citation that the LLM grounding verifier passed as "grounded"."""
    _register_source()
    state = {
        "answer": (
            "A decorator is a function that wraps another function to extend its behavior. [2]"
        ),
        "retrieved": [
            {
                "source_id": "src1",
                "page": 2,
                "text": (
                    "Decorators. A decorator is a function that wraps another function "
                    "to extend its behavior without modifying it directly."
                ),
            },
            {
                "source_id": "src1",
                "page": 1,
                "text": "Introduction to Python. This document introduces basic Python concepts.",
            },
        ],
    }
    result = resolve_citations_node(state)

    assert result["citation_check"]["repointed"] == 1
    assert [c["page"] for c in result["citations"]] == [2]
    # The marker in the prose is left alone; only what it resolves to is fixed.
    assert result["citations"][0]["marker"] == 2


def test_ambiguous_chunks_keep_the_model_s_own_marker(isolated_storage):
    """The attribution check only corrects the clear-cut case. Two chunks that
    both plausibly account for the sentence must not be second-guessed."""
    _register_source()
    shared = "A decorator is a function that wraps another function to extend its behavior."
    state = {
        "answer": f"{shared} [2]",
        "retrieved": [
            {"source_id": "src1", "page": 2, "text": shared},
            {"source_id": "src1", "page": 7, "text": shared},
        ],
    }
    result = resolve_citations_node(state)

    assert result["citation_check"]["repointed"] == 0
    assert [c["page"] for c in result["citations"]] == [7]


def test_markers_are_derived_when_the_model_writes_none():
    """CLAUDE.md §7 item 50 - on a real resume PDF the model emitted zero [n]
    markers on every attempt, so the retry loop burned three generate calls
    (3m16s for one short question) and still returned grounded=False over an
    answer taken straight from the evidence. Code supplies the markers now."""
    state = {
        "answer": (
            "The skills include Python, JavaScript and SQL. "
            "Projects used faster-whisper and pyannote for transcription."
        ),
        "retrieved": [
            {"source_id": "s1", "text": "Skills: Python, JavaScript, SQL, HTML and CSS."},
            {"source_id": "s1", "text": "Projects used faster-whisper and pyannote heavily."},
        ],
    }

    result = resolve_citations_node(state)

    assert result["citation_check"]["derived"] == 2
    assert result["citation_check"]["unmatched_sentences"] == 0
    assert result["grounded"] is True
    assert {c["marker"] for c in result["citations"]} == {1, 2}
    # The markers are in the returned answer, so what the reader sees and what
    # the citations point at are the same text.
    assert "[1]" in result["answer"] and "[2]" in result["answer"]


def test_derivation_refuses_to_cite_a_sentence_the_evidence_does_not_support():
    """The safety property: this must not launder a hallucination into a
    citation. A sentence that matches nothing stays bare and is counted."""
    state = {
        "answer": "Skills include Python and SQL. The author won a Nobel Prize in 1997.",
        "retrieved": [{"source_id": "s1", "text": "Skills include Python and SQL."}],
    }

    result = resolve_citations_node(state)

    assert result["citation_check"]["derived"] == 1
    assert result["citation_check"]["unmatched_sentences"] == 1
    assert "Nobel Prize in 1997. [" not in result["answer"]


def test_a_model_written_marker_is_left_completely_alone():
    """Derivation only runs when there are no markers at all - it must never
    second-guess an answer the model cited itself."""
    state = {
        "answer": "Skills include Python and SQL [1].",
        "retrieved": [{"source_id": "s1", "text": "Skills include Python and SQL."}],
    }

    result = resolve_citations_node(state)

    assert result["answer"] == "Skills include Python and SQL [1]."
    assert result["citation_check"]["derived"] == 0


def test_a_bulleted_answer_gets_cited():
    """CLAUDE.md §7 item 56 - markdown bullets usually have no full stop, so
    without MULTILINE `$` only matched end-of-STRING and every unterminated
    line was skipped outright. A whole skills list came back with zero
    citations and grounded=False despite being verbatim from the resume."""
    state = {
        "answer": (
            "Here are the skills:\n"
            "- **Programming**: Python, JavaScript, SQL, HTML, CSS\n"
            "- **ML & Data**: Scikit-learn, XGBoost, TensorFlow\n"
        ),
        "retrieved": [
            {
                "source_id": "s1",
                "text": "Programming: Python, JavaScript, SQL, HTML, CSS. "
                "ML and Data: Scikit-learn, XGBoost, TensorFlow.",
            }
        ],
    }

    result = resolve_citations_node(state)

    assert result["citation_check"]["derived"] == 2
    assert result["citation_check"]["unmatched_sentences"] == 0
    assert result["grounded"] is True


def test_a_marker_is_never_inserted_inside_an_email_or_version():
    """A terminator only ends a sentence when whitespace or end-of-line
    follows. Without that the split fired inside the address and produced
    "haridoss678@gmail. [1]com" - corrupted text AND a phantom unmatched
    fragment that forced the slow verify-and-retry path (§7 item 56)."""
    state = {
        "answer": "The email is haridoss678@gmail.com and the version is 3.11 exactly.",
        "retrieved": [
            {"source_id": "s1", "text": "Contact haridoss678@gmail.com, version 3.11 exactly."}
        ],
    }

    result = resolve_citations_node(state)

    assert "haridoss678@gmail.com" in result["answer"]
    assert "gmail. [" not in result["answer"]
    assert "3.11" in result["answer"]


def test_a_hedge_about_the_evidence_is_not_an_unsupported_claim():
    """ "No other X are mentioned in the evidence" asserts an ABSENCE, which no
    quotation can support. Counting it as unattributable sent an otherwise
    perfect answer down the verify-and-retry path - 108s against 35s for the
    same question without the hedge (CLAUDE.md §7 item 56)."""
    state = {
        "answer": (
            "The candidate email address is haridoss678@gmail.com. "
            "No other email addresses are mentioned in the evidence."
        ),
        "retrieved": [{"source_id": "s1", "text": "Contact haridoss678@gmail.com for details."}],
    }

    result = resolve_citations_node(state)

    assert result["citation_check"]["derived"] == 1
    assert result["citation_check"]["unmatched_sentences"] == 0, "the hedge must not count"
    assert result["grounded"] is True


def test_a_genuinely_unsupported_claim_is_still_counted():
    """The hedge rule must not become a blanket excuse - a real assertion the
    evidence does not contain still has to register as unattributable."""
    state = {
        "answer": "Skills include Python and SQL. The author won a Nobel Prize in 1997.",
        "retrieved": [{"source_id": "s1", "text": "Skills include Python and SQL."}],
    }

    result = resolve_citations_node(state)

    assert result["citation_check"]["unmatched_sentences"] == 1
