from rag_assistant.ingestion import diarization as d


def test_missing_token_skips_without_raising(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(d, "_pipeline", None)

    assert d.diarize("anything.wav") is None


def test_pipeline_failure_degrades_instead_of_failing_ingestion(monkeypatch):
    """CLAUDE.md §27: losing speaker labels degrades one feature; losing the
    transcript means the source can never answer anything. A gated-repo 403
    used to propagate and fail the whole ingestion - a real user would wait
    minutes on a long recording and end at FAILED with no transcript."""

    def boom():
        raise RuntimeError("403 Client Error. Cannot access gated repo")

    monkeypatch.setattr(d, "_get_pipeline", boom)

    assert d.diarize("anything.wav") is None


def test_speakers_are_numbered_by_who_speaks_first():
    """pyannote labels clusters SPEAKER_00/01 in arbitrary order; the reader
    expects "Speaker 1" to be the first voice heard."""
    turns = {
        (10.0, 12.0): "SPEAKER_01",
        (0.0, 5.0): "SPEAKER_03",  # speaks first despite the higher cluster id
        (20.0, 22.0): "SPEAKER_01",
        (30.0, 33.0): "SPEAKER_00",
    }

    relabelled = d._relabel_by_first_appearance(turns)

    assert relabelled[(0.0, 5.0)] == "Speaker 1"
    assert relabelled[(10.0, 12.0)] == "Speaker 2"
    assert relabelled[(20.0, 22.0)] == "Speaker 2"  # same person keeps the label
    assert relabelled[(30.0, 33.0)] == "Speaker 3"


def test_single_speaker_needs_no_label():
    turns = {(0.0, 5.0): "Speaker 1", (6.0, 9.0): "Speaker 1"}

    assert d.speaker_span(turns, 0.0, 9.0) == (None, False)


def test_dominant_speaker_wins_not_whoever_started_the_segment():
    """The measured bug (CLAUDE.md §7 item 29): a 25s batched segment is stamped
    with whoever spoke at its START, so a brief opener captured the whole
    segment. 67 of 68 segments on the real recording spanned >1 turn."""
    turns = {
        (0.0, 2.0): "Speaker 1",  # brief opener - used to win on start time
        (2.0, 25.0): "Speaker 2",  # actually holds the segment
    }

    speaker, multi = d.speaker_span(turns, 0.0, 25.0)

    assert speaker == "Speaker 2"
    assert multi is True


def test_second_speaker_above_the_threshold_is_flagged():
    turns = {(0.0, 20.0): "Speaker 1", (20.0, 25.0): "Speaker 2"}

    assert d.speaker_span(turns, 0.0, 25.0) == ("Speaker 1", True)


def test_momentary_second_speaker_below_the_threshold_is_not_flagged():
    """Sub-second overlap is a diarization boundary artefact, not an utterance -
    flagging it would suppress attribution on every segment for nothing."""
    turns = {(0.0, 24.5): "Speaker 1", (24.5, 25.0): "Speaker 2"}

    speaker, multi = d.speaker_span(turns, 0.0, 25.0)

    assert speaker == "Speaker 1"
    assert multi is False


def test_segment_outside_every_turn_gets_no_speaker():
    turns = {(0.0, 5.0): "Speaker 1", (6.0, 9.0): "Speaker 2"}

    assert d.speaker_span(turns, 50.0, 60.0) == (None, False)


def test_missing_timestamps_are_handled():
    turns = {(0.0, 5.0): "Speaker 1", (6.0, 9.0): "Speaker 2"}

    assert d.speaker_span(turns, None, None) == (None, False)
    # end_time absent - degrade to a point lookup rather than crash
    assert d.speaker_span(turns, 1.0, None)[0] == "Speaker 1"
