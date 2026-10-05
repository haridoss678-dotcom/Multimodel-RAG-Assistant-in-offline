import logging
import os

logger = logging.getLogger(__name__)


class DiarizationUnavailable(Exception):
    pass


_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise DiarizationUnavailable(
                "No HF_TOKEN set - pyannote's gated models need manual terms "
                "acceptance on huggingface.co (CLAUDE.md §7 item 3) before this can run."
            )
        from pyannote.audio import Pipeline

        # pyannote.audio 4.x renamed this argument from `use_auth_token` to
        # `token` - verified against the installed signature, not assumed
        # (CLAUDE.md §7 item 4).
        _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    return _pipeline


def _relabel_by_first_appearance(turns: dict[tuple[float, float], str]) -> dict:
    """pyannote labels clusters SPEAKER_00, SPEAKER_01... in no particular
    order. Renumber by who actually speaks first, so "Speaker 1" is the first
    voice heard - which is what a reader expects, and what the user asked for.
    """
    order: dict[str, str] = {}
    for _span, speaker in sorted(turns.items()):
        if speaker not in order:
            order[speaker] = f"Speaker {len(order) + 1}"
    return {span: order[speaker] for span, speaker in turns.items()}


def _load_waveform(wav_path: str) -> dict:
    """Loads the wav ourselves instead of letting pyannote open the path.

    pyannote 4.x decodes audio through TorchCodec, which on Windows needs the
    FFmpeg "full-shared" DLLs; with the ordinary winget FFmpeg build installed
    here it raises `Could not load libtorchcodec` before any inference runs.
    Its documented Mapping input - {"waveform": (channel, time) tensor,
    "sample_rate": int} - skips that decoder path entirely, using soundfile,
    which is already a dependency (§8 rule 4: no new dependency for this).

    Cost worth knowing: this holds the whole recording in memory as float32,
    about 107MB for 28 minutes of 16kHz mono. That is affordable here only
    because ASR has already finished and released its model by this point
    (§29's one-heavy-model-at-a-time rule).
    """
    import soundfile
    import torch

    data, sample_rate = soundfile.read(wav_path, dtype="float32", always_2d=True)
    # soundfile returns (time, channels); pyannote wants (channel, time).
    return {"waveform": torch.from_numpy(data.T), "sample_rate": sample_rate}


def diarize(wav_path: str) -> dict[tuple[float, float], str] | None:
    """Returns {(start, end): speaker_label} for each speaker turn, or None
    if diarization can't run (missing gated access - CLAUDE.md's Dependency
    Failure Matrix, §27: diarization is optional, ingestion continues
    without speaker metadata rather than failing).

    Always attempted once per source - there's no cheap way to know a file
    is single-speaker without running some speaker analysis (CLAUDE.md §32
    scope note). If it resolves to exactly one speaker, callers should treat
    that as "no speaker labels needed" rather than storing a label at all.
    """
    try:
        pipeline = _get_pipeline()
        diarization = pipeline(_load_waveform(wav_path))
    except DiarizationUnavailable:
        return None
    except Exception:
        # §27 makes diarization explicitly optional: losing speaker labels
        # degrades one feature, losing the transcript means the source can
        # never answer anything. Only DiarizationUnavailable was caught before,
        # so a gated-repo 403 (or any model/inference failure) propagated and
        # failed the WHOLE ingestion - a real user would have waited minutes on
        # a long recording only to end at FAILED with no transcript at all.
        # Model inference is one of §8 rule 5's named true boundaries, so a
        # broad catch is correct here; it is logged, not swallowed silently.
        logger.warning("diarization failed - continuing without speaker labels", exc_info=True)
        return None

    # pyannote 4.x wraps the result: the pipeline returns a DiarizeOutput whose
    # `.speaker_diarization` is the Annotation that 3.x returned directly.
    # Checked against the installed dataclass rather than assumed (§7 item 4) -
    # calling .itertracks() on the wrapper raises AttributeError.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    turns: dict[tuple[float, float], str] = {}
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns[(turn.start, turn.end)] = speaker
    return _relabel_by_first_appearance(turns)


# Seconds of a non-dominant speaker inside one ASR segment above which the
# segment is treated as genuinely containing more than one voice.
#
# Chosen by measurement on the real 28-minute recording, not by guess
# (CLAUDE.md §7 item 29). Mean diarization turn there is 4.5s, so ONE second of
# another speaker is a real utterance ("Correct.", "What is the DC?"), not a
# boundary artefact. Measured flagging rates over 68 segments:
#     >20% of segment duration (the obvious rule) ->  10 (15%)
#     >10% of duration                           ->  28 (41%)
#     >1.0 absolute second  [adopted]            ->  49 (72%)
#     any second speaker at all                  ->  63 (93%)
# The 20% rule was rejected because 53 segments contain a second speaker's words
# yet fall under it - the prompt would have confidently attributed those words to
# the dominant speaker. An ABSOLUTE threshold is right here, not a ratio: a
# 1-second sentence inside a 25-second segment is 4% of it, below any sensible
# ratio, and still a whole sentence put in the wrong person's mouth.
MULTI_SPEAKER_MIN_SECONDS = 1.0


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def speaker_span(
    turns: dict[tuple[float, float], str] | None,
    start: float | None,
    end: float | None,
) -> tuple[str | None, bool]:
    """Returns (dominant_speaker, contains_more_than_one_speaker) for a segment.

    Replaces a start-time lookup that was measurably wrong. Batched ASR merges
    VAD spans into ~25-second segments while diarization turns average 4.5s, so
    a segment spans ~5.4 turns - 67 of 68 on the real recording overlapped more
    than one. Stamping each segment with whoever happened to be talking at its
    START collapsed a genuine 2-speaker meeting to a single label
    (CLAUDE.md §7 item 29).

    Maximum overlap is the correct assignment, but measurement showed it is NOT
    sufficient on its own: the dominant speaker held 89.7% of all speech, so
    max-overlap still returns that one name for all 68 segments. Hence the
    second return value - when a segment really does contain another voice, the
    honest answer is to say so rather than to attribute confidently.
    """
    if not turns or start is None:
        return None, False
    if len({s for s in turns.values()}) <= 1:
        return None, False  # single speaker - no label needed (see diarize())

    end = end if end is not None else start
    if end <= start:
        # A zero-length span overlaps nothing, so overlap arithmetic returns no
        # speaker at all. Degrade to a containment lookup at that instant rather
        # than silently dropping the label.
        for (turn_start, turn_end), speaker in turns.items():
            if turn_start <= start < turn_end:
                return speaker, False
        return None, False

    totals: dict[str, float] = {}
    for (turn_start, turn_end), speaker in turns.items():
        seconds = _overlap(start, end, turn_start, turn_end)
        if seconds > 0:
            totals[speaker] = totals.get(speaker, 0.0) + seconds
    if not totals:
        return None, False

    dominant = max(totals, key=lambda s: totals[s])
    others = [seconds for speaker, seconds in totals.items() if speaker != dominant]
    return dominant, bool(others) and max(others) > MULTI_SPEAKER_MIN_SECONDS
