import re

from rag_assistant.config import settings
from rag_assistant.ingestion.models import ContentElement, IngestOptions

_model = None
_batched = None

# Domain vocabulary the model keeps mis-hearing as plausible-but-wrong words,
# worst at Tamil/English code-switch points: "E-way Bill" -> "EV bill",
# "landed cost" -> "land cost"/"London cost", "cost allocation" ->
# "cost-alignation". Biasing decoding towards these terms is the fix.
#
# `hotwords`, NOT `initial_prompt`: initial_prompt only primes the FIRST
# window, so on a 28-minute meeting it would influence ~30 seconds and nothing
# after. hotwords applies to every window. Checked in faster-whisper's source
# that it is not suppressed here - it is skipped only when `prefix` is set,
# which this call never sets (§7 item 4: verify, don't assume).
# Configurable via RAG_ASR_HOTWORDS (§11) rather than hardcoded, because
# different clients bring different jargon and that should not need a code
# change. Keep it SHORT whatever the client: measured on 45s of a real
# recording, four terms cost nothing (RTF 0.28 vs 0.30 for no hotwords at all)
# while ten terms ran at RTF 2.03 - a 7x penalty, because every extra hotword
# token is prepended to every decoding window. Adding terms is not free.

# Whisper hallucinates far more on long segments. VAD's own default for this is
# `inf` - it never split on length at all - so segments could run arbitrarily
# long before this cap.
_VAD_PARAMETERS = {"max_speech_duration_s": 30}

# A phrase repeated this many times back-to-back is a decoder loop, not speech.
_MIN_REPEATS = 3
_PUNCT_RE = re.compile(r"[^\w\s]")

# Human-readable annotation for the transcript DOWNLOAD only (§26). It is
# deliberately not part of any indexed text: an earlier version injected this
# string into the segment's own `text`, so BGE-M3 embedded the literal words
# "LOW CONFIDENCE review" into every flagged chunk's dense AND sparse vectors -
# polluting the lexical side of hybrid retrieval with a meaningless repeated
# n-gram, and risking false matches for real queries containing "review" or
# "check". The signal now lives in ContentElement/Chunk.low_confidence (§15)
# and is rendered as text only at display time.
REVIEW_ANNOTATION = "[LOW CONFIDENCE - review]"

# faster-whisper's own `log_prob_threshold` default: a decode averaging below
# this is what the library itself treats as failed. Used here only to TAG the
# segment, never to discard it - §7 item 20 measured that rejecting decodes at
# -0.5 deleted 65% of real speech, so the signal is surfaced rather than acted
# on destructively.
_LOW_CONFIDENCE_LOGPROB = -1.0


def _get_model():
    global _model
    if _model is None:
        # Imported lazily - only needed once ASR actually runs (RAM matters, §29).
        from faster_whisper import WhisperModel

        # Model size deviates from CLAUDE.md's §5 pick (large-v3-turbo, ~1.5GB+
        # loaded) for the same reason the LLM default was corrected from 4B to
        # 0.6B: this machine measured well under 1GB free RAM. "small" (~500MB)
        # is the size that can actually be tested right now; large-v3-turbo
        # stays the documented upgrade path once more RAM is free, same as
        # qwen3:1.7b for the LLM (§13) - not silently swapped, flagged here.
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def _get_batched():
    """Batched decoding wrapper, cached like the model itself.

    Measured on 45s of real audio, same settings, same slice: sequential
    RTF 1.75 vs batched 0.34 at batch_size=8 - roughly 5x, and with slightly
    more words, so it is not buying speed by dropping content (the failure
    mode three decoder thresholds hit, §7 item 20). Sizes 4/8/16 measured
    within noise of each other, so 8 is taken as the default.

    Known cost, accepted deliberately: batching merges VAD spans, so the same
    slice yielded 2 segments instead of 11. Timestamp citations therefore land
    within roughly 20 seconds rather than 4. Chunks also carry more speech
    each, which is not purely a loss - more context per retrieved chunk - but
    §20.2's "what was said around minute 32" gets correspondingly vaguer.
    """
    global _batched
    if _batched is None:
        from faster_whisper import BatchedInferencePipeline

        _batched = BatchedInferencePipeline(model=_get_model())
    return _batched


def _normalise(text: str) -> str:
    return _PUNCT_RE.sub("", text.lower()).strip()


def collapse_repeated_ngrams(text: str) -> tuple[str, bool]:
    """Collapses an n-gram repeated 3+ times back-to-back within one segment.

    Longest repeating unit wins, so "the cost the cost the cost" collapses to
    "the cost" rather than being mistaken for a single repeated word.
    """
    words = text.split()
    out: list[str] = []
    collapsed_any = False
    i = 0
    while i < len(words):
        collapsed_here = False
        for size in range((len(words) - i) // _MIN_REPEATS, 0, -1):
            gram = words[i : i + size]
            repeats, j = 1, i + size
            while words[j : j + size] == gram:
                repeats += 1
                j += size
            if repeats >= _MIN_REPEATS:
                out.extend(gram)
                i = j
                collapsed_any = collapsed_here = True
                break
        if not collapsed_here:
            out.append(words[i])
            i += 1
    return " ".join(out), collapsed_any


def collapse_repeated_segments(elements: list[ContentElement]) -> list[ContentElement]:
    """Removes decoder-loop repetition and flags what it touched.

    Handles repetition at BOTH levels, because the real failure showed up
    across segments, not inside one: a client recording produced four separate
    segments (901.3s, 903.3s, 905.3s, 907.3s) all reading "It is not making
    impact." Collapsing only within a segment would have left every one of
    them in place.

    A collapsed run keeps the first segment's start_time and the last one's
    end_time, so the citation still spans the audio it came from, and is marked
    for review rather than silently rewritten - the reader has to be able to
    tell a cleaned-up guess from a clean transcription (§9).
    """
    cleaned: list[ContentElement] = []
    index = 0
    while index < len(elements):
        element = elements[index]
        key = _normalise(element.text)

        run_end = index + 1
        if key:
            while run_end < len(elements) and _normalise(elements[run_end].text) == key:
                run_end += 1
        repeats = run_end - index

        text, collapsed_inside = collapse_repeated_ngrams(element.text)
        element.text = text  # always clean - the flag is metadata, never content
        if repeats >= _MIN_REPEATS:
            # A genuine loop: keep one, span the whole run, mark for review.
            element.low_confidence = True
            element.end_time = elements[run_end - 1].end_time
            index = run_end
        else:
            # Below the threshold this is ordinary speech - someone really did
            # say "Correct." twice. Advance one at a time so the duplicates are
            # KEPT; skipping to run_end here silently deleted them, which a
            # test caught before it could quietly drop real content.
            element.low_confidence = element.low_confidence or collapsed_inside
            index += 1
        cleaned.append(element)
    return cleaned


def transcribe(
    wav_path: str,
    on_progress=None,
    options: IngestOptions | None = None,
) -> list[ContentElement]:
    """faster-whisper only for V2 (CLAUDE.md §13 - IndicConformer routing for
    Tamil-heavy audio stays a benchmark-time upgrade, not built blind here).
    One ContentElement per segment, carrying real start/end timestamps -
    this is what makes timestamp citations possible later (§20.2)."""
    options = options or IngestOptions()
    model = _get_model()
    # vad_filter uses faster-whisper's bundled Silero VAD integration directly
    # (CLAUDE.md §5's separate "Silero VAD" line) rather than reimplementing
    # a manual VAD pass - same model, no duplicate integration.
    # task="translate" makes Whisper emit English for non-English speech, in the
    # same single pass as transcription - so Tamil/Tanglish audio can be stored
    # as English text at no extra ASR cost. Verified against the installed
    # faster-whisper 1.2.1 signature, not assumed (§7 item 4).
    #
    # Which one is chosen also decides what language ANSWERS come back in, which
    # is why it is exposed to the user rather than hard-coded: measured on this
    # model, generation mirrors the language of the EVIDENCE, not the question
    # (§13). A Tamil transcript therefore yields Tamil answers, an English
    # translation yields English ones.
    #
    # CLAUDE.md §13's warning still applies to the model *choice*: `large-v3-turbo`
    # is transcribe-only and `distil-large-v3` is English-only, so neither can
    # serve this path. `small` is multilingual and does both.
    # `language=None` means auto-detect, which is measurably the WORST option
    # for code-mixed speech. Benchmarked on 45s of a real Tanglish business
    # recording (mostly English vocabulary in Tamil sentence shapes):
    #
    #   auto-detect      RTF 4.44   hallucinated Tamil, stray Portuguese/Korean
    #   language="en"    RTF 1.40   clean, verbatim ("DCR is there in inventory")
    #   task="translate" RTF 0.30   clean English, slightly paraphrased
    #
    # Auto-detect locked onto Tamil at 0.97 confidence and then tried to render
    # English terms in Tamil script - which is why it is both wrong AND ~15x
    # slower than translating: the wrong target language produces long, garbled
    # token sequences. Hence both knobs are exposed to the caller rather than
    # left to detection.
    #
    # Decoding is constrained rather than left to the model's own confidence,
    # because on a real client recording it produced verbatim repetition loops
    # ("It is not making impact." four times in a row) and mis-heard domain
    # jargon. Each argument below earns its place:
    #
    #   condition_on_previous_text=False  one bad segment no longer seeds the
    #       next, which is what lets a loop run across several segments.
    #       Measured free here: RTF 0.28 vs 0.30, same word count. (This
    #       contradicts §7 item 18's earlier reading that it was slower - that
    #       was measured under auto-detect, a regime where everything was slow
    #       and garbled. Re-measured under the corrected language settings.)
    #   vad_parameters max_speech_duration_s=30  VAD's own default is `inf`, so
    #       nothing ever split on length; hallucination rises sharply past ~30s.
    #   hotwords                          biases decoding towards the domain
    #       vocabulary; see HOTWORDS for why the list must stay short.
    #   compression_ratio_threshold=2.4   the default, set explicitly because it
    #       is the knob that rejects repetitive output; leaving it implicit
    #       hides that this is deliberate.
    #   temperature                       deliberately NOT passed. The default
    #       is already the fallback ladder [0.0, 0.2, ...1.0] - it starts at 0
    #       and escalates only when a decode is rejected. Passing a scalar 0
    #       would DISABLE that ladder, removing the re-decode entirely.
    #
    # Two thresholds were specified for this fix, tried, measured, and REJECTED
    # because each silently deleted real speech from the same 45s slice:
    #   log_prob_threshold=-0.5   17 segments/125 words -> 8 segments/43 words.
    #   no_speech_threshold=0.15  -> 8 segments/41 words. Its docstring reads
    #       "if no_speech probability is HIGHER than this value... consider the
    #       segment silent", so lowering it from 0.6 makes the model far more
    #       eager to discard speech as silence - the opposite of the intent.
    # Applied together they cut a 45s slice to a single segment and an 80s
    # slice to nothing at all. Do not reinstate them without new measurements.
    decode_options = dict(
        beam_size=1,
        vad_filter=True,
        vad_parameters=_VAD_PARAMETERS,
        task=options.task,
        language=options.language,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        # Per-source, not a global default. See IngestOptions.hotwords.
        hotwords=options.hotwords or None,
        # The real segmentation control. vad_parameters' max_speech_duration_s
        # above is POPPED and overwritten by this under batched decoding, so
        # setting only that one did nothing at all (CLAUDE.md §7 item 34).
        chunk_length=settings.asr_chunk_length,
    )
    batch_size = settings.asr_batch_size
    if batch_size > 1:
        segments, info = _get_batched().transcribe(
            wav_path, batch_size=batch_size, **decode_options
        )
    else:
        segments, info = model.transcribe(wav_path, **decode_options)
    # segments is a lazy generator (CLAUDE.md §7 item 5) - iterating it IS the
    # transcription. Consuming it in a loop rather than list() lets progress be
    # reported as it goes: a 28-minute recording takes a long time on this CPU,
    # and without this the job row sat at one fixed percentage the whole way,
    # leaving a real user unable to tell processing from hung.
    total_seconds = getattr(info, "duration", 0) or 0
    last_reported = 0.0

    elements = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            # avg_logprob is the decoder's own mean token log-probability for
            # this segment (negative; nearer 0 is more confident) - confirmed
            # present on the installed Segment dataclass alongside
            # compression_ratio and no_speech_prob (§7 item 4). Recorded as-is
            # so it stays comparable to faster-whisper's own default cutoff of
            # -1.0, which is also what flags the segment here.
            avg_logprob = getattr(seg, "avg_logprob", None)
            elements.append(
                ContentElement(
                    element_type="transcript_segment",
                    text=text,
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence_score=avg_logprob,
                    low_confidence=avg_logprob is not None
                    and avg_logprob < _LOW_CONFIDENCE_LOGPROB,
                )
            )
        if on_progress and total_seconds:
            fraction = min(seg.end / total_seconds, 1.0)
            # Throttled: one SQLite write per 2% of audio, not per segment.
            if fraction - last_reported >= 0.02:
                last_reported = fraction
                on_progress(fraction)
    return collapse_repeated_segments(elements)
