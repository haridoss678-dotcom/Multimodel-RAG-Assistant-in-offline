from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class IngestOptions:
    """Per-source ingestion settings, carried as one value instead of four
    positional parameters threaded through six functions.

    Grouped deliberately: ingestion already had to pass `translate_to_english`
    and `asr_language` down four layers, the signatures drifted twice and left
    test stubs silently accepting the wrong arity, and hotwords now makes a
    third (CLAUDE.md §7 item 36). Adding a fourth option later touches this
    class and its two real call sites, not every function in between.

    task:      "transcribe" keeps the source language; "translate" emits English
               in the SAME pass at no extra ASR cost. This also decides the
               answer language, because generation mirrors the evidence (§23).
    language:  None means auto-detect, which is measurably the worst option for
               code-mixed speech - RTF 4.44 and wrong, against 0.30 for
               translate (§7 item 18). Name the language whenever it is known.
    hotwords:  comma-separated domain vocabulary biased during decoding.
               Defaults to EMPTY and is set per upload, because it is per-client
               jargon: applying one client's "DC, GRN, E-way Bill, landed cost"
               to a Python lecture biases decoding toward terms that are not in
               the audio, and hotword bias can insert them. Keep any list short -
               4 terms measured RTF 0.28, 10 terms 2.03.
    ocr_language:
               PDF-only: which PaddleOCR recognition model reads a scanned page.
               A page whose embedded fonts are legacy Tamil overrides this to
               "ta" regardless, since such a page is Tamil by definition.
    """

    task: str = "transcribe"
    language: str | None = None
    hotwords: str = ""
    ocr_language: str = "en"

    @property
    def version_tag(self) -> str:
        """Recorded as the source's asr_version, so a translated transcript is
        distinguishable from a natively-English one after the fact."""
        return f"faster-whisper-small:{self.task}:{self.language or 'auto'}"


class ContentElement(BaseModel):
    """Pre-chunking structural unit shared by all source formats (CLAUDE.md §15)."""

    element_type: str  # "paragraph" (PDF) or "transcript_segment" (audio/video/YouTube)
    text: str
    page: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    speaker: str | None = None
    # True when this segment provably contains more than one voice, so `speaker`
    # is the DOMINANT speaker rather than the only one. Batched ASR merges VAD
    # spans into ~25s segments while speaker turns average 4.5s, so on real
    # multi-speaker audio this is the common case, not the exception (67 of 68
    # segments on the reference recording). Carried through to the generation
    # prompt, which must then decline to attribute a specific statement to a
    # named person (CLAUDE.md §7 item 29).
    multi_speaker: bool = False
    language: str | None = None  # "en" | "ta" | "mixed", CLAUDE.md §15

    # §9's "low-confidence OCR/ASR segments are tagged as such in metadata" -
    # structured metadata, deliberately NOT a marker inside `text`. The first
    # implementation injected "[LOW CONFIDENCE - review]" into the text itself,
    # which meant BGE-M3 embedded those literal words into both the dense and
    # sparse vectors, and nothing in retrieval or grounding could read the
    # signal at all. confidence_score is faster-whisper's own `avg_logprob`
    # (negative; closer to 0 is better), carried through unchanged rather than
    # rescaled, so it stays comparable to the library's own thresholds.
    low_confidence: bool = False
    confidence_score: float | None = None
    # True when this text came from OCR rather than a real PDF text layer. A
    # citation resting on a recognised scan is a different kind of evidence
    # from one resting on embedded text, and nothing downstream could tell them
    # apart before (CLAUDE.md §7 item 44).
    ocr: bool = False


class Chunk(BaseModel):
    """The retrievable unit (CLAUDE.md §15)."""

    chunk_id: str
    source_id: str
    text: str
    element_type: str
    page: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    speaker: str | None = None
    # Carried through so the generation prompt can refuse to attribute a
    # statement to one person when the segment holds two (see ContentElement).
    multi_speaker: bool = False
    language: str | None = None
    parent_id: str | None = None
    # Position within the parent group, assigned at chunking time.
    #
    # Needed because context expansion has to reassemble a parent in DOCUMENT
    # order and had nothing to sort PDF chunks by: it sorted on
    # (start_time or 0, chunk_id), and every chunk on a page has start_time
    # None -> 0, so the tiebreak was chunk_id - a uuid4. A reassembled PDF page
    # was therefore in random order (CLAUDE.md §7 item 27). Transcript chunks
    # were unaffected because start_time orders them correctly.
    chunk_index: int = 0
    embedding_model_version: str
    chunking_version: str

    # Carried through from ContentElement so the signal survives chunking and
    # reaches the Qdrant payload, where §17 indexes `low_confidence` and §20's
    # reranker can act on it.
    low_confidence: bool = False
    confidence_score: float | None = None
    ocr: bool = False
