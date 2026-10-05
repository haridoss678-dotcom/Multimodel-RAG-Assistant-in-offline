import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# One PaddleOCR instance PER LANGUAGE. The engine binds a recognition model at
# construction time, so a single cached instance meant every page - including
# Tamil ones - was recognised by whatever language was loaded first.
_paddle_ocr: dict[str, object] = {}

# PaddleOCR returns a per-line recognition score in [0, 1], higher is better.
# Provisional threshold: clean synthetic text scores ~0.99 here, so 0.80 marks
# "clearly below a clean read" without flagging ordinary variation.
#
# HONEST LIMIT (CLAUDE.md §9's accuracy rule): unlike the ASR threshold, which
# is faster-whisper's own documented cutoff, this one is NOT yet validated
# against a genuinely degraded scan - no such fixture exists in this project.
# It is deliberately conservative, and it only TAGS; nothing is ever discarded
# on the strength of it (§7 item 20's lesson about confidence gates).
LOW_CONFIDENCE_OCR_SCORE = 0.80


@dataclass(frozen=True)
class OcrResult:
    """What OCR produced, plus how much to trust it.

    The text alone was all that was returned before, so PaddleOCR's per-line
    rec_scores were computed and thrown away - garbage OCR flowed into
    retrieval indistinguishable from a clean text layer, with no warning
    anywhere (CLAUDE.md §7 item 44). §9 requires low-confidence OCR to be
    tagged in metadata exactly as low-confidence ASR already is.
    """

    text: str
    engine: str
    confidence: float | None  # mean rec_score; None when the engine reports none
    low_confidence: bool


def _get_paddle_ocr(lang: str):
    if lang not in _paddle_ocr:
        # Imported lazily - RAM matters (§29), 778MB measured when loaded.
        from paddleocr import PaddleOCR

        # enable_mkldnn=False works around a real, confirmed environment bug on
        # this machine: paddlepaddle 3.3.1's oneDNN CPU path throws
        # NotImplementedError on this model's text-detection step. Verified
        # directly (CLAUDE.md §7 item 4) before writing this, not assumed.
        _paddle_ocr[lang] = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _paddle_ocr[lang]


def run_paddle_ocr(image_path: str, lang: str = "en") -> tuple[str, list[float]]:
    """Returns (text, per-line confidence scores).

    `rec_scores` sits alongside `rec_texts` in the paddlex result and was simply
    not being read. Confirmed present in the installed package rather than
    assumed (§7 item 4).
    """
    ocr = _get_paddle_ocr(lang)
    results = ocr.predict(image_path)
    lines: list[str] = []
    scores: list[float] = []
    for r in results:
        lines.extend(r.get("rec_texts", []))
        scores.extend(float(s) for s in r.get("rec_scores", []) or [])
    return "\n".join(lines), scores


def run_tesseract_ocr(image_path: str) -> str:
    """Fallback if PaddleOCR fails (CLAUDE.md §5/§27). Only the 'eng' pack is
    installed on this machine, so this path CANNOT read Tamil - PaddleOCR is
    the primary path specifically because it ships a Tamil recognition model.
    Returns no per-line confidence; pytesseract's confidence needs a different
    API (image_to_data) and is not wired up."""
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(image_path))


def run_ocr(image_path: str, lang: str = "en") -> OcrResult:
    """PaddleOCR in the requested language, Tesseract as a last resort."""
    try:
        text, scores = run_paddle_ocr(image_path, lang)
        if text.strip():
            mean = sum(scores) / len(scores) if scores else None
            return OcrResult(
                text=text,
                engine=f"paddleocr-{lang}",
                confidence=mean,
                low_confidence=mean is not None and mean < LOW_CONFIDENCE_OCR_SCORE,
            )
        logger.info("paddleocr(%s) returned no text for %s - trying tesseract", lang, image_path)
    except Exception:
        # Model inference is one of §8 rule 5's true boundaries, so catching
        # broadly is correct - but the previous `except Exception: pass` made a
        # real PaddleOCR failure indistinguishable from a blank page, which is
        # exactly the difference a reader needs (§7 item 44).
        logger.warning(
            "paddleocr(%s) failed on %s - falling back to tesseract",
            lang,
            image_path,
            exc_info=True,
        )

    if lang != "en":
        logger.warning(
            "falling back to Tesseract for a %s page, but only the 'eng' traineddata is "
            "installed - expect unusable output for non-Latin scripts",
            lang,
        )
    text = run_tesseract_ocr(image_path)
    return OcrResult(text=text, engine="tesseract-eng", confidence=None, low_confidence=False)
