import tempfile
import uuid
from pathlib import Path

import pymupdf as fitz
import pymupdf4llm

from rag_assistant.ingestion.language_detect import detect_language, is_legacy_tamil_font
from rag_assistant.ingestion.models import ContentElement
from rag_assistant.ingestion.ocr import OcrResult, run_ocr


def _ocr_page(doc: fitz.Document, page_index_zero: int, lang: str) -> OcrResult:
    page = doc[page_index_zero]
    pix = page.get_pixmap(dpi=200)
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.png"
    pix.save(str(tmp_path))
    try:
        return run_ocr(str(tmp_path), lang=lang)
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_pdf(
    file_bytes: bytes,
    filename: str,
    ocr_language: str = "en",
) -> tuple[list[ContentElement], str | None]:
    """Extracts text + page numbers, returning (elements, ocr_version).

    PyMuPDF4LLM (CLAUDE.md §13 fast path) for normal pages; PaddleOCR/Tesseract
    (§5) for pages that are scanned or use a legacy non-Unicode Tamil font,
    where the "text" layer would otherwise be garbage or empty.

    OCR LANGUAGE IS ROUTED PER PAGE (CLAUDE.md §7 item 44). Every page used to
    go through a hardcoded lang="en" recognition model, so a scanned Tamil page
    was read by an English model and produced silent garbage - and it interacted
    badly with the legacy-font feature, which correctly force-routes
    Bamini/TSCII/TAB pages to OCR and then handed them to the wrong model. A
    page whose fonts are legacy Tamil is Tamil by definition, so it overrides
    whatever default the upload supplied.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        doc = fitz.open(str(tmp_path))
        # Each entry is (page_index, language to recognise it in).
        needs_ocr: dict[int, str] = {}
        for i in range(doc.page_count):
            legacy_tamil = is_legacy_tamil_font(doc, i)
            has_text = bool(doc[i].get_text().strip())
            if not has_text or legacy_tamil:
                needs_ocr[i] = "ta" if legacy_tamil else ocr_language

        pages = pymupdf4llm.to_markdown(str(tmp_path), page_chunks=True)

        elements: list[ContentElement] = []
        engines: set[str] = set()
        for page in pages:
            page_number = page.get("metadata", {}).get("page_number", None)
            page_index_zero = (page_number - 1) if page_number else None

            ocr_result: OcrResult | None = None
            if page_index_zero is not None and page_index_zero in needs_ocr:
                ocr_result = _ocr_page(doc, page_index_zero, needs_ocr[page_index_zero])
                text = ocr_result.text.strip()
                engines.add(ocr_result.engine)
            else:
                text = page.get("text", "").strip()

            if not text:
                continue
            elements.append(
                ContentElement(
                    element_type="paragraph",
                    text=text,
                    page=page_number,
                    language=detect_language(text),
                    # A citation from an OCR'd page is not the same kind of
                    # evidence as one from a real text layer, and until now
                    # nothing downstream could tell them apart.
                    ocr=ocr_result is not None,
                    confidence_score=ocr_result.confidence if ocr_result else None,
                    low_confidence=bool(ocr_result and ocr_result.low_confidence),
                )
            )
        doc.close()
        return elements, ("+".join(sorted(engines)) if engines else None)
    finally:
        tmp_path.unlink(missing_ok=True)
