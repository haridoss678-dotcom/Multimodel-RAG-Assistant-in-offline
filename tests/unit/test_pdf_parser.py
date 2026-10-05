from rag_assistant.ingestion.pdf_parser import parse_pdf


def test_parse_pdf_returns_one_element_per_page(sample_pdf_bytes):
    elements, ocr_version = parse_pdf(sample_pdf_bytes, "notes.pdf")
    assert len(elements) == 2
    assert elements[0].page == 1
    assert elements[1].page == 2
    assert "Decorators" in elements[1].text
    assert "decorator" in elements[1].text.lower()
    # A digital PDF has a real text layer, so OCR must not have run.
    assert ocr_version is None
    assert all(e.ocr is False for e in elements)
    assert all(e.confidence_score is None for e in elements)


def test_legacy_tamil_font_page_is_ocred_as_tamil(monkeypatch, sample_pdf_bytes):
    """CLAUDE.md §7 item 44 - a Bamini/TSCII page is correctly force-routed to
    OCR and was then handed to a hardcoded ENGLISH recognition model, producing
    silent garbage. The page's own fonts decide the language."""
    from rag_assistant.ingestion import pdf_parser as pp
    from rag_assistant.ingestion.ocr import OcrResult

    langs_used = []

    def fake_run_ocr(image_path, lang="en"):
        langs_used.append(lang)
        return OcrResult(
            text="தமிழ் உரை", engine=f"paddleocr-{lang}", confidence=0.93, low_confidence=False
        )

    monkeypatch.setattr(pp, "is_legacy_tamil_font", lambda doc, i: True)
    monkeypatch.setattr(pp, "run_ocr", fake_run_ocr)

    elements, ocr_version = parse_pdf(sample_pdf_bytes, "tamil.pdf", ocr_language="en")

    # The upload said English; the legacy font overrides it.
    assert langs_used and set(langs_used) == {"ta"}
    assert ocr_version == "paddleocr-ta"
    assert all(e.ocr is True for e in elements)
    assert all(e.confidence_score == 0.93 for e in elements)


def test_low_confidence_ocr_is_flagged_on_the_element(monkeypatch, sample_pdf_bytes):
    """§9 requires low-confidence OCR tagged in metadata, exactly as ASR is.
    PaddleOCR's per-line rec_scores were being computed and discarded."""
    from rag_assistant.ingestion import pdf_parser as pp
    from rag_assistant.ingestion.ocr import OcrResult

    # The legacy-font signal is the supported way to force the OCR branch; it
    # exercises the same code path as a scanned page with no text layer.
    monkeypatch.setattr(pp, "is_legacy_tamil_font", lambda doc, i: True)
    monkeypatch.setattr(
        pp,
        "run_ocr",
        lambda path, lang="en": OcrResult(
            text="grbge txt", engine=f"paddleocr-{lang}", confidence=0.42, low_confidence=True
        ),
    )

    elements, _version = parse_pdf(sample_pdf_bytes, "scan.pdf")

    assert elements and all(e.low_confidence is True for e in elements)
    assert all(e.confidence_score == 0.42 for e in elements)
