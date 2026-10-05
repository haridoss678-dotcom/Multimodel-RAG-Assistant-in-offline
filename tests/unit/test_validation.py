import pytest

from rag_assistant.ingestion.validation import ValidationError, validate_file


def test_validate_accepts_real_pdf(sample_pdf_bytes):
    assert validate_file("notes.pdf", sample_pdf_bytes) == "pdf"


def test_validate_rejects_non_pdf_extension(sample_pdf_bytes):
    with pytest.raises(ValidationError, match="Unsupported format"):
        validate_file("notes.txt", sample_pdf_bytes)


def test_validate_rejects_empty_file():
    with pytest.raises(ValidationError, match="Empty"):
        validate_file("notes.pdf", b"")


def test_validate_rejects_corrupted_pdf():
    with pytest.raises(ValidationError, match="Corrupted"):
        validate_file("notes.pdf", b"not a real pdf")
