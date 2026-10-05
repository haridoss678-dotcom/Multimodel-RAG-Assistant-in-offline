import pymupdf as fitz

# The ceiling is set by MEMORY, not by policy, and that is worth stating plainly
# (CLAUDE.md §7 item 41). api/routes/sources.py does `file.file.read()` - the
# entire upload is held in RAM, hashed there, handed to the background task, and
# stays resident for the whole ingestion, which is 84 minutes for a 28-minute
# recording. On a machine that routinely shows under 1GB free, raising this
# number is not free: it converts a rejected upload into an OOM mid-pipeline.
#
# §1 claims video support for "minutes to hours" and a 2-hour 1080p mp4 exceeds
# this, so the cap and the claim genuinely disagree. Closing that gap needs the
# upload streamed to disk and hashed incrementally, with the pipeline taking a
# PATH rather than bytes - a real change across validation, the state machine,
# the PDF parser and the audio extractor, deliberately not attempted as part of
# an error-code fix. Until then this number stays where the hardware puts it.
MAX_FILE_BYTES = 500 * 1024 * 1024  # 500MB

_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


class ValidationError(Exception):
    pass


class FileTooLarge(ValidationError):
    """Distinct from an unsupported format: the type is fine, the size is not.

    Previously both returned 415 UNSUPPORTED_FORMAT, which told the user their
    file type was wrong when in fact it was simply too big - two different
    problems with two different fixes (§7 item 41)."""


def _validate_pdf(file_bytes: bytes) -> None:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValidationError("PDF has no pages")
        doc.close()
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Corrupted or unreadable PDF: {exc}") from exc


def sniff_type(filename: str) -> str:
    """Extension-only guess, used only to register a source before full
    validation runs (so a validation failure still has a row to mark FAILED
    against) - never used to decide whether ingestion actually proceeds."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith(_AUDIO_EXTS):
        return "audio"
    if lower.endswith(_VIDEO_EXTS):
        return "video"
    return "pdf"  # arbitrary fallback label; validate_file() below is what actually rejects it


def validate_file(filename: str, file_bytes: bytes) -> str:
    """Validates an upload and returns its source_type. V2 accepts
    PDF/audio/video (CLAUDE.md §32) - YouTube URLs go through a separate
    validation path (no file bytes involved), see validate_youtube_url()."""
    lower = filename.lower()
    if len(file_bytes) == 0:
        raise ValidationError("Empty file")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise FileTooLarge(
            f"File is {len(file_bytes) / 1e6:.0f}MB; the limit is "
            f"{MAX_FILE_BYTES / 1e6:.0f}MB because the whole upload is held in memory "
            "during ingestion on this machine."
        )

    if lower.endswith(".pdf"):
        _validate_pdf(file_bytes)
        return "pdf"
    if lower.endswith(_AUDIO_EXTS):
        return "audio"
    if lower.endswith(_VIDEO_EXTS):
        return "video"
    raise ValidationError(f"Unsupported format: {filename!r}")


def validate_youtube_url(url: str) -> None:
    if "youtube.com" not in url and "youtu.be" not in url:
        raise ValidationError(f"Not a YouTube URL: {url!r}")
