import logging
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from rag_assistant.config import settings
from rag_assistant.ingestion.asr import REVIEW_ANNOTATION
from rag_assistant.ingestion.models import IngestOptions
from rag_assistant.ingestion.pipeline import IngestionCancelled, ingest_file, ingest_youtube
from rag_assistant.ingestion.state_machine import (
    compute_source_id,
    load_stage_artifact,
    request_cancel,
)
from rag_assistant.ingestion.validation import (
    MAX_FILE_BYTES,
    FileTooLarge,
    ValidationError,
    validate_file,
    validate_youtube_url,
)
from rag_assistant.ingestion.youtube import YouTubeError, extract_video_id
from rag_assistant.storage.db import get_connection
from rag_assistant.storage.vector_store import delete_source as delete_source_vectors
from rag_assistant.storage.vector_store import get_chunks_by_source

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sources", tags=["sources"])

# CLAUDE.md §29: a single active heavy job at a time. Background ingestions
# queue on this lock instead of running concurrently - on this machine's RAM
# budget two simultaneous ASR/embedding runs would thrash, not parallelise.
_ingest_lock = threading.Lock()


class YouTubeRequest(BaseModel):
    url: str
    translate_to_english: bool = False
    asr_language: str = ""
    hotwords: str = ""


def _ingest_options(
    translate_to_english: bool,
    asr_language: str,
    hotwords: str,
    ocr_language: str = "en",
) -> IngestOptions:
    """Builds the per-source speech settings from request fields.

    hotwords default to EMPTY rather than to settings.asr_hotwords. That default
    was one client's inventory vocabulary ("DC, GRN, E-way Bill, landed cost")
    applied to EVERY source, which biases decoding on a Python lecture or a
    maritime document toward terms that are not in the audio - and hotword bias
    can insert the biased term on similar-sounding speech, so this was a
    correctness risk in a multi-client corpus, not a harmless default
    (CLAUDE.md §7 item 36). The four terms remain available as a named preset
    via RAG_ASR_HOTWORDS, to be sent per upload when they actually apply.
    """
    return IngestOptions(
        task="translate" if translate_to_english else "transcribe",
        language=asr_language or None,
        hotwords=hotwords,
        ocr_language=ocr_language or "en",
    )


def _run_ingest_job(fn: Callable[..., str], *args: object) -> None:
    with _ingest_lock:
        try:
            fn(*args)
        except IngestionCancelled:
            # Expected: the user deleted or cancelled the source mid-run. Not an
            # error, and deliberately not logged with a traceback.
            logger.info("ingestion cancelled by user")
        except Exception:
            # pipeline.py has already recorded FAILED + the reason on the
            # source/job rows, which is what GET /jobs/{id} serves to the UI.
            # This only keeps the traceback from vanishing in a worker thread.
            logger.exception("background ingestion failed")


@router.post("/upload")
def upload_source(
    file: UploadFile,
    background: BackgroundTasks,
    translate_to_english: bool = Form(False),
    asr_language: str = Form(""),
    hotwords: str = Form(""),
    ocr_language: str = Form("en"),
):
    """Accepts the file and returns immediately; ingestion runs in the
    background, tracked via GET /jobs/{job_id} (CLAUDE.md §25/§26).

    Ingestion takes tens of seconds to minutes on this hardware (measured in
    §7 item 12), so holding the HTTP connection open for the whole pipeline
    gave the UI a dead spinner with no progress and risked client timeouts.
    Validation still runs synchronously so an unsupported file is rejected
    with a real 415 instead of failing silently in a background thread.

    `translate_to_english` and `asr_language` (audio/video only) control the
    transcript's language, and therefore the answers' language too - generation
    mirrors the evidence on this model (§13).

    Both matter for speed as well as accuracy, which is not obvious: leaving
    detection to Whisper on code-mixed speech measured **15x slower and wrong**
    versus naming the language (§7 item 18). An empty `asr_language` means
    auto-detect.
    """
    # Size is checked BEFORE reading where the client declared one, so an
    # oversized upload is rejected without first pulling it into RAM - the
    # previous order read 500MB and then refused it (§7 item 41).
    declared = getattr(file, "size", None)
    if declared is not None and declared > MAX_FILE_BYTES:
        detail = {
            "code": "FILE_TOO_LARGE",
            "message": (
                f"File is {declared / 1e6:.0f}MB; the limit is "
                f"{MAX_FILE_BYTES / 1e6:.0f}MB on this machine."
            ),
        }
        raise HTTPException(status_code=413, detail=detail)

    file_bytes = file.file.read()
    try:
        validate_file(file.filename, file_bytes)
    except FileTooLarge as exc:
        # The type is fine, the size is not - a different problem with a
        # different fix, so it gets its own code instead of 415.
        detail = {"code": "FILE_TOO_LARGE", "message": str(exc)}
        raise HTTPException(status_code=413, detail=detail) from exc
    except ValidationError as exc:
        detail = {"code": "UNSUPPORTED_FORMAT", "message": str(exc)}
        raise HTTPException(status_code=415, detail=detail) from exc

    source_id = compute_source_id(file_bytes)
    background.add_task(
        _run_ingest_job,
        ingest_file,
        file_bytes,
        file.filename,
        _ingest_options(translate_to_english, asr_language, hotwords, ocr_language),
    )
    return {"source_id": source_id, "job_id": source_id}


@router.post("/youtube")
def upload_youtube(req: YouTubeRequest, background: BackgroundTasks):
    try:
        validate_youtube_url(req.url)
    except ValidationError as exc:
        detail = {"code": "VALIDATION_ERROR", "message": str(exc)}
        raise HTTPException(status_code=400, detail=detail) from exc

    try:
        # Identity comes from the normalised video id, so the same video via
        # youtu.be/X, watch?v=X or a link with &t=30s is ONE source (§7 item 37).
        source_id = compute_source_id(extract_video_id(req.url).encode("utf-8"))
    except YouTubeError as exc:
        detail = {"code": "VALIDATION_ERROR", "message": str(exc)}
        raise HTTPException(status_code=400, detail=detail) from exc

    background.add_task(
        _run_ingest_job,
        ingest_youtube,
        req.url,
        _ingest_options(req.translate_to_english, req.asr_language, req.hotwords),
    )
    return {"source_id": source_id, "job_id": source_id}


@router.get("")
def list_sources():
    conn = get_connection()
    rows = conn.execute(
        "SELECT source_id, filename, source_type, state, created_at "
        "FROM sources ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/{source_id}")
def get_source(source_id: str):
    conn = get_connection()
    row = conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    conn.close()
    if row is None:
        detail = {"code": "NOT_FOUND", "message": "source not found"}
        raise HTTPException(status_code=404, detail=detail)
    return dict(row)


@router.delete("/{source_id}")
def delete_source(source_id: str):
    """Full cascade delete (CLAUDE.md §19.3): SQLite rows + Qdrant vectors +
    on-disk processed artifacts - one logical operation, not just the file."""
    conn = get_connection()
    row = conn.execute("SELECT 1 FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    if row is None:
        conn.close()
        detail = {"code": "NOT_FOUND", "message": "source not found"}
        raise HTTPException(status_code=404, detail=detail)

    conn.close()
    # Signal any in-flight worker BEFORE removing its rows, so it stops at its
    # next stage boundary instead of finishing and re-creating what we delete
    # here (CLAUDE.md §7 item 35). This is also the only writer of CANCELLED,
    # which was previously a state nothing ever set.
    request_cancel(source_id)

    conn = get_connection()
    conn.execute("DELETE FROM jobs WHERE source_id = ?", (source_id,))
    conn.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
    conn.commit()
    conn.close()

    delete_source_vectors(source_id)

    processed_dir = Path(settings.data_dir) / "processed" / source_id
    shutil.rmtree(processed_dir, ignore_errors=True)
    for raw_file in Path(settings.data_dir).glob(f"raw/{source_id}.*"):
        raw_file.unlink(missing_ok=True)

    return {"deleted": source_id}


_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}


@router.get("/{source_id}/media")
def get_media(source_id: str):
    """Streams the original recording, with HTTP Range support so a player can SEEK.

    This is what turns a citation from a claim into evidence: "meeting.m4a
    @1923s" is inert text until clicking it plays second 1923 of the actual
    recording. For verifying what a client said at minute 32 - the real use
    case - that is the difference between trusting the system and checking it.
    It is also the fastest human check on the ASR and speaker-attribution
    limits recorded in §7 items 29 and 44.

    Range handling comes from Starlette's FileResponse, verified present in the
    installed 1.6.0 (it references Range, 206 and Content-Range) rather than
    assumed - and exercised by an explicit Range test, because "the framework
    probably does this" is how §7 item 4 gets written again.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT filename, source_type FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "source not found"})
    if row["source_type"] not in ("audio", "video", "youtube"):
        raise HTTPException(
            415,
            {"code": "UNSUPPORTED_FORMAT", "message": "media playback is audio/video only"},
        )

    matches = sorted(Path(settings.data_dir).glob(f"raw/{source_id}.*"))
    if not matches:
        # YouTube sources have no raw file - only the downloaded temp wav,
        # which is not retained.
        raise HTTPException(
            404,
            {"code": "NOT_FOUND", "message": "no original media stored for this source"},
        )
    path = matches[0]
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        filename=row["filename"],
    )


@router.get("/{source_id}/page/{page_number}.png")
def get_page_image(source_id: str, page_number: int):
    """Renders one PDF page as a PNG, so clicking "notes.pdf p.52" shows page 52.

    Uses PyMuPDF, already a dependency and already used for OCR rasterisation
    in pdf_parser._ocr_page - far cheaper than vendoring pdf.js for a
    read-only view. Rendered pages are cached under
    data/processed/<source_id>/pages/ because a citation is clicked repeatedly.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT source_type FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "source not found"})
    if row["source_type"] != "pdf":
        raise HTTPException(
            415, {"code": "UNSUPPORTED_FORMAT", "message": "page images are PDF-only"}
        )

    cache_dir = Path(settings.data_dir) / "processed" / source_id / "pages"
    cached = cache_dir / f"{page_number}.png"
    if cached.exists():
        return FileResponse(cached, media_type="image/png")

    matches = sorted(Path(settings.data_dir).glob(f"raw/{source_id}.*"))
    if not matches:
        raise HTTPException(404, {"code": "NOT_FOUND", "message": "original PDF not stored"})

    import pymupdf as fitz

    doc = fitz.open(str(matches[0]))
    try:
        # page_number is 1-indexed everywhere in this project, matching
        # pymupdf4llm's own `page_number` metadata key (§7 item 4).
        if not 1 <= page_number <= doc.page_count:
            raise HTTPException(
                404,
                {
                    "code": "NOT_FOUND",
                    "message": f"page {page_number} is outside 1..{doc.page_count}",
                },
            )
        pixmap = doc[page_number - 1].get_pixmap(dpi=120)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(cached))
    finally:
        doc.close()
    return FileResponse(cached, media_type="image/png")


@router.get("/{source_id}/transcript", response_class=PlainTextResponse)
def get_transcript(source_id: str):
    """Downloadable transcript for audio/video/YouTube sources (CLAUDE.md §25)."""
    conn = get_connection()
    row = conn.execute(
        "SELECT source_type FROM sources WHERE source_id = ?", (source_id,)
    ).fetchone()
    conn.close()
    if row is None:
        not_found = {"code": "NOT_FOUND", "message": "source not found"}
        raise HTTPException(status_code=404, detail=not_found)
    if row["source_type"] not in ("audio", "video", "youtube"):
        detail = {
            "code": "UNSUPPORTED_FORMAT",
            "message": "transcripts are only available for audio/video/YouTube sources",
        }
        raise HTTPException(status_code=415, detail=detail)

    # Built from the ASR artifact, not from indexed chunks (CLAUDE.md §7
    # item 47). elements.json is what the transcriber actually produced; chunks
    # are a derived view, so a chunking change silently altered the downloadable
    # transcript, and the download was impossible before INDEXING finished even
    # though the transcript had existed since OCR_OR_ASR. Falls back to chunks
    # for sources ingested before the artifact was written.
    raw_elements = load_stage_artifact(source_id, "elements")
    if raw_elements:
        segments = sorted(
            (e for e in raw_elements if e.get("element_type") == "transcript_segment"),
            key=lambda e: e.get("start_time") or 0,
        )
    else:
        chunks = get_chunks_by_source(source_id)
        segments = sorted(
            (c for c in chunks if c.get("element_type") == "transcript_segment"),
            key=lambda c: c.get("start_time") or 0,
        )
    lines = []
    for seg in segments:
        start = seg.get("start_time") or 0
        speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        # The review annotation is rendered HERE, at the display layer, and
        # nowhere else. It used to be baked into the stored text, which put the
        # literal words into the embeddings (§15) - a human reading the
        # downloaded transcript still needs the warning, the index does not.
        flag = f" {REVIEW_ANNOTATION}" if seg.get("low_confidence") else ""
        # A merged segment holds more than one voice, so the label is the
        # dominant speaker, not the only one - say so rather than letting a
        # reader take "[Speaker 1]" as an attribution (§7 item 29).
        if seg.get("multi_speaker"):
            flag += " [MULTIPLE SPEAKERS]"
        lines.append(f"[{start:.1f}s] {speaker}{seg['text']}{flag}")
    return "\n".join(lines)
