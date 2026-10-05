from pathlib import Path

from rag_assistant.config import settings
from rag_assistant.ingestion import state_machine as sm
from rag_assistant.ingestion.audio import extract_audio_to_wav, process_wav
from rag_assistant.ingestion.chunking import chunk_elements
from rag_assistant.ingestion.embed_index import embed_and_index
from rag_assistant.ingestion.models import Chunk, ContentElement, IngestOptions
from rag_assistant.ingestion.pdf_parser import parse_pdf
from rag_assistant.ingestion.validation import (
    ValidationError,
    sniff_type,
    validate_file,
    validate_youtube_url,
)
from rag_assistant.ingestion.video import process_video
from rag_assistant.ingestion.youtube import extract_video_id, process_youtube
from rag_assistant.storage.db import get_connection


class IngestionCancelled(Exception):
    """Raised when a source is cancelled or deleted while its job is running."""


def _abort_if_cancelled(source_id: str) -> None:
    """Stage-boundary cancellation check (CLAUDE.md §7 item 35).

    Called before each stage so a deleted source stops costing CPU within one
    stage rather than running an entire ASR pass for nothing.
    """
    if sm.is_cancelled(source_id):
        raise IngestionCancelled(source_id)


def _update_job(source_id: str, stage: str, progress_pct: float, error: str | None = None) -> None:
    # Never resurrect a job row for a source that has been deleted. This INSERT
    # is ON CONFLICT DO UPDATE, so without the guard a worker that outlived its
    # source re-created the very row the delete cascade had just removed.
    if not sm.source_exists(source_id):
        return
    conn = get_connection()
    conn.execute(
        "INSERT INTO jobs (job_id, source_id, stage, progress_pct, error, started_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now')) "
        "ON CONFLICT(job_id) DO UPDATE SET "
        "stage=excluded.stage, progress_pct=excluded.progress_pct, "
        "error=excluded.error, updated_at=datetime('now')",
        (source_id, source_id, stage, progress_pct, error),
    )
    conn.commit()
    conn.close()


def _extract(
    source_id: str,
    source_type: str,
    file_bytes: bytes | None,
    filename_or_url: str,
    options: IngestOptions,
) -> list[ContentElement]:
    """The extraction dispatch. PDF's OCR happens inline within parse_pdf()
    (EXTRACTING covers it, CLAUDE.md's own per-page OCR routing, §13) - audio/
    video/YouTube instead split EXTRACTING (cheap: get to a wav file) from
    OCR_OR_ASR (expensive: transcribe) so a crash during the slow ASR step
    doesn't force redoing the fast extraction (CLAUDE.md §19.1's own stated
    reason for that state existing)."""
    if source_type == "pdf":
        sm.set_state(source_id, sm.EXTRACTING)
        _update_job(source_id, sm.EXTRACTING, 0.3)
        elements, ocr_version = parse_pdf(
            file_bytes, filename_or_url, ocr_language=options.ocr_language
        )
        if ocr_version:
            # Only set when OCR actually ran, so a NULL genuinely means "read
            # from the text layer" rather than "we forgot to record it".
            sm.record_ocr_version(source_id, ocr_version)
        return elements

    sm.set_state(source_id, sm.EXTRACTING)
    _update_job(source_id, sm.EXTRACTING, 0.2)
    wav_artifact = sm.load_stage_artifact(source_id, sm.EXTRACTING)
    # The saved wav lives in the OS temp dir, which Windows clears on its own
    # schedule. A resume days later can therefore find the artifact recorded
    # but the file gone - re-extract in that case rather than handing a dead
    # path to ffmpeg/ASR (§19.1's resume must degrade, not break).
    if wav_artifact is not None:
        recorded = wav_artifact.get("wav_path")
        if recorded and not Path(recorded).exists():
            wav_artifact = None

    if wav_artifact is None:
        if source_type == "audio":
            suffix = Path(filename_or_url).suffix
            wav_path = extract_audio_to_wav(file_bytes, suffix)
        elif source_type == "video":
            suffix = Path(filename_or_url).suffix
            wav_path = extract_audio_to_wav(file_bytes, suffix)
        else:
            wav_path = None  # youtube: process_youtube handles its own extraction
        sm.save_stage_artifact(source_id, sm.EXTRACTING, {"wav_path": wav_path})
    else:
        wav_path = wav_artifact.get("wav_path")

    sm.set_state(source_id, sm.OCR_OR_ASR)
    _update_job(source_id, sm.OCR_OR_ASR, 0.4)

    def on_asr_progress(fraction: float) -> None:
        # Maps transcription's own 0-1 onto the 0.40-0.75 slice of the overall
        # job, so the UI's progress bar actually advances during what is by far
        # the longest stage (a 28-minute recording took over an hour here).
        _update_job(source_id, sm.OCR_OR_ASR, 0.40 + 0.35 * fraction)

    if source_type in ("audio", "video"):
        sm.record_asr_version(source_id, options.version_tag)

    if source_type == "video":
        if wav_path is None:
            return process_video(
                file_bytes,
                Path(filename_or_url).suffix,
                on_progress=on_asr_progress,
                options=options,
            )
        return process_wav(wav_path, on_progress=on_asr_progress, options=options)
    if source_type == "audio":
        return process_wav(wav_path, on_progress=on_asr_progress, options=options)
    if source_type == "youtube":
        # Speech settings reach the ASR fallback path too - it used to default
        # to auto-detect, the slowest and least accurate option (§7 item 40).
        return process_youtube(filename_or_url, options=options)
    raise ValueError(f"Unknown source_type: {source_type}")


def ingest_file(
    file_bytes: bytes,
    filename: str,
    options: IngestOptions | None = None,
) -> str:
    """PDF/audio/video upload path (CLAUDE.md §32 V2). Resumable per-stage,
    same pattern as V1's ingest_pdf (§19.1) - now dispatching by source_type."""
    options = options or IngestOptions()
    source_id = sm.compute_source_id(file_bytes)
    existing_state = sm.get_existing_state(source_id)

    if existing_state == sm.READY:
        return source_id  # duplicate upload (§19.2)

    if existing_state is None:
        # Register with a provisional type first (mirrors V1's ordering) so a
        # validation failure below still has a row to mark FAILED against.
        sm.register_source(source_id, filename, sniff_type(filename))
        ext = Path(filename).suffix
        raw_path = Path(settings.data_dir) / "raw" / f"{source_id}{ext}"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(file_bytes)

    try:
        sm.set_state(source_id, sm.VALIDATING)
        _update_job(source_id, sm.VALIDATING, 0.1)
        source_type = validate_file(filename, file_bytes)

        elements = _run_extract_through_chunk(source_id, source_type, file_bytes, filename, options)
        _embed_index_and_finish(source_id, elements)
        return source_id
    except ValidationError as exc:
        sm.set_state(source_id, sm.FAILED, failed_stage=sm.VALIDATING, failure_reason=str(exc))
        _update_job(source_id, sm.FAILED, 0.0, error=str(exc))
        raise
    except IngestionCancelled:
        # Not a failure: the user deleted or cancelled this source. Writing
        # FAILED here would re-create the row the cascade just removed.
        raise
    except Exception as exc:
        current_state = sm.get_existing_state(source_id) or sm.EXTRACTING
        sm.set_state(source_id, sm.FAILED, failed_stage=current_state, failure_reason=str(exc))
        _update_job(source_id, sm.FAILED, 0.0, error=str(exc))
        raise


def ingest_youtube(url: str, options: IngestOptions | None = None) -> str:
    """YouTube URL path (CLAUDE.md §32/§13) - no file bytes, so identity comes
    from the NORMALISED video id rather than the raw URL.

    Hashing the URL meant youtu.be/X, youtube.com/watch?v=X, the same link with
    &t=30s, and any tracking parameter each produced a different source_id for
    the same video, defeating §19.2's duplicate detection (§7 item 37).
    """
    validate_youtube_url(url)
    options = options or IngestOptions()
    source_id = sm.compute_source_id(extract_video_id(url).encode("utf-8"))
    existing_state = sm.get_existing_state(source_id)

    if existing_state == sm.READY:
        return source_id

    try:
        if existing_state is None:
            sm.register_source(source_id, url, "youtube")

        elements = _run_extract_through_chunk(source_id, "youtube", None, url, options)
        _embed_index_and_finish(source_id, elements)
        return source_id
    except IngestionCancelled:
        raise
    except Exception as exc:
        current_state = sm.get_existing_state(source_id) or sm.EXTRACTING
        sm.set_state(source_id, sm.FAILED, failed_stage=current_state, failure_reason=str(exc))
        _update_job(source_id, sm.FAILED, 0.0, error=str(exc))
        raise


def _run_extract_through_chunk(
    source_id: str,
    source_type: str,
    file_bytes: bytes | None,
    filename_or_url: str,
    options: IngestOptions,
) -> list[Chunk]:
    _abort_if_cancelled(source_id)
    raw_elements = sm.load_stage_artifact(source_id, "elements")
    if raw_elements is None:
        elements = _extract(source_id, source_type, file_bytes, filename_or_url, options)
        sm.save_stage_artifact(source_id, "elements", [e.model_dump() for e in elements])
    else:
        elements = [ContentElement(**e) for e in raw_elements]

    _abort_if_cancelled(source_id)
    sm.set_state(source_id, sm.CHUNKING)
    _update_job(source_id, sm.CHUNKING, 0.6)
    raw_chunks = sm.load_stage_artifact(source_id, sm.CHUNKING)
    if raw_chunks is None:
        chunks = chunk_elements(elements, source_id)
        sm.save_stage_artifact(source_id, sm.CHUNKING, [c.model_dump() for c in chunks])
    else:
        chunks = [Chunk(**c) for c in raw_chunks]
    return chunks


def _embed_index_and_finish(source_id: str, chunks: list[Chunk]) -> None:
    # Checked immediately before the write that actually creates orphans: once
    # embed_and_index has run, vectors exist for a source whose rows may already
    # have been cascaded away.
    _abort_if_cancelled(source_id)
    sm.set_state(source_id, sm.EMBEDDING)
    _update_job(source_id, sm.EMBEDDING, 0.8)
    embed_and_index(chunks)

    sm.set_state(source_id, sm.INDEXING)
    _update_job(source_id, sm.INDEXING, 0.9)

    sm.set_state(source_id, sm.READY)
    _update_job(source_id, sm.READY, 1.0)


# Backward-compatible name for existing V1 tests/imports.
def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    return ingest_file(file_bytes, filename)
