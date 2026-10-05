import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from rag_assistant.config import settings
from rag_assistant.storage.db import get_connection

# CLAUDE.md §19.1 - full state list is
# UPLOADED -> VALIDATING -> EXTRACTING -> OCR_OR_ASR -> STRUCTURING -> CHUNKING
#   -> EMBEDDING -> INDEXING -> READY, with FAILED/CANCELLED at any stage.
# V2 adds OCR_OR_ASR (audio/video/YouTube transcription - the expensive step
# that must not be redone on resume). PDF's OCR still happens inline within
# EXTRACTING (§13's per-page routing) rather than as a separate state - see
# pipeline.py's _extract() for why. STRUCTURING (full Docling structure
# detection) is not yet implemented (§32 V2 scope call #2) - still skipped.
UPLOADED = "UPLOADED"
VALIDATING = "VALIDATING"
EXTRACTING = "EXTRACTING"
OCR_OR_ASR = "OCR_OR_ASR"
CHUNKING = "CHUNKING"
EMBEDDING = "EMBEDDING"
INDEXING = "INDEXING"
READY = "READY"
FAILED = "FAILED"
CANCELLED = "CANCELLED"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def compute_source_id(file_bytes: bytes) -> str:
    """Content-hash source identity (CLAUDE.md §19.2) - used for duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_existing_state(source_id: str) -> str | None:
    conn = get_connection()
    row = conn.execute("SELECT state FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    conn.close()
    return row["state"] if row else None


def is_cancelled(source_id: str) -> bool:
    """True when an in-flight ingestion should stop and write nothing further.

    Covers BOTH ways a job can be orphaned (CLAUDE.md §7 item 35):
      - the source was explicitly cancelled (state == CANCELLED), and
      - the source row is simply GONE, because DELETE /sources/{id} ran the
        cascade while the background worker was still mid-pipeline.

    The second case is the bug this closes. The worker kept going, and its next
    `_update_job` INSERT ... ON CONFLICT re-created the job row it had just had
    deleted, then embedding re-inserted vectors for a source that no longer
    existed - leaving rows and vectors with no parent.
    """
    state = get_existing_state(source_id)
    return state is None or state == CANCELLED


def request_cancel(source_id: str) -> None:
    """Marks a source cancelled so an in-flight worker stops at its next stage
    boundary. Safe to call whether or not a job is actually running."""
    conn = get_connection()
    conn.execute(
        "UPDATE sources SET state = ?, updated_at = ? WHERE source_id = ?",
        (CANCELLED, _now(), source_id),
    )
    conn.commit()
    conn.close()


def source_exists(source_id: str) -> bool:
    return get_existing_state(source_id) is not None


def register_source(source_id: str, filename: str, source_type: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO sources (source_id, filename, source_type, state, "
        "parser_version, chunking_version, embedding_model_version, reranker_version, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id,
            filename,
            source_type,
            UPLOADED,
            settings.parser_version,
            settings.chunking_version,
            settings.embedding_model_version,
            # reranker_version was declared and never written (§7 item 42). It
            # is a constant today, but recording it is what makes a future
            # reranker change traceable to the answers produced before it.
            settings.reranker_version,
            _now(),
            _now(),
        ),
    )
    conn.commit()
    conn.close()


def set_state(
    source_id: str,
    state: str,
    failed_stage: str | None = None,
    failure_reason: str | None = None,
) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE sources SET state = ?, failed_stage = ?, failure_reason = ?, "
        "updated_at = ? WHERE source_id = ?",
        (state, failed_stage, failure_reason, _now(), source_id),
    )
    conn.commit()
    conn.close()


def record_ocr_version(source_id: str, ocr_version: str) -> None:
    """Records which OCR engine produced a scanned page's text.

    The column existed and was never written (CLAUDE.md §7 item 42), so a
    source OCR'd by the Tesseract fallback was indistinguishable from one read
    from a real text layer - which matters now that OCR confidence is captured
    and the engine differs in quality per language."""
    conn = get_connection()
    conn.execute(
        "UPDATE sources SET ocr_version = ?, updated_at = ? WHERE source_id = ?",
        (ocr_version, _now(), source_id),
    )
    conn.commit()
    conn.close()


def record_asr_version(source_id: str, asr_version: str) -> None:
    """Records which ASR mode produced this source's text (§19.4's version
    fields). Matters for interpretation, not just provenance: a translated
    transcript reads as English even though the speech was Tamil, so without
    this there is no way to tell a translation from a native English recording
    after the fact."""
    conn = get_connection()
    conn.execute(
        "UPDATE sources SET asr_version = ?, updated_at = ? WHERE source_id = ?",
        (asr_version, _now(), source_id),
    )
    conn.commit()
    conn.close()


def _artifact_dir(source_id: str) -> Path:
    d = Path(settings.data_dir) / "processed" / source_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_stage_artifact(source_id: str, stage: str, data: object) -> None:
    """Persist a stage's output before advancing state (CLAUDE.md §19.1) so a
    crash after this stage resumes here instead of redoing earlier work."""
    path = _artifact_dir(source_id) / f"{stage}.json"
    path.write_text(json.dumps(data), encoding="utf-8")


def load_stage_artifact(source_id: str, stage: str) -> object | None:
    path = _artifact_dir(source_id) / f"{stage}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
