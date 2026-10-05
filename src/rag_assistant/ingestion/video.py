from rag_assistant.ingestion.audio import extract_audio_to_wav, process_wav
from rag_assistant.ingestion.models import ContentElement, IngestOptions


def process_video(
    file_bytes: bytes,
    suffix: str,
    on_progress=None,
    options: IngestOptions | None = None,
) -> list[ContentElement]:
    """video -> ffmpeg audio extraction -> the same audio pipeline (CLAUDE.md
    §8 - audio-track-only; no frame/visual understanding)."""
    wav_path = extract_audio_to_wav(file_bytes, suffix)
    return process_wav(wav_path, on_progress=on_progress, options=options)
