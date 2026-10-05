import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from rag_assistant.ingestion.asr import transcribe
from rag_assistant.ingestion.diarization import diarize, speaker_span
from rag_assistant.ingestion.models import ContentElement, IngestOptions


class AudioExtractionError(Exception):
    pass


_ffmpeg_path: str | None = None

# Real, confirmed-needed fallback locations (CLAUDE.md §7 item 8): a tool
# being in the persistent registry PATH does not guarantee a given process
# actually sees it - a real user hit exactly this (WinError 2, ffmpeg not
# found) running the server from a terminal opened before the PATH was
# updated. Resolving a concrete path once, with known-install-location
# fallbacks, is more robust than assuming PATH is correct at call time.
_FALLBACK_GLOBS = [
    r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]


def _find_ffmpeg() -> str:
    global _ffmpeg_path
    if _ffmpeg_path is not None:
        return _ffmpeg_path

    on_path = shutil.which("ffmpeg")
    if on_path:
        _ffmpeg_path = on_path
        return _ffmpeg_path

    for pattern in _FALLBACK_GLOBS:
        relative_pattern = pattern[3:].replace("\\", "/")  # strip the "C:\" drive prefix
        matches = sorted(Path("C:/").glob(relative_pattern), reverse=True)
        if matches:
            _ffmpeg_path = str(matches[0])
            return _ffmpeg_path

    raise AudioExtractionError(
        "ffmpeg was not found on PATH or in any known install location. "
        "Install it (winget install Gyan.FFmpeg) and restart the server "
        "process so it picks up the updated PATH."
    )


def process_wav(
    wav_path: str,
    on_progress=None,
    options: IngestOptions | None = None,
) -> list[ContentElement]:
    """Shared by both the plain-audio and video ingestion paths (CLAUDE.md
    §8 - video is audio-track-only, so once extracted it's identical).

    on_progress receives a 0.0-1.0 fraction as transcription advances, so a
    long recording reports real progress instead of one frozen percentage."""
    elements = transcribe(wav_path, on_progress=on_progress, options=options)
    turns = diarize(wav_path)  # None if unavailable (§27) - graceful, not fatal
    for el in elements:
        # Maximum-overlap assignment plus an honest multi-speaker flag, rather
        # than "whoever spoke at this segment's start time" - see speaker_span().
        el.speaker, el.multi_speaker = speaker_span(turns, el.start_time, el.end_time)
    return elements


def extract_audio_to_wav(input_bytes: bytes, suffix: str) -> str:
    """ffmpeg extraction to mono 16kHz wav (CLAUDE.md §13) - the format
    faster-whisper expects. `suffix` is the original file's extension
    (.mp3/.mp4/...) so ffmpeg can detect the input container correctly."""
    ffmpeg = _find_ffmpeg()
    tmp_dir = Path(tempfile.gettempdir())
    in_path = tmp_dir / f"{uuid.uuid4()}{suffix}"
    out_path = tmp_dir / f"{uuid.uuid4()}.wav"
    in_path.write_bytes(input_bytes)
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(in_path), "-ac", "1", "-ar", "16000", str(out_path)],
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0 or not out_path.exists():
            raise AudioExtractionError(result.stderr.decode(errors="replace")[-2000:])
        return str(out_path)
    finally:
        in_path.unlink(missing_ok=True)
