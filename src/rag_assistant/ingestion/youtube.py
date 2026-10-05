import logging
import re

from rag_assistant.ingestion.audio import process_wav
from rag_assistant.ingestion.language_detect import detect_language
from rag_assistant.ingestion.models import ContentElement, IngestOptions

logger = logging.getLogger(__name__)


class YouTubeError(Exception):
    pass


class CaptionsUnavailable(Exception):
    """No captions exist for this video - expected, fall through to ASR."""


def extract_video_id(url: str) -> str:
    """Normalises any YouTube URL form to its 11-character video id.

    Also the basis of the source_id (CLAUDE.md §7 item 37). Hashing the raw URL
    meant youtu.be/X, youtube.com/watch?v=X, the same link with &t=30s, and any
    tracking parameter each produced a DIFFERENT source_id for the SAME video -
    defeating the duplicate detection that §19.2 exists to provide.
    """
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise YouTubeError(f"Could not parse a video ID from {url!r}")
    return match.group(1)


def _fetch_captions(video_id: str) -> list[ContentElement]:
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript = YouTubeTranscriptApi().fetch(video_id)
    return [
        ContentElement(
            element_type="transcript_segment",
            text=snippet.text,
            start_time=snippet.start,
            end_time=snippet.start + snippet.duration,
            # Set here for the same reason the PDF path sets it: without this
            # every caption chunk carried language=None and §17's indexed
            # `language` field had nothing to filter (CLAUDE.md §7 item 38).
            language=detect_language(snippet.text),
        )
        for snippet in transcript
    ]


def process_youtube(url: str, options: IngestOptions | None = None) -> list[ContentElement]:
    """Existing captions first, ASR fallback only if none exist (CLAUDE.md
    §13 - a pure CPU-saving step: captions already carry timestamps, so a
    2-hour video indexes in seconds instead of ~40 minutes of ASR)."""
    video_id = extract_video_id(url)

    try:
        return _fetch_captions(video_id)
    except Exception as exc:
        # A bare `except: pass` here swallowed genuine transport failures and
        # HTTP 429 rate limits, then silently burned the expensive download+ASR
        # path as if the video simply had no captions (CLAUDE.md §7 item 39).
        # "No captions" is expected and quiet; anything else is logged loudly,
        # because falling through costs tens of minutes of CPU.
        detail = f"{type(exc).__name__}: {exc}"
        if "429" in detail or "Too Many Requests" in detail:
            raise YouTubeError(
                f"YouTube rate-limited the captions request for {video_id} ({detail}). "
                "Retry later rather than spending an ASR pass on it."
            ) from exc
        logger.info(
            "no usable captions for %s (%s) - falling back to download + ASR", video_id, detail
        )

    import tempfile
    from pathlib import Path

    import yt_dlp

    tmp_dir = tempfile.mkdtemp()
    out_template = str(Path(tmp_dir) / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise YouTubeError(f"Could not download audio for {url}: {exc}") from exc

    wav_files = list(Path(tmp_dir).glob("*.wav"))
    if not wav_files:
        raise YouTubeError(f"yt-dlp produced no audio file for {url}")
    # Carries the caller's speech settings instead of silently defaulting to
    # auto-detect, which measured RTF 4.44 AND wrong output on code-mixed
    # speech - so a Tamil video without captions took the 15x-worse path
    # (CLAUDE.md §7 item 40).
    return process_wav(str(wav_files[0]), options=options)
