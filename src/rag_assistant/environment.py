"""Startup environment discovery (CLAUDE.md §6, previously unimplemented).

§6 specifies detecting RAM, core count, GPU absence, required binaries and
whether the process is running from the project venv - and none of it existed.
That absence is exactly how the ffmpeg `WinError 2` reached a real user
mid-pipeline instead of at startup (§7 item 11): the failure was discoverable
before any work began, and nothing looked.

Everything here is CHEAP and NON-FATAL. It never imports torch (2.3s and
hundreds of MB just to ask about a GPU this project does not use), never loads
a model, and never raises - a diagnostic that takes down the server it is
diagnosing is worse than no diagnostic.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import httpx
import psutil

from rag_assistant.config import settings

logger = logging.getLogger(__name__)

# Weights the pipeline needs. Absence is not an error - they download on first
# use - but knowing about it up front explains an otherwise mysterious
# multi-minute first request.
_EXPECTED_MODELS = {
    "BAAI/bge-m3": "models--BAAI--bge-m3",
    "BAAI/bge-reranker-v2-m3": "models--BAAI--bge-reranker-v2-m3",
    "faster-whisper small": "models--Systran--faster-whisper-small",
}


def _hf_cache_dir() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _ffmpeg_status() -> tuple[bool, str]:
    from rag_assistant.ingestion.audio import AudioExtractionError, _find_ffmpeg

    try:
        return True, _find_ffmpeg()
    except AudioExtractionError as exc:
        return False, str(exc)


def _ollama_status() -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001 - network boundary, never fatal
        return False, f"unreachable at {settings.ollama_host} ({type(exc).__name__})"
    if settings.llm_model not in names:
        return False, f"reachable, but {settings.llm_model!r} is not pulled (have: {names})"
    return True, f"{settings.llm_model} available"


def describe_environment() -> dict:
    """Collects the facts. Returned as a dict so /health can serve it too."""
    memory = psutil.virtual_memory()
    venv_python = (Path.cwd() / "venv" / "Scripts" / "python.exe").resolve()
    running_python = Path(sys.executable).resolve()

    ffmpeg_ok, ffmpeg_detail = _ffmpeg_status()
    ollama_ok, ollama_detail = _ollama_status()

    cache = _hf_cache_dir()
    models = {name: (cache / folder).exists() for name, folder in _EXPECTED_MODELS.items()}

    storage = Path(settings.qdrant_path).parent
    storage.mkdir(parents=True, exist_ok=True)

    return {
        "python": str(running_python),
        # §7 item 1: a uv-managed global python intercepts `python`/`pip` on
        # this machine, so running outside the venv is a real, silent failure
        # mode rather than a hypothetical one.
        "in_project_venv": running_python == venv_python,
        "ram_total_gb": round(memory.total / 1e9, 2),
        "ram_available_gb": round(memory.available / 1e9, 2),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        # Detected without importing torch. The pipeline is CPU-only by
        # construction anyway (device="cpu" / devices=["cpu"] are hardcoded), so
        # this is informational: a present GPU is deliberately unused (§9).
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "ffmpeg_ok": ffmpeg_ok,
        "ffmpeg": ffmpeg_detail,
        "ollama_ok": ollama_ok,
        "ollama": ollama_detail,
        "models_cached": models,
        "disk_free_gb": round(shutil.disk_usage(storage).free / 1e9, 2),
        "llm_num_ctx": settings.llm_num_ctx,
        "asr_batch_size": settings.asr_batch_size,
        "asr_chunk_length": settings.asr_chunk_length,
    }


def log_environment() -> dict:
    """Logs the findings once at startup, warning only on real problems."""
    try:
        env = describe_environment()
    except Exception:  # noqa: BLE001 - diagnostics must never break startup
        logger.warning("environment check failed", exc_info=True)
        return {}

    logger.info(
        "environment: %.1fGB RAM free of %.1fGB | %s physical / %s logical cores"
        " | %.1fGB disk free",
        env["ram_available_gb"],
        env["ram_total_gb"],
        env["cores_physical"],
        env["cores_logical"],
        env["disk_free_gb"],
    )
    logger.info(
        "environment: num_ctx=%s asr_batch=%s asr_chunk_length=%ss",
        env["llm_num_ctx"],
        env["asr_batch_size"],
        env["asr_chunk_length"],
    )

    if not env["in_project_venv"]:
        logger.warning(
            "NOT running from the project venv (%s) - a global interpreter will not have this "
            "project's pinned dependencies (CLAUDE.md §7 item 1)",
            env["python"],
        )
    # Measured: latency degrades sharply below ~1GB free, and a second model
    # load can fail outright with mkl_malloc (§7 item 9).
    if env["ram_available_gb"] < 1.0:
        logger.warning(
            "only %.2fGB RAM free - answers slow sharply below ~1GB and a model load can fail "
            "outright; closing other applications genuinely helps here",
            env["ram_available_gb"],
        )
    if not env["ffmpeg_ok"]:
        logger.warning(
            "ffmpeg NOT resolvable - audio and video ingestion will fail: %s", env["ffmpeg"]
        )
    if not env["ollama_ok"]:
        logger.warning("Ollama problem - chat will fail: %s", env["ollama"])

    missing = [name for name, cached in env["models_cached"].items() if not cached]
    if missing:
        logger.info(
            "model weights not yet cached: %s - they download on first use, which is why a "
            "first request can take minutes",
            ", ".join(missing),
        )
    if env["nvidia_smi_present"]:
        logger.info("an NVIDIA GPU is present but deliberately unused - this build is CPU-only")
    return env
