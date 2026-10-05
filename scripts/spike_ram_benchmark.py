"""Phase 0 RAM-budget spike (CLAUDE.md §29).

§29's concurrency policy — "never keep more than roughly one other heavy model
resident at once" — has been reasoning from three components marked "Not
measured this session". This measures them.

Each model is loaded in a FRESH subprocess and its peak RSS recorded, because
measuring several in one process would attribute shared library memory
(torch, numpy, CUDA stubs) to whichever loaded first and understate the rest.
The baseline subprocess measures the interpreter plus imports alone, so each
figure below is the model's own marginal cost.

Run with nothing else heavy in flight (§7 item 9): a concurrent ingestion both
skews the numbers and can OOM the measurement outright.

    venv\\Scripts\\python.exe scripts/spike_ram_benchmark.py
"""

import subprocess
import sys
from pathlib import Path

# Each entry loads exactly one model, then prints its peak RSS in MB.
PROBES: dict[str, str] = {
    "baseline (interpreter + psutil only)": "pass",
    "faster-whisper small (int8)": (
        "from faster_whisper import WhisperModel; "
        "m = WhisperModel('small', device='cpu', compute_type='int8')"
    ),
    "BGE-reranker-v2-m3 (PyTorch)": (
        "from FlagEmbedding import FlagReranker; "
        "m = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False, devices=['cpu']); "
        "m.compute_score([('a','b')], normalize=True)"
    ),
    "BGE-M3 embeddings": (
        "from FlagEmbedding import BGEM3FlagModel; "
        "m = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, devices=['cpu']); "
        "m.encode(['warm up'], return_dense=True, return_sparse=True, return_colbert_vecs=False)"
    ),
    "PaddleOCR (PP-OCRv6)": (
        "from rag_assistant.ingestion.ocr import _get_paddle_ocr; _get_paddle_ocr()"
    ),
    "pyannote 4.0.7 diarization": (
        "from rag_assistant.config import settings; "
        "from rag_assistant.ingestion.diarization import _get_pipeline; _get_pipeline()"
    ),
}

_TEMPLATE = """
import os, sys
import psutil
{body}
proc = psutil.Process(os.getpid())
peak = getattr(proc.memory_info(), 'peak_wset', proc.memory_info().rss)
print(round(peak / (1024 * 1024)))
"""


def measure(body: str) -> int | None:
    """Peak RSS of a subprocess that loads one model, in MB."""
    result = subprocess.run(
        [sys.executable, "-c", _TEMPLATE.format(body=body)],
        capture_output=True,
        text=True,
        # Model loaders print progress bars containing non-cp1252 bytes; without
        # an explicit codec Python decodes the pipe with the Windows ANSI
        # codepage and the reader thread dies with UnicodeDecodeError mid-run.
        encoding="utf-8",
        errors="replace",
        cwd=Path(__file__).parent.parent,
        timeout=1800,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()
        print(f"    FAILED: {tail[-1] if tail else 'unknown error'}")
        return None
    return int(result.stdout.strip().splitlines()[-1])


def main() -> int:
    print("Peak RSS per model, each in its own fresh process (CLAUDE.md §29).")
    print("Run this with no ingestion in flight, or the numbers are meaningless.\n")

    baseline = None
    for label, body in PROBES.items():
        print(f"  measuring: {label} ...")
        peak = measure(body)
        if peak is None:
            continue
        if baseline is None:
            baseline = peak
            print(f"    {peak:5d} MB total (this is the baseline)")
        else:
            print(f"    {peak:5d} MB total | {peak - baseline:5d} MB marginal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
