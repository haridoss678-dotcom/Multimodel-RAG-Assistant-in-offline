# Local Multimodal RAG Assistant

A locally hosted assistant for asking questions about your own PDFs, audio,
video, and YouTube sources. It indexes sources for retrieval and returns
answers with page or timestamp citations. The default answer model is
Qwen3 0.6B served locally by Ollama.

## What it supports

- PDF text extraction, with OCR for scanned pages.
- Audio transcription and timestamp-aware retrieval.
- Video questions about the **audio track transcript only**. It does not
  inspect frames, slides, or on-screen text.
- YouTube captions, with audio download/transcription as a fallback.
- Hybrid vector retrieval, reranking, conversation history, and source-scoped
  queries.
- A browser UI and API served by the same local FastAPI process.

This project targets Windows and Python 3.11. Model downloads and inference can
use substantial disk space, memory, and CPU time. On a CPU-only machine, the
first model loads and long audio jobs can take a while.

## Prerequisites

1. **Python 3.11** available as the Windows `py` launcher command.
2. **Ollama for Windows** for local Qwen inference.
3. **FFmpeg** for audio/video extraction and YouTube audio fallback. For
   example, install it with `winget install Gyan.FFmpeg`.
4. **Tesseract OCR** is optional; it is the fallback if the primary OCR engine
   cannot process a page. Install the Windows application separately and make
   sure `tesseract.exe` is on `PATH`. The Python package alone is not enough.
5. **Hugging Face account/token** is optional for most features, but required
   for speaker diarization. Accept the terms for the pyannote diarization model
   and its dependent models, then set `HF_TOKEN` in a local `.env` file.

The first use may download model weights from Ollama, Hugging Face, and the
OCR/ASR model providers. These downloads are separate from the Python packages.

## Install

Open PowerShell in the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:0.6b
```

`requirements.txt` installs the project and its Python dependencies. Ollama,
FFmpeg, and the Tesseract application are separate system prerequisites.

## Run on Windows

PowerShell may block unsigned scripts by default. In the same PowerShell
window, allow scripts for that window only and start the server:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run_server.ps1
```

This does not change the machine-wide execution policy. If your organization
enforces an execution policy, it may prevent this override; ask your system
administrator rather than changing a machine-wide policy.

If you prefer not to change the current shell's policy, start a child Windows
PowerShell process with a process-scoped bypass:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_server.ps1
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The API documentation is
at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), and
[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) reports server
health and detected environment details.

The launcher uses `.venv\Scripts\python.exe` when present (and falls back to
`venv\Scripts\python.exe`), checks for an existing RAG server, and writes
rotating logs to `logs\server.log`. Stop it with **Ctrl+C in the server
terminal**. Alternatively, from another PowerShell window run:

```powershell
.\scripts\run_server.ps1 -Stop
```

Only one process can use the embedded Qdrant storage at a time. Stop the
server before starting another process that accesses the same project storage.

## Use the assistant

1. Open the local UI.
2. Upload a supported file or submit a YouTube URL.
3. Wait for the source's job to reach `READY`.
4. Select the source in the UI to scope your question to it.
5. Ask a question and follow its page/timestamp citations.

Audio/video transcripts and indexed sources are stored locally in the project.
YouTube features contact YouTube, and model setup downloads weights from
Ollama/Hugging Face. The application does not use a hosted LLM API by default.

## Optional diarization configuration

Create a local `.env` file (it is ignored by Git) and add your Hugging Face
token:

```dotenv
HF_TOKEN=hf_your_token_here
```

Accept the model terms on Hugging Face before using diarization. Without the
token and accepted terms, ingestion continues without speaker labels.

## Tests and development

Run the fast unit and API tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit tests\api
```

Run lint checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

The end-to-end evaluation tests use real local models and can take several
minutes and significant memory. Do not run them while an ingestion job is
active:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Project data and privacy

Runtime uploads and indexes are written under `data\raw`,
`data\processed`, and `storage`; server logs are under `logs`. These are local
runtime state, not source files, and are excluded from Git. Keep your own
backups if you need to preserve them.

For multimodal sources, ingestion may download model weights. YouTube ingestion
uses the network to fetch captions or, if necessary, audio. Once available,
transcription, indexing, retrieval, and Qwen generation run locally.

## Publishing to GitHub

This checkout does not currently have a Git repository initialized. Before
publishing, confirm that you are comfortable making all tracked source and
documentation public. In particular:

- **Do not commit `.env`**; it can contain `HF_TOKEN`. `.env` is ignored.
- Do not commit `.venv`, `venv`, model caches, uploaded documents, generated
  indexes/databases, or logs. The ignore rules exclude these.
- Review `data\eval` and `tests\fixtures` to ensure their test content is safe
  to publish.
- `CLAUDE.md` and `PROJECT_GUIDE.md` contain extensive project notes and
  machine-specific measurements. Keep them only if you intend to publish
  those notes; otherwise remove them from the public project or add them to
  `.gitignore` before staging.
- Choose a license before publishing if you want to grant others permission
  to use or modify the code. No license is added automatically here.

Create an **empty** repository on GitHub (do not initialize it there with a
README), then in PowerShell replace the remote URL with your repository URL:

```powershell
git init -b main
git add .
git status --short
git diff --cached --name-only
git diff --cached --stat
```

Review the staged file list carefully. Confirm `.env`, local databases, logs,
uploads, and virtual environments are not staged. Then commit and push:

```powershell
git commit -m "Initial project"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If a token or private document was ever committed, removing the file in a later
commit is not enough: revoke/rotate the credential and remove the sensitive
content from Git history before publishing.
