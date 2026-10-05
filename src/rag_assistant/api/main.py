import asyncio
import logging
import logging.handlers
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_assistant.api.routes import chat, conversations, jobs, sources
from rag_assistant.environment import describe_environment, log_environment

# Without this, an unhandled exception's traceback is easy to miss in the
# terminal (CLAUDE.md §30's observability requirement needs it visible at
# minimum, even before a dedicated trace_records query tool exists).
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)

# ...and a FILE copy, because the console one disappears with the console.
#
# CLAUDE.md §7 item 59. A real user reported the server "just stopping" and
# there was nothing to look at: logs went only to stdout, so a closed terminal,
# a Stop-Process, or a hard fault all left byte-for-byte the same evidence -
# none. Windows recorded no fault either, which meant the one question that
# mattered ("did it exit, or was it killed?") had no answer at all.
#
# The startup and shutdown lines below are what make the file diagnostic rather
# than merely verbose: a log ending in "shutting down cleanly" was a Ctrl+C or
# a closed console, and a log that simply STOPS was a forced kill or a native
# crash. That single distinction is the whole diagnosis.
#
# Rotating, because an unbounded log on a machine this size is its own bug.
_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "server.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger(__name__)


def _ignore_benign_disconnects(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Silences the Windows proactor's disconnect noise, and nothing else.

    On Windows, asyncio's ProactorEventLoop closes a transport by calling
    socket.shutdown() in _call_connection_lost(). When the client has already
    gone - a Streamlit rerun, a browser refresh, a page closed mid-answer, or a
    client-side timeout - the socket is dead and that call raises
    ConnectionResetError (WinError 10054) or ConnectionAbortedError (10053).

    It fires *after* the response is finished, from the event loop's unhandled-
    exception path, so it never affects the request: it is pure log noise. But
    it prints a full traceback every time, which trains the reader to ignore
    tracebacks - exactly the habit that let the earlier ffmpeg and Qdrant-lock
    failures hide in this project's logs.

    Deliberately narrow: only these two exception types are dropped, and
    everything else is handed to the default handler untouched.
    """
    exception = context.get("exception")
    if isinstance(exception, ConnectionResetError | ConnectionAbortedError):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Must be installed on the *running* loop, which only exists once the
    # server has started - hence lifespan rather than import time.
    asyncio.get_running_loop().set_exception_handler(_ignore_benign_disconnects)
    # CLAUDE.md §6, finally implemented. Its absence is how the ffmpeg
    # WinError 2 reached a real user mid-pipeline rather than at startup.
    app.state.environment = log_environment()
    logger.info(
        "RAG Assistant API ready - pid=%s started_at=%s",
        os.getpid(),
        datetime.now(UTC).isoformat(timespec="seconds"),
    )
    try:
        yield
    finally:
        # Reached on Ctrl+C and on a closed console; NOT reached when the
        # process is force-killed or crashes natively. Its presence or absence
        # in logs/server.log is the evidence that says which happened.
        logger.info("RAG Assistant API shutting down cleanly - pid=%s", os.getpid())


app = FastAPI(title="RAG Assistant", lifespan=lifespan)

app.include_router(sources.router)
app.include_router(jobs.router)
app.include_router(chat.router)
app.include_router(conversations.router)

# The UI is served by this same process: one port, one URL, and one fewer
# Python process on a machine where Streamlit's own ~100-200MB RSS is not free.
# Mounted AFTER the routers so a static file can never shadow an API route.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/health")
def health():
    """Liveness plus the startup environment facts.

    Serving them here means "why is this slow / why did audio fail" is
    answerable without reading server logs - the diagnosis §6 asks for, made
    reachable. Recomputed on demand so free RAM is current rather than whatever
    it was at boot."""
    return {"status": "ok", "environment": describe_environment()}
