import asyncio

from rag_assistant.api.main import _ignore_benign_disconnects


class _RecordingLoop:
    def __init__(self):
        self.handled = []

    def default_exception_handler(self, context):
        self.handled.append(context)


def test_windows_disconnect_noise_is_dropped():
    """WinError 10054/10053 fire from the proactor AFTER a response completes,
    when the client has already gone. They print a full traceback that trains
    the reader to ignore tracebacks - the habit that hid real failures here."""
    loop = _RecordingLoop()

    _ignore_benign_disconnects(loop, {"exception": ConnectionResetError(10054, "forcibly closed")})
    _ignore_benign_disconnects(loop, {"exception": ConnectionAbortedError(10053, "aborted")})

    assert loop.handled == []


def test_real_errors_still_reach_the_default_handler():
    """The suppression must stay narrow - anything that is not a client
    disconnect has to keep surfacing."""
    loop = _RecordingLoop()

    _ignore_benign_disconnects(loop, {"exception": RuntimeError("something real broke")})
    _ignore_benign_disconnects(loop, {"message": "no exception object at all"})

    assert len(loop.handled) == 2


def test_lifespan_installs_the_handler():
    from fastapi import FastAPI

    from rag_assistant.api.main import lifespan

    # A real app object, because lifespan now also stashes the startup
    # environment report on app.state (CLAUDE.md §6).
    app = FastAPI()

    async def exercise():
        async with lifespan(app):
            handler = asyncio.get_running_loop().get_exception_handler()
            assert handler is _ignore_benign_disconnects

    asyncio.run(exercise())


def test_lifespan_records_the_environment_report():
    """§6's whole point is that an environment problem is diagnosable as such
    at startup, instead of surfacing mid-pipeline the way ffmpeg's WinError 2
    did to a real user (§7 item 11)."""
    from fastapi import FastAPI

    from rag_assistant.api.main import lifespan

    app = FastAPI()

    async def exercise():
        async with lifespan(app):
            env = app.state.environment
            assert env["cores_logical"] >= 1
            assert "ffmpeg_ok" in env and "ollama_ok" in env
            assert env["ram_total_gb"] > 0

    asyncio.run(exercise())


def test_environment_check_never_breaks_startup(monkeypatch):
    """A diagnostic that takes down the server it is diagnosing is worse than
    no diagnostic."""
    from rag_assistant import environment as env_module

    def boom():
        raise RuntimeError("psutil exploded")

    monkeypatch.setattr(env_module, "describe_environment", boom)

    assert env_module.log_environment() == {}


def test_a_clean_shutdown_is_recorded_so_a_kill_is_distinguishable(caplog):
    """CLAUDE.md §7 item 59 - the one question a vanished server has to answer.

    A user reported the server "just stopping". Windows recorded no fault and
    the logs went only to a console that was gone, so a Ctrl+C, a forced
    Stop-Process and a native crash all left identical evidence: none. These
    two lines are the entire diagnosis - a log ending in "shutting down
    cleanly" was a signal or a closed console, and a log that simply STOPS
    mid-stream was a forced kill or a crash.
    """
    import asyncio
    import logging

    from fastapi import FastAPI

    from rag_assistant.api.main import lifespan

    app = FastAPI()

    async def exercise():
        async with lifespan(app):
            pass

    with caplog.at_level(logging.INFO, logger="rag_assistant.api.main"):
        asyncio.run(exercise())

    messages = [r.message for r in caplog.records]
    assert any("ready" in m and "pid=" in m for m in messages), "startup must be recorded"
    assert any("shutting down cleanly" in m for m in messages), (
        "without this line, an orderly exit is indistinguishable from a kill"
    )
