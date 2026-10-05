import io

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _reset_graph_singletons() -> None:
    """Drops build_graph's process-lifetime singletons.

    build_graph holds the compiled graph AND an open SqliteSaver connection for
    the life of the process (deliberately - see its own note on the context
    manager that must stay referenced). A test that patches
    langgraph_checkpoint_path after the saver is already open would be ignored,
    so the singletons have to be cleared on the way in as well as the way out.
    """
    from rag_assistant.graph import build_graph

    build_graph._graph = None
    build_graph._checkpointer = None
    build_graph._checkpointer_cm = None


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Points all storage at a temp dir so tests never touch real project
    data (real storage/, data/).

    Includes langgraph_checkpoint_path: without it, any test that invokes the
    graph wrote execution state into the REAL storage/langgraph_checkpoints.db
    and left a connection to it open for the rest of the pytest process - which
    also made that file grow on every test run (CLAUDE.md §7 item 23)."""
    from rag_assistant import config
    from rag_assistant.storage import vector_store

    monkeypatch.setattr(config.settings, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(config.settings, "sqlite_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(config.settings, "qdrant_path", str(tmp_path / "qdrant"))
    monkeypatch.setattr(
        config.settings, "langgraph_checkpoint_path", str(tmp_path / "checkpoints.db")
    )
    vector_store._client = None
    _reset_graph_singletons()
    yield tmp_path
    vector_store._client = None
    _reset_graph_singletons()


def _make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        y = 750
        for line in text.splitlines():
            c.drawString(72, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """A tiny, real, synthetic 2-page PDF with known content and a fact
    that only appears on page 2 - used to check page-accurate citations."""
    return _make_pdf(
        [
            "Introduction to Python\nThis document introduces basic Python concepts.",
            "Decorators\nA decorator is a function that wraps another function to "
            "extend its behavior without modifying it directly.",
        ]
    )
