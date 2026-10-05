"""Per-node progress for a single /chat turn (UI prompt A2).

/chat is deliberately not streamed - §26 explains why, and that decision is
correct: the graph can regenerate an answer twice and citations only resolve
after generation finishes, so streamed text could be retracted mid-flight. But
the consequence was a ~107s spinner with no feedback, and a spinner is
indistinguishable from a hang. That is the same lesson §7 item 17 already
learned for ingestion progress and never applied to chat.

So: not streamed output, but streamed PROGRESS. The graph reports which node it
is on; the client polls for it.

Deliberately in-process and bounded. This is ephemeral UI state for an
in-flight request - it is not conversation history (§4.3) and not graph
execution state (§4.2), so it belongs in neither SQLite file.
"""

import threading
import time
from collections import OrderedDict

# The nodes a turn can pass through, in order, with the measured cost of the
# expensive ones so the wait is legible rather than mysterious (§23.1).
NODE_SEQUENCE: tuple[str, ...] = (
    "classify",
    "rewrite",
    "decompose",
    "retrieve",
    "rerank",
    "expand_context",
    "summarize",
    "generate",
    "resolve_citations",
    "verify_grounding",
    "format_response",
)

NODE_LABELS: dict[str, str] = {
    "classify": "Understanding the question",
    "rewrite": "Resolving what you referred to",
    "decompose": "Splitting into sub-questions",
    "retrieve": "Searching your sources",
    "rerank": "Ranking the evidence",
    "expand_context": "Pulling surrounding context",
    "summarize": "Condensing oversized evidence",
    "generate": "Writing the answer",
    "resolve_citations": "Checking every citation",
    "verify_grounding": "Verifying the answer is grounded",
    "format_response": "Formatting",
}

# Bounded so a long-running server cannot accumulate entries for turns whose
# client went away. Oldest is dropped first.
_MAX_TRACKED = 64

_lock = threading.Lock()
_turns: OrderedDict[str, dict] = OrderedDict()


def start(request_id: str) -> None:
    if not request_id:
        return
    with _lock:
        _turns[request_id] = {
            "node": None,
            "index": 0,
            "total": len(NODE_SEQUENCE),
            "started_at": time.monotonic(),
        }
        _turns.move_to_end(request_id)
        while len(_turns) > _MAX_TRACKED:
            _turns.popitem(last=False)


def note(request_id: str | None, node: str) -> None:
    """Called as each node begins."""
    if not request_id:
        return
    with _lock:
        entry = _turns.get(request_id)
        if entry is None:
            return
        entry["node"] = node
        # Position in the canonical sequence, not a counter, so the retry loop
        # (generate -> verify -> generate) moves the label backwards honestly
        # instead of inventing progress it has not made.
        if node in NODE_SEQUENCE:
            entry["index"] = NODE_SEQUENCE.index(node) + 1


def get(request_id: str) -> dict | None:
    with _lock:
        entry = _turns.get(request_id)
        if entry is None:
            return None
        node = entry["node"]
        return {
            "node": node,
            "label": NODE_LABELS.get(node or "", "Starting"),
            "index": entry["index"],
            "total": entry["total"],
            "elapsed_seconds": round(time.monotonic() - entry["started_at"], 1),
        }


def finish(request_id: str | None) -> None:
    if not request_id:
        return
    with _lock:
        _turns.pop(request_id, None)
