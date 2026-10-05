import logging
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from rag_assistant.api import progress
from rag_assistant.config import settings
from rag_assistant.graph.nodes.citation import resolve_citations_node
from rag_assistant.graph.nodes.classify import classify_node
from rag_assistant.graph.nodes.context_expansion import expand_context_node
from rag_assistant.graph.nodes.decompose import decompose_node
from rag_assistant.graph.nodes.format_response import format_response_node
from rag_assistant.graph.nodes.generation import generate_node
from rag_assistant.graph.nodes.reranking import rerank_node
from rag_assistant.graph.nodes.retrieval import retrieve_node
from rag_assistant.graph.nodes.rewrite import looks_like_followup, rewrite_node
from rag_assistant.graph.nodes.summarize import _total_chars, max_evidence_chars, summarize_node
from rag_assistant.graph.nodes.verify_grounding import MAX_RETRIES, verify_grounding_node
from rag_assistant.graph.state import GraphState

logger = logging.getLogger(__name__)

# LangGraph counts super-steps, and on this graph every node execution is one.
# The worst-case real path is:
#   classify, rewrite, decompose, retrieve, rerank, expand_context, summarize  = 7
#   then (generate, resolve_citations, verify_grounding) once per attempt,
#   which is 1 initial + MAX_RETRIES retries                                   = 3*(1+2) = 9
#   then format_response                                                       = 1
# ...which is 17 at MAX_RETRIES=2, against a limit that was hardcoded to 15.
# So the MOST degraded path - precisely when the retry loop is doing its job -
# raised GraphRecursionError instead of returning the answer flagged
# "not fully verified" as §17.2 promises (CLAUDE.md §7 item 24).
# Derived, not hardcoded, so raising MAX_RETRIES cannot silently reintroduce it.
_WORST_CASE_STEPS = 7 + 3 * (1 + MAX_RETRIES) + 1
RECURSION_LIMIT = _WORST_CASE_STEPS + 8  # headroom for future nodes

_graph = None
_checkpointer = None
_checkpointer_cm = None  # keeps the context manager itself alive - see note below


def _with_progress(name: str, fn):
    """Reports a node as it starts, so /chat can show real progress.

    Applied at REGISTRATION rather than inside each node body: eleven identical
    two-line edits would be eleven places to forget one. The wrapper is
    transparent - it adds a dict write and calls through.
    """

    def wrapped(state: GraphState) -> dict:
        progress.note(state.get("request_id"), name)
        return fn(state)

    wrapped.__name__ = f"{name}_with_progress"
    return wrapped


def _after_classify(state: GraphState) -> str:
    c = state.get("classification", {})
    # The classifier's own needs_rewrite flag is unreliable on this model
    # (§13), so a deterministic check backs it up - but only when there is
    # actually prior context to resolve against, since rewriting the very
    # first question of a conversation has nothing to resolve and would just
    # spend an LLM call.
    if c.get("needs_rewrite") or (
        state.get("conversation_context") and looks_like_followup(state.get("question", ""))
    ):
        return "rewrite"
    if c.get("needs_decomposition"):
        return "decompose"
    return "retrieve"


def _after_rewrite(state: GraphState) -> str:
    c = state.get("classification", {})
    if c.get("needs_decomposition"):
        return "decompose"
    return "retrieve"


def _after_expand(state: GraphState) -> str:
    # Called, not read as a constant, so a monkeypatched num_ctx (tests) or a
    # changed .env is reflected immediately rather than frozen at import time.
    if _total_chars(state.get("retrieved", [])) > max_evidence_chars():
        return "summarize"
    return "generate"


def _after_verify(state: GraphState) -> str:
    if state.get("verification", {}).get("retry"):
        return "generate"
    return "format_response"


def get_graph():
    """V3's full graph (CLAUDE.md §4.2):
    classify -> (rewrite?) -> (decompose?) -> retrieve (routes by
    classification: temporal/broad-scope/multi-hop/hybrid, §20.1-§20.3) ->
    rerank -> expand_context -> (summarize? for oversized evidence, §20.3) ->
    generate -> resolve_citations (deterministic, §22.1) -> verify_grounding
    (semantic, §22.2, retries back to generate up to MAX_RETRIES) ->
    format_response (heuristic, §21 Prompt 7) -> END.
    No conversation memory *inside* the graph - that lives in SQLite (§4.3)
    and is read into `conversation_context` before invoking the graph.
    """
    global _graph, _checkpointer, _checkpointer_cm
    if _graph is None:
        # A separate SQLite file from the app's own DB (CLAUDE.md §4.2/§12) -
        # this is transient *execution-state* for resuming/retrying a single
        # graph run, never the conversation log itself (that's SQLite via
        # memory/conversation_memory.py, a deliberately distinct system, §4.3).
        # Kept open for the process's lifetime (a long-lived singleton, same
        # pattern as the Qdrant client) rather than closed after one use.
        #
        # Confirmed by direct testing, a real bug caught before it shipped:
        # from_conn_string() is a @contextmanager generator - keeping only its
        # *yielded* SqliteSaver reference (via .__enter__()) is not enough. With
        # nothing referencing the generator/context-manager object itself, the
        # garbage collector finalizes it (running the code after `yield`, which
        # closes the connection) at an unpredictable later point - here, right
        # after the next test module loaded, which manifested as "Cannot operate
        # on a closed database" on the *second* call, not the first. The context
        # manager object itself must stay referenced for as long as the saver
        # is used, not just its yielded value.
        if _checkpointer is None:
            _checkpointer_cm = SqliteSaver.from_conn_string(settings.langgraph_checkpoint_path)
            _checkpointer = _checkpointer_cm.__enter__()

        builder = StateGraph(GraphState)
        # Registered through _with_progress so every node reports itself as it
        # starts, without eleven identical edits inside the node bodies.
        for name, fn in (
            ("classify", classify_node),
            ("rewrite", rewrite_node),
            ("decompose", decompose_node),
            ("retrieve", retrieve_node),
            ("rerank", rerank_node),
            ("expand_context", expand_context_node),
            ("summarize", summarize_node),
            ("generate", generate_node),
            ("resolve_citations", resolve_citations_node),
            ("verify_grounding", verify_grounding_node),
            ("format_response", format_response_node),
        ):
            builder.add_node(name, _with_progress(name, fn))

        builder.set_entry_point("classify")
        builder.add_conditional_edges(
            "classify",
            _after_classify,
            {"rewrite": "rewrite", "decompose": "decompose", "retrieve": "retrieve"},
        )
        builder.add_conditional_edges(
            "rewrite", _after_rewrite, {"decompose": "decompose", "retrieve": "retrieve"}
        )
        builder.add_edge("decompose", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "expand_context")
        builder.add_conditional_edges(
            "expand_context", _after_expand, {"summarize": "summarize", "generate": "generate"}
        )
        builder.add_edge("summarize", "generate")
        builder.add_edge("generate", "resolve_citations")
        builder.add_edge("resolve_citations", "verify_grounding")
        builder.add_conditional_edges(
            "verify_grounding",
            _after_verify,
            {"generate": "generate", "format_response": "format_response"},
        )
        builder.add_edge("format_response", END)
        _graph = builder.compile(checkpointer=_checkpointer)
    return _graph


def ask(
    question: str,
    source_id: str | None = None,
    source_ids: list[str] | None = None,
    conversation_context: str = "",
    conversation_id: str | None = None,
    request_id: str | None = None,
) -> GraphState:
    """conversation_id is accepted (and used in the checkpoint's metadata for
    traceability) but deliberately NOT used as the checkpoint thread_id.

    Confirmed by direct testing: LangGraph persists any state key not present
    in a new invoke's input dict across calls on the *same* thread_id (e.g. a
    stale `verification.retries` count from a completely unrelated prior
    question leaked into a fresh one). Reusing conversation_id as the thread
    would corrupt turn-scoped scratch state (retrieved/citations/verification/
    classification/...) across separate turns - exactly the "checkpoint is
    this run's execution state, not the conversation" boundary CLAUDE.md
    §3.2/§4.2 already draws. Each call gets its own fresh, one-shot
    thread_id instead; conversation continuity is entirely
    memory/conversation_memory.py's job (§4.3), unaffected by this.
    """
    graph = get_graph()
    thread_id = str(uuid.uuid4())
    try:
        return graph.invoke(
            {
                "question": question,
                "source_id": source_id,
                "source_ids": source_ids,
                "conversation_context": conversation_context,
                "request_id": request_id,
            },
            {
                "recursion_limit": RECURSION_LIMIT,
                "configurable": {"thread_id": thread_id, "conversation_id": conversation_id},
            },
        )
    finally:
        # Every turn creates a fresh one-shot thread (correct - see the note
        # above), and nothing ever read it again or deleted it, so
        # langgraph_checkpoints.db grew without bound storing write-only state
        # (CLAUDE.md §7 item 46). Deleted in `finally` so a failed turn is
        # cleaned up too. delete_thread() was verified present on the installed
        # langgraph-checkpoint-sqlite 3.1.1 rather than assumed (§7 item 4);
        # pruning must never take down the answer it is tidying up after.
        try:
            _checkpointer.delete_thread(thread_id)
        except Exception:
            logger.warning("could not prune checkpoint thread %s", thread_id, exc_info=True)
