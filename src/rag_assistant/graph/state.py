from typing import TypedDict


class GraphState(TypedDict, total=False):
    """The full V3 graph state (CLAUDE.md §4.2): classify -> (rewrite/
    decompose) -> route -> retrieve (hybrid/temporal/broad-scope) -> rerank
    -> expand -> generate -> verify -> resolve citations -> format.
    Conversation history lives in SQLite (§4.3), not here - this is only
    the current turn's working state."""

    question: str
    source_id: str | None
    # Several sources at once. source_id stays for single-source callers;
    # retrieval prefers this list when present.
    source_ids: list[str] | None
    conversation_id: str | None
    conversation_context: str
    # Client-supplied id for THIS turn, used only to report per-node progress
    # (api/progress.py). Ephemeral UI state - not history, not checkpoint state.
    request_id: str | None

    classification: dict  # Classification.model_dump()
    standalone_query: str
    sub_queries: list[dict]

    retrieved: list[dict]
    answer: str
    citations: list[dict]
    grounded: bool
    verification: dict
    formatted_answer: str | None

    # Observability (CLAUDE.md §30) - recorded as the graph runs so a bad
    # answer can be attributed to a stage (bad retrieval vs. bad generation vs.
    # mis-citation) instead of defaulting to "the LLM was wrong".
    retrieval_strategy: str
    candidate_count: int
    reranked_count: int
    citation_check: dict
