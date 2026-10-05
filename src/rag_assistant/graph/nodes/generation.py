from rag_assistant.graph.state import GraphState
from rag_assistant.llm.ollama_client import chat
from rag_assistant.prompts.generation import build_messages
from rag_assistant.storage.db import get_connection


def _attach_filenames(evidence_items: list[dict]) -> list[dict]:
    conn = get_connection()
    out = []
    for item in evidence_items:
        row = conn.execute(
            "SELECT filename FROM sources WHERE source_id = ?", (item["source_id"],)
        ).fetchone()
        out.append({**item, "filename": row["filename"] if row else item["source_id"]})
    conn.close()
    return out


def generate_node(state: GraphState) -> dict:
    """Prompt 4 (CLAUDE.md §21), one call, non-streaming here for a
    synchronous, testable graph invocation. Token-by-token streaming to the
    UI (§26) is layered on top by the API/chat route calling
    llm.ollama_client.chat(..., stream=True) directly - the graph's own
    invoke path stays simple and deterministic for tests."""
    evidence_items = _attach_filenames(state["retrieved"])
    if not evidence_items:
        return {"answer": "I couldn't find anything about that in your documents.", "citations": []}
    # standalone_query is already reference-resolved by the rewriter (§21,
    # Prompt 2) when present - conversation history doesn't need to be
    # re-injected here (CLAUDE.md's "history is not evidence" rule, §4.3).
    question = state.get("standalone_query") or state["question"]
    extra_constraint = state.get("verification", {}).get("verdict")
    # CLAUDE.md §23 - the classifier's detected language drives the output
    # register (was hardcoded to English until a real Tamil-query test caught
    # this never actually being wired up, §13).
    classification = state.get("classification", {})
    language = classification.get("language")
    # §21's "answer at {response_depth} depth" - the classifier detects this and
    # §20.3 already widens retrieval for it, but it never reached the prompt, so
    # a request for full detail still produced a one-liner.
    response_depth = classification.get("response_depth")
    messages = build_messages(
        question, evidence_items, language=language, response_depth=response_depth
    )
    if extra_constraint and extra_constraint != "grounded":
        messages[0]["content"] += (
            "\n\nNOTE: a previous answer had unsupported claims. Only state what the "
            "evidence directly supports; omit anything you're not sure is backed by it."
        )
    answer = chat(messages, stream=False, label="generate")
    return {"answer": answer}
