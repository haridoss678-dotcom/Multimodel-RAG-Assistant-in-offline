from rag_assistant.graph.state import GraphState
from rag_assistant.llm.ollama_client import chat_structured
from rag_assistant.prompts.decomposer import build_messages, schema


def decompose_node(state: GraphState) -> dict:
    """Prompt 3 (CLAUDE.md §21/§20.1) - only reached when the classifier set
    needs_decomposition=True. Falls back to a single-item list (the question
    itself, unsplit) if the LLM call fails twice - multi-hop retrieval then
    degrades to ordinary single-query retrieval rather than crashing."""
    query = state.get("standalone_query") or state["question"]
    messages = build_messages(query)

    for _attempt in range(2):
        try:
            raw = chat_structured(messages, schema(), label="decompose")
            sub_queries = raw["sub_queries"]
            if sub_queries:
                return {"sub_queries": sorted(sub_queries, key=lambda s: s["order"])}
        except Exception:
            continue

    return {"sub_queries": [{"order": 0, "query": query, "depends_on": []}]}
