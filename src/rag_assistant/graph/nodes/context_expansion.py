from rag_assistant.graph.nodes.summarize import max_evidence_chars
from rag_assistant.graph.state import GraphState
from rag_assistant.ingestion.chunking import CHUNK_OVERLAP
from rag_assistant.storage.vector_store import get_by_parent_id


def _merge_siblings(siblings: list[dict]) -> str:
    """Rebuilds a parent group in document order, without the overlap.

    Two things this has to get right:

    1. ORDER. Sorting used to be (start_time or 0, chunk_id). Every chunk on a
       PDF page has start_time None -> 0, so the real tiebreak was chunk_id - a
       uuid4 - and a reassembled page came back shuffled. chunk_index is the
       document-order key (CLAUDE.md §7 item 27).

    2. OVERLAP. _split_prose emits CHUNK_OVERLAP=150 characters of deliberate
       repetition between adjacent prose chunks, so a naively joined page
       contained every boundary twice - wasted context budget and a duplicated
       sentence for the model to trip over. The real overlap is measured rather
       than assumed to be exactly 150, because code blocks are emitted whole and
       do not overlap their neighbours at all.
    """
    ordered = sorted(siblings, key=lambda s: (s.get("start_time") or 0, s.get("chunk_index") or 0))
    merged = ""
    for sibling in ordered:
        text = sibling.get("text", "")
        if not merged:
            merged = text
            continue
        overlap = 0
        for size in range(min(CHUNK_OVERLAP, len(merged), len(text)), 0, -1):
            if merged[-size:] == text[:size]:
                overlap = size
                break
        merged += ("\n" + text) if overlap == 0 else text[overlap:]
    return merged


def expand_context_node(state: GraphState) -> dict:
    """Parent-page/time-window expansion (CLAUDE.md §32 V2 scope call #2):
    each retrieved chunk's *text* is replaced with its full parent group so
    the LLM sees surrounding context, but citation fields (page/start_time/
    source_id) stay pinned to the originally-matched chunk, not the parent -
    citations must point to what was actually relevant, not the whole page.

    Expansion is BUDGET-AWARE, and that is the point of it (CLAUDE.md §7
    item 28). Expanding all five reranked hits routinely pushed evidence past
    the summarizer's threshold, and the summarizer collapses a whole batch onto
    its FIRST chunk's page/timestamp - so the feature that creates citation
    precision was the same feature that triggered the step which destroys it.
    Hits are expanded in rerank order (best first) and expansion stops once the
    projected total would exceed the budget; the remainder keep their own
    matched text. Partial expansion with exact citations beats full expansion
    with collapsed ones.
    """
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {"retrieved": retrieved}

    budget = max_evidence_chars()
    expanded: list[dict] = []
    used = sum(len(c.get("text", "")) for c in retrieved)

    for chunk in retrieved:
        own_text = chunk.get("text", "")
        parent_id = chunk.get("parent_id")
        siblings = get_by_parent_id(parent_id) if parent_id else []
        if not siblings:
            expanded.append(chunk)
            continue

        merged_text = _merge_siblings(siblings)
        # `used` currently counts this chunk's own text; swapping in the parent
        # costs the difference.
        projected = used + len(merged_text) - len(own_text)
        if projected > budget:
            expanded.append(chunk)  # keep the precise match rather than overflow
            continue
        used = projected
        expanded.append({**chunk, "text": merged_text})

    return {"retrieved": expanded}
