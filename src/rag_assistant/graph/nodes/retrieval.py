import re

from rag_assistant.graph.state import GraphState
from rag_assistant.ingestion.embed_index import embed_query
from rag_assistant.ingestion.time_parse import extract_time_reference, parse_time_reference
from rag_assistant.storage.vector_store import search_hybrid, search_temporal

TOP_K = 20  # pre-rerank candidate count (CLAUDE.md §20.1) - reranking cuts this down
BROAD_SCOPE_TOP_K = 60  # comprehensive queries pull wider coverage (§20.3)

# Multi-hop fan-out cap (CLAUDE.md §7 item 32). Retrieval used to merge TOP_K
# candidates PER sub-query with no ceiling on sub-query count, and the
# cross-encoder is the most expensive component in a turn (23.2s for 20 pairs).
# Three sub-queries therefore tripled rerank work on top of the extra LLM calls,
# pushing a turn toward the 600s client timeout. The total candidate budget is
# now shared between sub-queries rather than multiplied by them.
MAX_SUB_QUERIES = 3

# CLAUDE.md §20.1's routing table: these intents narrow retrieval to a chunk
# type before ranking. Only applied when it actually finds something - a
# type-narrowed search that matches nothing falls through to plain hybrid.
# `fact_lookup` is §20.1's "table/content lookup" and shares the table filter;
# before this it appeared in the classifier enum and the spec but nowhere in
# code (CLAUDE.md §7 item 33).
_ELEMENT_TYPE_BY_INTENT = {
    "code_lookup": "code",
    "table_lookup": "table",
    "fact_lookup": "table",
}

# Deterministic speaker extraction for §20.1's speaker_specific route.
# Deliberately NOT a new classifier field: diarization emits exactly
# "Speaker 1"/"Speaker 2" (diarization._relabel_by_first_appearance), so a regex
# over the question is both simpler and more reliable than asking a 0.6B model
# for it - the same "route on the signal simple enough to be reliable" rule
# §20.2 already applies to timestamps.
_SPEAKER_RE = re.compile(r"\bspeaker\s*#?\s*(\d{1,2})\b", re.IGNORECASE)


def extract_speaker(question: str) -> str | None:
    match = _SPEAKER_RE.search(question or "")
    return f"Speaker {int(match.group(1))}" if match else None


def _hybrid(
    query: str,
    top_k: int,
    source_ids: list[str] | None,
    element_type: str | None = None,
    speaker: str | None = None,
) -> list[dict]:
    dense_vector, sparse_vector = embed_query(query)
    return search_hybrid(
        dense_vector,
        sparse_vector,
        top_k=top_k,
        source_ids=source_ids,
        element_type=element_type,
        speaker=speaker,
    )


def _dedup(chunks: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            out.append(c)
    return out


def _result(chunks: list[dict], strategy: str) -> dict:
    return {
        "retrieved": chunks,
        "retrieval_strategy": strategy,
        "candidate_count": len(chunks),
    }


def retrieve_node(state: GraphState) -> dict:
    """Routes by the classifier's output (CLAUDE.md §20.1's table, now
    implementable since Prompt 1 exists):
    - a parsed time_reference -> temporal lookup (§20.2), the most reliable
      signal even when the classifier's `intent` label itself is imperfect
      (small-model quality limitation, §13/§29 - the extracted time value is
      simpler for it to get right than the intent category)
    - code_lookup/table_lookup -> hybrid search narrowed to that element_type,
      falling back to an unnarrowed search when the corpus has no such chunks
    - response_depth=comprehensive -> broad-scope retrieval (§20.3, wider top_k)
    - sub_queries present (multi-hop, §20.1) -> retrieve per sub-query, merged
    - otherwise -> ordinary hybrid dense+sparse search (§32 V2)
    """
    classification = state.get("classification", {})
    # A single source_id is still accepted (older callers, the api's own
    # optional field); source_ids is the general form - "answer using these
    # three files" is one filtered query, not three merged in Python.
    source_ids = state.get("source_ids") or []
    if not source_ids and state.get("source_id"):
        source_ids = [state["source_id"]]
    source_ids = source_ids or None
    query = state.get("standalone_query") or state["question"]

    # Time reference: the classifier's field OR a regex over the question.
    # The regex is the reliable half - a 0.6B model drops the field often
    # enough that routing on it alone made temporal retrieval a coin flip
    # (§7 item 52), and the same "route on the simple signal" rule already
    # governs speaker routing below.
    time_seconds = parse_time_reference(classification.get("time_reference") or "")
    if time_seconds is None:
        time_seconds = extract_time_reference(state["question"])
    if time_seconds is not None and source_ids:
        results = search_temporal(source_ids, time_seconds, limit=TOP_K)
        if results:
            return _result(results, "temporal")
        # fall through to normal search if nothing in that window

    sub_queries = (state.get("sub_queries") or [])[:MAX_SUB_QUERIES]
    if sub_queries:
        # Share the candidate budget instead of multiplying it, so reranking
        # cost does not scale with how many sub-queries the decomposer invented.
        per_query = max(1, TOP_K // len(sub_queries))
        merged: list[dict] = []
        for sq in sub_queries:
            merged.extend(_hybrid(sq["query"], per_query, source_ids))
        return _result(_dedup(merged), "multi_hop")

    # Speaker-scoped retrieval (§20.1). Routed on an extracted "Speaker N" in
    # the question rather than on the intent label, for the same reliability
    # reason as temporal routing. Falls through when the corpus has no chunk
    # from that speaker, so asking about a speaker who was never diarized
    # degrades to ordinary search rather than reporting "not found".
    speaker = extract_speaker(state["question"])
    if speaker:
        by_speaker = _hybrid(query, TOP_K, source_ids, speaker=speaker)
        if by_speaker:
            return _result(by_speaker, f"speaker:{speaker}")

    if classification.get("response_depth") == "comprehensive":
        return _result(_hybrid(query, BROAD_SCOPE_TOP_K, source_ids), "broad_scope")

    # A type-narrowed search MERGES with the unnarrowed one rather than
    # replacing it (CLAUDE.md §7 item 58). Replacing was a latent bug with a
    # specific victim: a video whose code is only ever EXPLAINED ALOUD has no
    # element_type="code" chunk at all - its transcript is all
    # transcript_segment - so the moment the corpus also holds a PDF with a
    # fenced code block, "explain the code from the video" matched the PDF's
    # code, `typed` came back non-empty, and the video was excluded outright.
    # The empty-result fallback below never fired, because the filter did find
    # something - just not from the source the question was about.
    #
    # Typed hits go FIRST so a genuine code block still outranks narration into
    # the reranker, and the merged set is capped at TOP_K so rerank cost does
    # not grow (the same shared-budget rule as multi-hop above, §7 item 32).
    # One extra Qdrant query, and the query embedding is computed once for both.
    element_type = _ELEMENT_TYPE_BY_INTENT.get(classification.get("intent", ""))
    if element_type:
        dense_vector, sparse_vector = embed_query(query)
        typed = search_hybrid(
            dense_vector,
            sparse_vector,
            top_k=TOP_K,
            source_ids=source_ids,
            element_type=element_type,
        )
        plain = search_hybrid(dense_vector, sparse_vector, top_k=TOP_K, source_ids=source_ids)
        if typed:
            merged = _dedup(typed + plain)[:TOP_K]
            return _result(merged, f"hybrid+{element_type}")
        return _result(plain, "hybrid")

    return _result(_hybrid(query, TOP_K, source_ids), "hybrid")
