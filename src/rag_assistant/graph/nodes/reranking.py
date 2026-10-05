import gc

from rag_assistant.config import settings
from rag_assistant.graph.state import GraphState

TOP_K_AFTER_RERANK = 5  # the "normal" depth default, and the fallback

# How many chunks survive reranking, by requested answer depth.
#
# CLAUDE.md §7 item 45. §19's technique 12 claimed response_depth "drives both K
# AND the depth instruction", but only the PRE-rerank pool widened
# (BROAD_SCOPE_TOP_K=60) and reranking then truncated to a hardcoded 5 - the
# same number a one-line question got. So the 60-candidate widening was
# discarded at this step and "tell me everything" reached the LLM with exactly
# as much evidence as "what is X".
#
# The upper end is safe because the budget is enforced DOWNSTREAM rather than
# here: expand_context stops expanding once the projected evidence would exceed
# max_evidence_chars(), and summarize_node still catches anything past it. So a
# larger K widens coverage and degrades to less expansion per chunk, instead of
# overflowing the context window.
_TOP_K_BY_DEPTH = {"short": 3, "normal": 5, "detailed": 8, "comprehensive": 12}

# Below this many candidates, keep ALL of them regardless of depth.
#
# Measured on a real 2-page resume (CLAUDE.md §7 item 51): the whole document
# indexes to 5 chunks, retrieval returned all 5, and a "short" classification
# then truncated to 3 - silently discarding 40% of everything the document
# says. The user asked "what are the skills in that PDF" and got a partial
# list, because two fifths of the resume never reached the model. Ranking is
# only meaningful when there is something to rank away; with a handful of
# chunks the cheapest correct answer is to send them all, and the evidence
# budget (enforced downstream by budget-aware expansion) is what stops that
# from overflowing.
KEEP_ALL_BELOW = 8

# §20 policy for low-confidence ASR segments (§15/§17's low_confidence flag).
# Chosen over hard exclusion deliberately: a flagged segment is *suspect*, not
# known-wrong, and dropping it outright would silently remove the only evidence
# for a question whose answer really does sit in a noisy passage - the user
# would get "not found" with no way to tell why. Halving the rerank score means
# any clean chunk of comparable relevance wins, while a flagged chunk can still
# surface when nothing better exists - and when it does, the citation carries a
# visible warning (§22.1) rather than a silent one. §9 wants this visible.
LOW_CONFIDENCE_PENALTY = 0.5

_reranker = None


def release_reranker() -> None:
    """Drops the cross-encoder so its memory returns to the OS.

    The reranker is the single largest consumer in the pipeline - 2221MB
    measured, nearly double BGE-M3 despite being half its size on disk - and it
    is used for a few seconds per turn. CLAUDE.md §29 REQUIRES that "a component
    releases its model before the next pipeline stage loads its own", and
    nothing did; a turn accumulated ~4.4GB on a 7.77GB machine that routinely
    shows under 1GB free, which §29 itself names as the likely cause of the
    measured 107s turn.

    Not free, and measured rather than assumed: reloading costs 12.6-18.8s
    (§7 item 48) against ~6s of actual rerank compute, so this is only worth it
    if the freed 2.2GB saves more paging than the reload costs. Off by default
    until that end-to-end comparison says otherwise (RAG_RELEASE_RERANKER).
    """
    global _reranker
    if _reranker is None:
        return
    _reranker = None
    gc.collect()


def _get_reranker():
    global _reranker
    if _reranker is None:
        # Imported lazily - RAM matters (§29), only loaded during the rerank step.
        from FlagEmbedding import FlagReranker

        # Standard (PyTorch) weights, not the ONNX-int8 build CLAUDE.md's §5
        # names - getting a pre-quantized ONNX bge-reranker-v2-m3 running needs
        # extra tooling (onnxruntime conversion) beyond what a single V2 pass
        # should add. This is the same kind of scope call as the LLM/ASR model
        # size (§13) - flagged here, not silently substituted: ONNX-int8 stays
        # a documented CPU-latency upgrade to benchmark later.
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False, devices=["cpu"])
    return _reranker


def keep_count(state: GraphState, candidate_count: int | None = None) -> int:
    """How many reranked chunks to keep, from the classifier's response_depth -
    unless the candidate pool is small enough that dropping any of it loses
    real coverage (see KEEP_ALL_BELOW)."""
    depth = (state.get("classification") or {}).get("response_depth")
    by_depth = _TOP_K_BY_DEPTH.get(depth or "normal", TOP_K_AFTER_RERANK)
    if candidate_count is not None and candidate_count <= KEEP_ALL_BELOW:
        return candidate_count
    return by_depth


def rerank_node(state: GraphState) -> dict:
    """Cross-encoder reranking over the hybrid candidate set (CLAUDE.md §32
    V2 scope, §20.1's routing table)."""
    candidates = state.get("retrieved", [])
    if len(candidates) <= 1:
        return {"retrieved": candidates, "reranked_count": len(candidates)}

    reranker = _get_reranker()
    pairs = [(state["question"], c["text"]) for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):
        scores = [scores]

    scores = [
        score * LOW_CONFIDENCE_PENALTY if candidate.get("low_confidence") else score
        for candidate, score in zip(candidates, scores)
    ]
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top = [c for c, _ in reranked[: keep_count(state, len(candidates))]]
    if settings.release_reranker:
        release_reranker()
    return {"retrieved": top, "reranked_count": len(top)}
