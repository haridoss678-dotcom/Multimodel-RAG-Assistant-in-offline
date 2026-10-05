from rag_assistant.graph.state import GraphState
from rag_assistant.llm.ollama_client import chat_structured
from rag_assistant.prompts.generation import build_evidence_block
from rag_assistant.prompts.grounding_verifier import build_messages, schema

MAX_RETRIES = 1

# Measured on the real resume (CLAUDE.md §7 item 54): a turn that hit the
# semantic retry path ran generate, verify, generate, verify, generate, verify
# - 381s - and STILL ended grounded=False. The retries did not fix it; they
# tripled the cost of not fixing it. One retry is kept because it occasionally
# does help; a second was measured to buy nothing.


def _derive_verdict(claims: list[dict]) -> str:
    """The model's own `overall_verdict` field was found unreliable by direct
    testing: qwen3:0.6b correctly judges individual claims (supported=true)
    but then contradicts itself in the aggregate field (overall_verdict=
    "unsupported" for the same claim). Deterministic aggregation over the
    per-claim judgments is more trustworthy than the model's own summary -
    same "don't trust the model where code can verify" principle as
    citation resolution itself (CLAUDE.md §22.3)."""
    if not claims:
        return "unsupported"
    supported_count = sum(1 for c in claims if c.get("supported"))
    if supported_count == len(claims):
        return "grounded"
    if supported_count == 0:
        return "unsupported"
    return "partially_grounded"


def verify_grounding_node(state: GraphState) -> dict:
    """Prompt 6 (CLAUDE.md §22.2) - the semantic layer, run after the
    deterministic citation check (§22.1) already ran in resolve_citations.
    On failure, signals a retry (routed by build_graph.py's conditional
    edge back to generate_node) up to MAX_RETRIES, then gives up and flags
    the answer rather than silently presenting it as trustworthy."""
    answer = state.get("answer", "")
    evidence_block = build_evidence_block(state.get("retrieved", []))
    retries = state.get("verification", {}).get("retries", 0)

    # An answer with no [n] markers reaches here only when derivation could not
    # supply any either (resolve_citations runs first and attaches markers when
    # the evidence supports them, §7 item 50). So by this point it means one of
    # two things, and the first version of this short-circuit conflated them.
    #
    #  1. NOTHING WAS RETRIEVED. generate_node returns its canned "I couldn't
    #     find anything" line. That is the CORRECT outcome (§3) - there is
    #     nothing to verify and regenerating cannot conjure evidence that is
    #     not in the corpus.
    #
    #  2. EVIDENCE WAS RETRIEVED but no sentence matched it well enough to earn
    #     a derived marker. That is a genuinely unsupported answer.
    #
    # NEITHER is worth regenerating. Measured on a real resume PDF: the retry
    # loop spent three generate calls producing three DIFFERENT uncited answers
    # (3m16s and 2m09s for one short question) and still ended at
    # grounded=False. Retrying a model that will not adopt the notation buys
    # the same failure three times - so it is not retried any more.
    citation_check = state.get("citation_check") or {}
    if not citation_check.get("markers_found", True):
        return {
            "verification": {"verdict": "unsupported", "retries": retries, "retry": False},
            "grounded": False,
        }

    # THE SEMANTIC LAYER RUNS ONLY WHEN THE DETERMINISTIC ONE FOUND SOMETHING
    # TO DOUBT (CLAUDE.md §7 item 54).
    #
    # A clean deterministic result means: the model cited, every marker indexes
    # real evidence, none needed re-pointing, and no sentence failed
    # attribution. At that point a second opinion costs 30-60s of every turn
    # and comes from a 0.6B model this project has already measured as
    # unreliable at exactly this job - §22.2 records it passing a mis-numbered
    # citation as "grounded", which is the one failure it existed to catch.
    # Paying a minute for an opinion that demonstrably missed the case it was
    # built for is the unnecessary call §9 says to skip.
    #
    # It still runs whenever there IS doubt - a re-pointed marker, an
    # out-of-range one, or a sentence that could not be attributed - which is
    # precisely where a semantic check can tell us something overlap cannot.
    #
    # What this does NOT weaken: §28's prompt-injection contract rests on the
    # DETERMINISTIC check (an injected answer has zero real markers, so
    # grounded=False before this node is ever reached), not on this layer.
    # NOTE what is deliberately NOT in this condition: `repointed`.
    # A re-pointed marker is a problem code ALREADY SOLVED - the citation now
    # points at the item that actually supports the sentence, which is more
    # trustworthy than what the model wrote, not less. Treating a completed
    # correction as unresolved doubt sent every bulleted answer (where
    # re-pointing is routine) down the verify-and-retry path, measured at
    # 273-300s per turn (§7 item 55). Out-of-range markers and unattributable
    # sentences are genuine unresolved doubt; a re-point is not.
    clean = (
        citation_check.get("markers_found")
        and not citation_check.get("out_of_range")
        and not citation_check.get("unmatched_sentences")
    )
    if clean:
        derived = citation_check.get("derived") or 0
        return {
            "verification": {
                "verdict": "grounded",
                "retries": retries,
                "retry": False,
                "verified_by": "deterministic-attribution" if derived else "deterministic-citation",
            },
            "grounded": state.get("grounded", False),
        }

    try:
        raw = chat_structured(build_messages(answer, evidence_block), schema(), label="verify")
        verdict = _derive_verdict(raw.get("claims", []))
    except Exception:
        verdict = "unsupported"

    if verdict != "grounded" and retries < MAX_RETRIES:
        return {"verification": {"verdict": verdict, "retries": retries + 1, "retry": True}}

    return {
        "verification": {"verdict": verdict, "retries": retries, "retry": False},
        "grounded": state.get("grounded", False) and verdict == "grounded",
    }
