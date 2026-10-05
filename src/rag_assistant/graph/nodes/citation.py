import re

from rag_assistant.graph.state import GraphState
from rag_assistant.storage.db import get_connection

_MARKER_RE = re.compile(r"\[(\d+)\]")
# Unicode-aware "content word": letters/digits, 3+ chars. Works for Tamil as
# well as English, which matters because this runs on every answer (§23).
_WORD_RE = re.compile(r"[^\W_]{3,}", re.UNICODE)

# A re-point only happens when some *other* evidence item both accounts for the
# claim reasonably well on its own (MIN_COVERAGE) and beats the cited item by a
# clear margin (REPOINT_MARGIN). Two similar chunks won't meet the margin, so
# the model's own choice stands - this deliberately only corrects the obvious.
MIN_COVERAGE = 0.35
REPOINT_MARGIN = 0.25
# Don't second-guess a marker attached to only a few words - too little signal
# to beat the model's own choice with.
MIN_CLAIM_WORDS = 4

# Deriving a marker the model never wrote is held to a STRICTER bar than
# re-pointing one it did write, because there is no model judgement to defer to
# - the code is the only thing asserting the link.
#
# Why this exists at all (CLAUDE.md §7 item 50): on a real resume PDF the model
# emitted ZERO [n] markers on every attempt, so the retry loop spent three
# generate calls (measured 3m16s and 2m09s for one short question) and still
# returned grounded=False with a "not verified" banner over an answer that was
# in fact drawn straight from the evidence. Retrying a model that will not
# adopt the notation just buys the same failure three times. The sentences were
# verifiably present in the evidence - so code attributes them, which is the
# same "don't trust the model where code can verify" principle §22.3 already
# applies to citation lookup, extended from checking a number to supplying one.
DERIVE_MIN_COVERAGE = 0.50
DERIVE_MIN_SENTENCE_WORDS = 4

# A terminator only ends a sentence when whitespace or end-of-line follows it.
#
# Without that lookahead the split fired inside "haridoss678@gmail.com" and the
# derived marker landed mid-address - "haridoss678@gmail. [1]com" - corrupting
# the answer text AND producing a phantom unmatched fragment that pushed the
# whole turn onto the slow verify-and-retry path (138s instead of 45s).
# Measured regression from derivation itself; version numbers like "3.11" broke
# the same way (CLAUDE.md §7 item 56).
#
# MULTILINE matters as much as the lookahead: a markdown bullet usually has no
# full stop, so without it `$` only matched at end-of-STRING and every
# unterminated line was skipped outright - a whole bulleted skills answer came
# back with zero citations and grounded=False despite being verbatim from the
# resume.
_SENTENCE_RE = re.compile(r"[^\n]+?(?:[.!?]+(?=\s|$)|$)", re.UNICODE | re.MULTILINE)


# Sentences that talk ABOUT the evidence rather than asserting anything from
# it. "No other email addresses are mentioned in the evidence" is the model
# hedging: a claim that something is ABSENT cannot, by construction, be
# supported by a quotation. Counting it as an unattributable claim sent an
# otherwise-perfect answer down the verify-and-retry path - measured at 108s
# against 35s for the identical question without the hedge
# (CLAUDE.md §7 item 56).
_META_CLAIM_RE = re.compile(
    r"^\W*(no other|there (are|is) no|no additional|none of the|"
    r"the (evidence|document|pdf|transcript|text) (does not|doesn't|contains no)|"
    r"nothing (else|further)|each \w+ is listed)",
    re.IGNORECASE,
)


def _is_lead_in(sentence: str) -> bool:
    """A preamble, heading, or statement about the evidence - not a claim.

    "The skills mentioned in the PDF are as follows:" asserts nothing that
    could be supported or contradicted by evidence, and a markdown heading even
    less so. Treating these as unattributable claims made every bulleted answer
    look partly unsupported.
    """
    stripped = sentence.strip()
    if stripped.endswith(":") or stripped.startswith("#"):
        return True
    return bool(_META_CLAIM_RE.match(stripped))


def derive_markers(answer: str, evidence_items: list[dict]) -> tuple[str, dict]:
    """Attaches [n] markers to an answer the model wrote without any.

    Each sentence is matched against the evidence by word overlap and earns a
    marker only when one item covers at least DERIVE_MIN_COVERAGE of its
    content words. Sentences that match nothing are left bare - that is the
    signal that the answer really did wander off the evidence, and it is what
    keeps this from laundering a hallucination into a citation.

    Returns (answer_with_markers, stats). The rewrite is confined to appending
    markers; no word of the model's own text is altered.
    """
    if not evidence_items:
        return answer, {"derived": 0, "unmatched": 0, "coverage": 0.0}

    evidence_words = [_content_words(item.get("text", "")) for item in evidence_items]
    out: list[str] = []
    derived = 0
    unmatched = 0
    coverages: list[float] = []
    cursor = 0

    for match in _SENTENCE_RE.finditer(answer):
        sentence = match.group(0)
        out.append(answer[cursor : match.start()])
        cursor = match.end()

        words = _content_words(sentence)
        if len(words) < DERIVE_MIN_SENTENCE_WORDS or _is_lead_in(sentence):
            # Too short, or a lead-in like "The skills are as follows:" - not a
            # claim, so neither citable nor a failure to cite. Counting these as
            # unmatched was measured forcing a whole bulleted answer down the
            # expensive verify-and-retry path over its own header line
            # (CLAUDE.md §7 item 55).
            out.append(sentence)
            continue

        best_index, best_coverage = _best_match(words, evidence_words)
        if best_index and best_coverage >= DERIVE_MIN_COVERAGE:
            out.append(f"{sentence.rstrip()} [{best_index}]")
            derived += 1
            coverages.append(best_coverage)
        else:
            out.append(sentence)
            unmatched += 1

    out.append(answer[cursor:])
    mean_coverage = sum(coverages) / len(coverages) if coverages else 0.0
    return "".join(out), {
        "derived": derived,
        "unmatched": unmatched,
        "coverage": round(mean_coverage, 3),
    }


# Function words carry no attribution signal but DO dilute coverage. Measured:
# "The candidate's email address is haridoss678@gmail.com." scored 3/7 = 0.43
# against evidence containing the address, purely because "the" and "address"
# style filler counted against it - just under the 0.50 bar, so a perfectly
# attributable sentence went uncited and pushed the turn onto the slow path
# (CLAUDE.md §7 item 56).
#
# Deliberately tiny and English-only: it is subtractive, so Tamil and Tanglish
# words are simply never in it and are unaffected. This is not a stemmer or a
# stopword corpus - it is the handful of words that appear in almost every
# English sentence regardless of meaning.
_FUNCTION_WORDS = frozenset(
    """the and are was were for with that this from have has had its
    they them their there then than also into onto out about which who whom
    whose what when where why how not but nor yet all any each our your his
    her been being does did done such same only just very more most other
    some can could would should may might must will shall""".split()
)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if w.lower() not in _FUNCTION_WORDS}


def _coverage(claim_words: set[str], chunk_words: set[str]) -> float:
    """Fraction of the claim's words that the chunk actually contains."""
    if not claim_words:
        return 0.0
    return len(claim_words & chunk_words) / len(claim_words)


def _best_match(claim_words: set[str], evidence_words: list[set[str]]) -> tuple[int, float]:
    best_index, best_coverage = 0, 0.0
    for index, words in enumerate(evidence_words, start=1):
        coverage = _coverage(claim_words, words)
        if coverage > best_coverage:
            best_index, best_coverage = index, coverage
    return best_index, best_coverage


def _claims(answer: str) -> list[tuple[set[str], int]]:
    """Pairs each marker with the words of the claim it annotates.

    A marker annotates the text *before* it, which is not the same as the
    sentence containing it: models routinely write "...behavior. [2]", putting
    the marker after the full stop. An earlier version of this function split
    on sentence boundaries and so handed that marker an empty fragment, which
    silently disabled the whole check on the most common real-world shape -
    caught by the regression test, not by reading the code.

    Consecutive markers ("[1][2]") share the preceding claim text.
    """
    claims: list[tuple[set[str], int]] = []
    cursor = 0
    last_words: set[str] = set()
    for match in _MARKER_RE.finditer(answer):
        words = _content_words(answer[cursor : match.start()])
        if words:
            last_words = words
        claims.append((last_words, int(match.group(1))))
        cursor = match.end()
    return claims


def resolve_citations_node(state: GraphState) -> dict:
    """Deterministic citation validation + server-side resolution (CLAUDE.md
    §22.1/§22.3) - the model only ever emits [n] markers; real page/source
    metadata is looked up here, never trusted from model output.

    Two independent checks run here, both in code, no LLM call:

    1. Range check: a marker pointing outside this turn's evidence is dropped
       and the answer is flagged ungrounded (§22.1, unchanged).

    2. Attribution check: the marker's *number* is verified against the text
       of the sentence carrying it. Confirmed by direct testing that this is a
       real failure mode, not a theoretical one - qwen3:0.6b produced the
       correct answer sentence verbatim from evidence item [1] but wrote "[2]"
       after it on roughly 40% of runs of the same question, which resolved to
       a real-but-wrong page. The LLM grounding verifier (§22.2) did NOT catch
       it, returning "grounded" for the mis-cited version, so a semantic model
       check alone is not sufficient protection here.

    Attribution compares word overlap *relatively* (which evidence item best
    accounts for this sentence), never as an absolute truth threshold - an
    absolute cutoff would punish legitimate paraphrases that happen to share
    few exact words. The answer text is left untouched; only the metadata the
    marker resolves to is corrected, so "[2]" in the prose still resolves to
    the source the sentence actually came from.
    """
    answer = state.get("answer", "")
    evidence_items = state.get("retrieved", [])
    evidence_words = [_content_words(item.get("text", "")) for item in evidence_items]

    # If the model wrote no markers at all but evidence WAS retrieved, supply
    # them from the evidence rather than regenerating and hoping (§7 item 50).
    # Measured: regenerating produced a different answer each time, none of
    # them cited, at ~40s per attempt.
    derivation: dict = {}
    if evidence_items and not _MARKER_RE.search(answer):
        answer, derivation = derive_markers(answer, evidence_items)

    attributions: dict[int, int] = {}  # marker as written -> evidence item index (1-based)
    out_of_range = False
    repointed = 0
    saw_marker = False

    for claim_words, marker in _claims(answer):
        saw_marker = True
        if not 1 <= marker <= len(evidence_items):
            out_of_range = True  # invented index - caught here, not trusted
            continue
        if marker in attributions:
            continue  # same marker seen earlier in the answer; first wins

        attributions[marker] = marker
        if len(claim_words) < MIN_CLAIM_WORDS:
            continue

        best_index, best_coverage = _best_match(claim_words, evidence_words)
        cited_coverage = _coverage(claim_words, evidence_words[marker - 1])
        if (
            best_index
            and best_index != marker
            and best_coverage >= MIN_COVERAGE
            and (best_coverage - cited_coverage) >= REPOINT_MARGIN
        ):
            attributions[marker] = best_index
            repointed += 1

    conn = get_connection()
    citations = []
    for marker in sorted(attributions):
        item = evidence_items[attributions[marker] - 1]
        row = conn.execute(
            "SELECT filename FROM sources WHERE source_id = ?", (item["source_id"],)
        ).fetchone()
        citation = {
            "marker": marker,
            "source_id": item["source_id"],
            "filename": row["filename"] if row else item["source_id"],
        }
        # Page citation for PDFs, timestamp citation for audio/video/YouTube
        # (CLAUDE.md §20.2) - never both, whichever the source actually has.
        if item.get("page") is not None:
            citation["page"] = item["page"]
        if item.get("start_time") is not None:
            citation["start_time"] = item["start_time"]
            citation["end_time"] = item.get("end_time")
        if item.get("speaker"):
            citation["speaker"] = item["speaker"]
        # Surfaced per-citation, not just logged: when an answer rests on a
        # segment the ASR itself was unsure about, the reader has to be able to
        # see that. The grounding checks (§22) only verify the claim matches the
        # cited text - neither asks whether that text is itself trustworthy, so
        # a hallucinated segment can otherwise be cited and presented as
        # confidently grounded (§9's accuracy-honesty rule).
        if item.get("low_confidence"):
            citation["low_confidence"] = True
        citations.append(citation)
    conn.close()

    return {
        # The answer is returned WITH the derived markers when derivation ran,
        # so what the reader sees and what the citations refer to are the same
        # text. When the model cited on its own, this is byte-identical to what
        # it produced.
        "answer": answer,
        "citations": citations,
        "grounded": saw_marker and not out_of_range,
        "citation_check": {
            "markers_found": saw_marker,
            "out_of_range": out_of_range,
            "repointed": repointed,
            # Surfaced, never hidden: a derived citation is code asserting the
            # link, not the model, and the diagnostics panel says so.
            "derived": derivation.get("derived", 0),
            "unmatched_sentences": derivation.get("unmatched", 0),
            "derived_coverage": derivation.get("coverage", 0.0),
        },
    }
