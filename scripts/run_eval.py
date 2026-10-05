"""Runs data/eval/golden_qa.jsonl through the real graph (CLAUDE.md §21/§31).
Simple hit/miss + citation correctness per item - not yet the full Recall@K/
MRR/WER/CER metric suite §21 eventually wants, but real, not a stub: every
question actually goes through ingestion + the live graph, no mocking.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from rag_assistant.graph.build_graph import ask  # noqa: E402
from rag_assistant.ingestion.pipeline import ingest_file  # noqa: E402
from rag_assistant.memory import conversation_memory as cm  # noqa: E402

GOLDEN_PATH = Path(__file__).parent.parent / "data" / "eval" / "golden_qa.jsonl"


def _load_fixtures():
    from conftest import _make_pdf

    pdf_bytes = _make_pdf(
        [
            "Introduction to Python\nThis document introduces basic Python concepts.",
            "Decorators\nA decorator is a function that wraps another function to "
            "extend its behavior without modifying it directly.",
        ]
    )
    pdf_source_id = ingest_file(pdf_bytes, "python_notes.pdf")

    audio_path = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_speech.wav"
    audio_source_id = None
    if audio_path.exists():
        audio_source_id = ingest_file(audio_path.read_bytes(), "lecture.wav")

    return {"pdf": pdf_source_id, "audio": audio_source_id}


def _check(item: dict, result: dict) -> tuple[bool, str]:
    answer = (result.get("answer") or "").lower()

    if "expected_answer_contains" in item:
        needle = item["expected_answer_contains"].lower()
        if needle not in answer:
            return False, f"expected answer to contain {needle!r}"

    if item.get("expected_no_citations") and result.get("citations"):
        return False, f"expected no real citations, got {result['citations']}"

    if "expected_grounded" in item and result.get("grounded") != item["expected_grounded"]:
        return False, f"expected grounded={item['expected_grounded']}, got {result.get('grounded')}"

    if "expected_page" in item:
        pages = [c.get("page") for c in result.get("citations", [])]
        if item["expected_page"] not in pages:
            return False, f"expected a citation on page {item['expected_page']}, got {pages}"

    return True, "ok"


def main() -> int:
    sources = _load_fixtures()
    items = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]

    passed = 0
    for item in items:
        source_id = sources.get(item.get("source"))

        if item.get("requires_prior_turn"):
            conv_id = cm.create_conversation()
            cm.save_turn(conv_id, "user", item["requires_prior_turn"])
            prior = ask(item["requires_prior_turn"], source_id=source_id)
            cm.save_turn(conv_id, "assistant", prior.get("answer", ""))
            context = cm.load_recent_context(conv_id)
        else:
            context = ""

        result = ask(item["query"], source_id=source_id, conversation_context=context)
        ok, reason = _check(item, result)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {item['id']} ({item['category']}): {reason}")
        passed += ok

    print(f"\n{passed}/{len(items)} passed")
    return 0 if passed == len(items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
