import re

from rag_assistant.graph.state import GraphState
from rag_assistant.llm.ollama_client import chat
from rag_assistant.prompts.response_formatter import build_messages

# Cheap heuristic instead of another LLM call (CLAUDE.md §9 - skip calls that
# aren't needed) - only clear, explicit format requests trigger Prompt 7.
_FORMAT_PATTERNS = {
    r"\bbullet(s)?\b": "a bulleted list",
    r"\bjust the code\b|\bonly the code\b|\bcode only\b": (
        "only the code, in a fenced code block, no prose"
    ),
    r"\bin (\d+) words\b": "a summary of about {0} words",
    r"\bin (\d+) sentences\b": "exactly {0} sentences",
    r"\bone[- ]line\b|\bshort answer\b": "a single short line",
}


def detect_requested_format(question: str) -> str | None:
    lower = question.lower()
    for pattern, template in _FORMAT_PATTERNS.items():
        match = re.search(pattern, lower)
        if match and match.groups():
            return template.format(*match.groups())
        if match:
            return template
    return None


def format_response_node(state: GraphState) -> dict:
    requested_format = detect_requested_format(state["question"])
    if not requested_format:
        return {"formatted_answer": None}
    try:
        formatted = chat(
            build_messages(state["answer"], requested_format), stream=False, label="format"
        )
        return {"formatted_answer": formatted}
    except Exception:
        return {"formatted_answer": None}
