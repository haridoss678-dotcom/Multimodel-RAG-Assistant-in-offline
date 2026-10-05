import json
import uuid
from datetime import UTC, datetime

from rag_assistant.storage.db import get_connection

MAX_RECENT_MESSAGES = 6
SUMMARIZE_THRESHOLD = 10  # message count that triggers summarizing older turns

# New messages that must age past the recent window before the rolling summary is
# rebuilt. Without this, needs_summarization() returned True on EVERY turn once a
# conversation passed SUMMARIZE_THRESHOLD, so each turn from message 11 onward
# paid an extra 15-35s LLM call forever - a cost absent from the §23.1 latency
# profile - and appended another conversation_summaries row each time
# (CLAUDE.md §7 item 30). 6 matches MAX_RECENT_MESSAGES: the summary is rebuilt
# once a whole window's worth of turns has aged out of it.
RESUMMARIZE_AFTER_NEW_MESSAGES = 6

# How long a conversation label may be before it stops being a label.
MAX_TITLE_CHARS = 60


def heading_from_question(question: str) -> str:
    """A short, recognisable label from the question a conversation opened with.

    Used only when there is no document to name - a scoped conversation is
    labelled by its source instead (api/routes/chat.py). CLAUDE.md §7 item 43
    records a real user objecting twice to question-labels: as the *only*
    scheme they read as the app suggesting questions back. As the fallback for
    an unscoped conversation there is nothing else honest to show.
    """
    heading = " ".join(question.split())
    if not heading:
        return "All sources"
    if len(heading) > MAX_TITLE_CHARS:
        heading = heading[:MAX_TITLE_CHARS].rsplit(" ", 1)[0] + "…"
    return heading[0].upper() + heading[1:]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_conversation(title: str | None = None) -> str:
    """`title` is what the conversation list shows instead of a timestamp.

    It is set once, at creation, from what the turn is actually about - the
    scoped document's filename where there is one, otherwise a heading taken
    from the opening question. A date tells a user nothing about which
    conversation they are looking for; the document name does.
    """
    conversation_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO conversations (conversation_id, title, created_at, last_active_at) "
        "VALUES (?, ?, ?, ?)",
        (conversation_id, title, _now(), _now()),
    )
    conn.commit()
    conn.close()
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,)
    ).fetchone()
    conn.close()
    return row is not None


def save_turn(
    conversation_id: str,
    role: str,
    content: str,
    cited_sources: list[dict] | None = None,
    language: str | None = None,
    grounded: bool | None = None,
) -> None:
    """Appends one message (CLAUDE.md §4.3) - this is the actual conversation
    log, distinct from the LangGraph checkpoint's transient execution state.

    `grounded` is stored so the caution shown on a live answer survives a
    reload. None means "not applicable / unknown" (user turns, and rows written
    before the column existed) and must not be read as "verified"."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (message_id, conversation_id, role, content, "
        "cited_sources_json, language, grounded, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(uuid.uuid4()),
            conversation_id,
            role,
            content,
            json.dumps(cited_sources) if cited_sources else None,
            language,
            None if grounded is None else int(grounded),
            _now(),
        ),
    )
    conn.execute(
        "UPDATE conversations SET last_active_at = ? WHERE conversation_id = ?",
        (_now(), conversation_id),
    )
    conn.commit()
    conn.close()


def list_conversations(limit: int = 50) -> list[dict]:
    """Most recently active first, each with the question that started it so a
    UI can label it with something recognisable rather than a raw UUID."""
    conn = get_connection()
    # One-time backfill. Conversations that existed before the column did have
    # no title, and would sit among the labelled ones showing a bare date -
    # exactly the thing the label replaced. No scope was recorded for them, so
    # the opening question is all there is to go on. Normalising happens in
    # Python rather than SQL so these rows get the same trimming and casing as
    # a freshly-titled one; it deliberately does NOT run over rows that already
    # have a title, since those are filenames and must not be rewritten.
    untitled = conn.execute(
        "SELECT c.conversation_id, "
        "(SELECT m.content FROM messages m WHERE m.conversation_id = c.conversation_id "
        "  AND m.role = 'user' ORDER BY m.created_at ASC LIMIT 1) AS first_question "
        "FROM conversations c WHERE c.title IS NULL"
    ).fetchall()
    for row in untitled:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE conversation_id = ?",
            (heading_from_question(row["first_question"] or ""), row["conversation_id"]),
        )
    if untitled:
        conn.commit()
    rows = conn.execute(
        "SELECT c.conversation_id, c.title, c.created_at, c.last_active_at, "
        "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) "
        "  AS message_count, "
        "(SELECT m.content FROM messages m WHERE m.conversation_id = c.conversation_id "
        "  AND m.role = 'user' ORDER BY m.created_at ASC LIMIT 1) AS first_question "
        "FROM conversations c ORDER BY c.last_active_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_messages(conversation_id: str) -> list[dict]:
    """cited_sources_json AND grounded are included so a reopened conversation
    shows each answer exactly as it was shown live - citations intact and, for
    an unverified answer, still visibly unverified."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT message_id, role, content, cited_sources_json, language, grounded, created_at "
        "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_summary(conversation_id: str) -> str | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT summary_text FROM conversation_summaries WHERE conversation_id = ? "
        "ORDER BY rowid DESC LIMIT 1",
        (conversation_id,),
    ).fetchone()
    conn.close()
    return row["summary_text"] if row else None


def load_recent_context(conversation_id: str) -> str:
    """What Prompts 1/2/3 see for reference resolution (CLAUDE.md §4.3):
    the last few raw turns, plus a rolling summary of anything older, so this
    never grows unbounded into the token budget (§7's TPM lesson still
    applies conceptually even though the budget constraint is now RAM/
    latency, §13)."""
    messages = get_all_messages(conversation_id)
    recent = messages[-MAX_RECENT_MESSAGES:]
    recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    summary = get_latest_summary(conversation_id)
    if summary:
        return f"[Summary of earlier conversation]: {summary}\n\n{recent_text}"
    return recent_text


def save_summary(conversation_id: str, up_to_message_id: str, summary_text: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO conversation_summaries "
        "(conversation_id, up_to_message_id, summary_text) VALUES (?, ?, ?)",
        (conversation_id, up_to_message_id, summary_text),
    )
    conn.commit()
    conn.close()


def latest_summary_up_to(conversation_id: str) -> str | None:
    """The message_id the newest rolling summary already covers."""
    conn = get_connection()
    row = conn.execute(
        "SELECT up_to_message_id FROM conversation_summaries WHERE conversation_id = ? "
        "ORDER BY rowid DESC LIMIT 1",
        (conversation_id,),
    ).fetchone()
    conn.close()
    return row["up_to_message_id"] if row else None


def needs_summarization(conversation_id: str) -> tuple[bool, list[dict]]:
    """Returns whether older messages should be summarized now, and which
    ones (everything except the most recent MAX_RECENT_MESSAGES).

    Gated on how much is genuinely NEW since the last summary. The first version
    only checked total length, so it returned True on every turn once past
    SUMMARIZE_THRESHOLD and re-summarized nearly the same text each time - an
    extra LLM call per turn, permanently, for a conversation that had already
    been summarized (CLAUDE.md §7 item 30).
    """
    messages = get_all_messages(conversation_id)
    if len(messages) <= SUMMARIZE_THRESHOLD:
        return False, []

    older = messages[:-MAX_RECENT_MESSAGES]
    if not older:
        return False, []

    covered_up_to = latest_summary_up_to(conversation_id)
    if covered_up_to is not None:
        ids = [m["message_id"] for m in messages]
        if covered_up_to in ids:
            already_covered = ids.index(covered_up_to) + 1
            if len(older) - already_covered < RESUMMARIZE_AFTER_NEW_MESSAGES:
                return False, []
    return True, older


def delete_conversation(conversation_id: str) -> bool:
    """Removes a conversation and everything derived from it.

    All three tables, one operation: the messages, any rolling summaries, and
    the conversation row itself. A partial delete would leave orphaned
    messages that `list_conversations` no longer shows but that still count
    toward the summarization threshold - the same "partial deletion is a
    data-consistency bug" rule §19.3 applies to sources.

    Returns False when the conversation does not exist, so the route can 404
    rather than silently reporting success.
    """
    conn = get_connection()
    exists = conn.execute(
        "SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,)
    ).fetchone()
    if exists is None:
        conn.close()
        return False
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    return True
