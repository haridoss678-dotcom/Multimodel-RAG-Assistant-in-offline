"""Rolling summarization must not re-run on every turn.

CLAUDE.md §7 item 30. needs_summarization() only compared total message count
against SUMMARIZE_THRESHOLD, so once a conversation passed 10 messages it
returned True on EVERY subsequent turn - re-summarizing nearly identical text and
adding an unbudgeted 15-35s LLM call to each turn indefinitely.
"""

from rag_assistant.memory import conversation_memory as cm


def _fill(conversation_id: str, count: int) -> None:
    for i in range(count):
        cm.save_turn(conversation_id, "user" if i % 2 == 0 else "assistant", f"message {i}")


def test_below_the_threshold_nothing_is_summarized(isolated_storage):
    conversation_id = cm.create_conversation()
    _fill(conversation_id, cm.SUMMARIZE_THRESHOLD)

    needed, older = cm.needs_summarization(conversation_id)

    assert needed is False
    assert older == []


def test_first_time_past_the_threshold_summarizes(isolated_storage):
    conversation_id = cm.create_conversation()
    _fill(conversation_id, cm.SUMMARIZE_THRESHOLD + 2)

    needed, older = cm.needs_summarization(conversation_id)

    assert needed is True
    # Everything except the most recent window.
    assert len(older) == cm.SUMMARIZE_THRESHOLD + 2 - cm.MAX_RECENT_MESSAGES


def test_does_not_resummarize_on_the_very_next_turn(isolated_storage):
    """The actual leak: once summarized, the next turn re-ran the whole thing."""
    conversation_id = cm.create_conversation()
    _fill(conversation_id, cm.SUMMARIZE_THRESHOLD + 2)
    _needed, older = cm.needs_summarization(conversation_id)
    cm.save_summary(conversation_id, older[-1]["message_id"], "a summary")

    cm.save_turn(conversation_id, "user", "one more question")

    needed, _ = cm.needs_summarization(conversation_id)
    assert needed is False, "re-summarized after a single new message"


def test_resummarizes_once_a_full_window_of_new_messages_has_aged_out(isolated_storage):
    conversation_id = cm.create_conversation()
    _fill(conversation_id, cm.SUMMARIZE_THRESHOLD + 2)
    _needed, older = cm.needs_summarization(conversation_id)
    cm.save_summary(conversation_id, older[-1]["message_id"], "a summary")

    # Enough new turns that a whole window has aged past the summary.
    _fill(conversation_id, cm.RESUMMARIZE_AFTER_NEW_MESSAGES + cm.MAX_RECENT_MESSAGES)

    needed, older_again = cm.needs_summarization(conversation_id)
    assert needed is True
    assert len(older_again) > len(older)


def test_a_summary_for_an_unknown_message_id_does_not_block_forever(isolated_storage):
    """Defensive: if the covered message_id is not in this conversation's list
    (deleted history, hand-edited row), fall back to summarizing rather than
    silently never summarizing again."""
    conversation_id = cm.create_conversation()
    _fill(conversation_id, cm.SUMMARIZE_THRESHOLD + 2)
    cm.save_summary(conversation_id, "not-a-real-message-id", "stale summary")

    needed, _ = cm.needs_summarization(conversation_id)

    assert needed is True
