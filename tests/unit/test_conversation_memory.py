from rag_assistant.memory import conversation_memory as cm


def test_create_and_save_turn(isolated_storage):
    conv_id = cm.create_conversation()
    assert cm.conversation_exists(conv_id)

    cm.save_turn(conv_id, "user", "What is a decorator?")
    cm.save_turn(conv_id, "assistant", "A decorator wraps a function. [1]")

    messages = cm.get_all_messages(conv_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_load_recent_context_includes_recent_turns(isolated_storage):
    conv_id = cm.create_conversation()
    cm.save_turn(conv_id, "user", "Explain embeddings.")
    cm.save_turn(conv_id, "assistant", "Embeddings are vectors. [1]")
    cm.save_turn(conv_id, "user", "What about dimensions?")

    context = cm.load_recent_context(conv_id)
    assert "Explain embeddings" in context
    assert "dimensions" in context


def test_needs_summarization_triggers_past_threshold(isolated_storage):
    conv_id = cm.create_conversation()
    for i in range(12):
        cm.save_turn(conv_id, "user" if i % 2 == 0 else "assistant", f"message {i}")

    needed, older = cm.needs_summarization(conv_id)
    assert needed is True
    assert len(older) == 12 - cm.MAX_RECENT_MESSAGES


def test_save_and_load_summary(isolated_storage):
    conv_id = cm.create_conversation()
    cm.save_turn(conv_id, "user", "old message")
    messages = cm.get_all_messages(conv_id)
    cm.save_summary(conv_id, messages[0]["message_id"], "User asked about old topics.")

    context = cm.load_recent_context(conv_id)
    assert "Summary of earlier conversation" in context
    assert "old topics" in context
