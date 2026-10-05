from fastapi import APIRouter, HTTPException

from rag_assistant.memory import conversation_memory as cm

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
def list_conversations():
    """Past conversations, newest first (CLAUDE.md §4.3). Without this the
    history was being written to SQLite correctly but was unreachable - the UI
    had no way to reopen anything once a new conversation started."""
    return cm.list_conversations()


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str):
    if not cm.conversation_exists(conversation_id):
        detail = {"code": "NOT_FOUND", "message": "conversation not found"}
        raise HTTPException(status_code=404, detail=detail)
    return {"conversation_id": conversation_id, "messages": cm.get_all_messages(conversation_id)}


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Removes one conversation and its messages and summaries (§4.3).

    The list grows with every question asked, and until now nothing could
    remove an entry - the only control was "New conversation", which added
    another one."""
    if not cm.delete_conversation(conversation_id):
        detail = {"code": "NOT_FOUND", "message": "conversation not found"}
        raise HTTPException(status_code=404, detail=detail)
    return {"deleted": conversation_id}
