from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from database import (
    ChatMessageSchema,
    get_user_chat_history,
    get_user_conversations,
    save_chat_message,
)
from deps import get_current_user_email
from eilaaj.pipeline import query_rag
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/")
def send_message(
    payload: ChatMessageSchema,
    user_email: str = Depends(get_current_user_email),
):
    history = get_user_chat_history(user_email, payload.conversation_id)
    save_chat_message(user_email, "user", payload.message, payload.conversation_id)

    reply = query_rag(payload.message, history=history)

    save_chat_message(user_email, "bot", reply, payload.conversation_id)
    return {"reply": reply}


@router.get("/history")
def chat_history(
    conversation_id: str = "default",
    user_email: str = Depends(get_current_user_email),
):
    return {"history": get_user_chat_history(user_email, conversation_id)}


@router.get("/conversations")
def list_conversations(user_email: str = Depends(get_current_user_email)):
    return {"conversations": get_user_conversations(user_email)}