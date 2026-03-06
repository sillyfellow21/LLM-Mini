# ============================================================
# backend/routes/chat.py — Chat Endpoints
# ============================================================
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.middleware.auth import get_current_user, get_optional_user
from backend.models.db_models import User
from backend.models.schemas import NewChatReq, RenameChatReq
from backend.controllers import chat_controller as ctrl

router = APIRouter()


@router.get("/chats")
async def get_chats(db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    return ctrl.get_chats(db, user)


@router.post("/chats")
async def create_chat(req: NewChatReq, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)):
    return ctrl.create_chat(req, db, user)


@router.get("/chats/{chat_id}/messages")
async def get_messages(chat_id: str, db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    return ctrl.get_messages(chat_id, db, user)


@router.put("/chats/{chat_id}")
async def rename_chat(chat_id: str, req: RenameChatReq,
                      db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)):
    return ctrl.rename_chat(chat_id, req, db, user)


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, db: Session = Depends(get_db),
                      user: User = Depends(get_current_user)):
    return ctrl.delete_chat(chat_id, db, user)


@router.get("/chat/stream")
async def chat_stream(request: Request, message: str, task: str = None,
                      chat_id: str = None, db: Session = Depends(get_db)):
    user = get_optional_user(request, db)
    return ctrl.stream_chat(request, message, task, chat_id, db, user)
