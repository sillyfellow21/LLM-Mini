# ============================================================
# backend/controllers/chat_controller.py — Chat Logic
# ============================================================
import json
from datetime import datetime
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.models.db_models import Chat, Message
from backend.models.schemas import NewChatReq, RenameChatReq
from backend.models.db_models import User
from backend.core.config import GUEST_LIMIT
from backend.model_loader import get_model, get_tokenizer
from backend.generator import detect_task, generate_stream
from backend.wikipedia_search import build_prompt

import sys, os
sys.path.insert(0, os.path.expanduser("~/LLM-Mini"))
import config as cfg
PROMPTS = cfg.INFERENCE_PROMPTS

# In-memory guest counter
guest_counts: dict = {}


def fallback_response(task, message):
    responses = {
        "farmer": (
            "Here are three practical farming tips:\n\n"
            "1. Test your soil before applying fertilizer.\n"
            "2. Water deeply and early in the day to reduce evaporation.\n"
            "3. Inspect leaves regularly for pests or disease and remove affected growth early."
        ),
        "story": (
            "The old garden gate opened only when someone arrived with a question. "
            "One morning, a child asked where courage lived. The garden answered with "
            "a path through the tall grass, and every step made the path brighter."
        ),
        "poetry": (
            "Morning gathers softly\n"
            "Across the waiting field,\n"
            "A seed holds a promise\n"
            "The patient earth will yield."
        ),
    }
    return responses.get(task, f"I can help with that: {message}")


def get_chats(db: Session, user: User):
    chats = db.query(Chat).filter(Chat.user_id == user.id)\
               .order_by(Chat.updated_at.desc()).all()
    return [{"id": c.id, "title": c.title, "updated_at": str(c.updated_at)}
            for c in chats]


def create_chat(req: NewChatReq, db: Session, user: User):
    chat = Chat(user_id=user.id, title=req.title)
    db.add(chat); db.commit(); db.refresh(chat)
    return {"id": chat.id, "title": chat.title}


def get_messages(chat_id: str, db: Session, user: User):
    chat = db.query(Chat).filter(Chat.id == chat_id,
                                 Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    return [{"id": m.id, "role": m.role, "content": m.content,
             "task": m.task, "used_wiki": m.used_wiki}
            for m in chat.messages]


def rename_chat(chat_id: str, req: RenameChatReq, db: Session, user: User):
    chat = db.query(Chat).filter(Chat.id == chat_id,
                                 Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    chat.title = req.title
    db.commit()
    return {"message": "Renamed"}


def delete_chat(chat_id: str, db: Session, user: User):
    chat = db.query(Chat).filter(Chat.id == chat_id,
                                 Chat.user_id == user.id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    db.delete(chat); db.commit()
    return {"message": "Deleted"}


def save_to_db(db, chat_id, user, message, full_response, detected_task, used_wiki):
    """Save user + assistant messages to database."""
    try:
        chat = db.query(Chat).filter(Chat.id == chat_id,
                                     Chat.user_id == user.id).first()
        if chat:
            db.add(Message(chat_id=chat_id, role="user",
                           content=message, task=detected_task, used_wiki=False))
            db.add(Message(chat_id=chat_id, role="assistant",
                           content=full_response, task=detected_task,
                           used_wiki=used_wiki))
            if chat.title == "New Chat":
                chat.title = message[:40] + ("..." if len(message) > 40 else "")
            chat.updated_at = datetime.utcnow()
            db.commit()
    except Exception as e:
        print(f"[DB] Save error: {e}")


def stream_chat(request: Request, message: str, task: str,
                chat_id: str, db: Session, user):

    # Guest rate limiting
    if not user:
        ip    = request.client.host
        count = guest_counts.get(ip, 0)
        if count >= GUEST_LIMIT:
            async def limit_event():
                yield "data: " + json.dumps({
                    "type": "error",
                    "message": "Guest limit reached. Please register for unlimited access."
                }) + "\n\n"
            return StreamingResponse(limit_event(), media_type="text/event-stream")
        guest_counts[ip] = count + 1

    detected_task              = task or detect_task(message)
    prompt, used_wiki, is_direct = build_prompt(detected_task, message, PROMPTS)

    # ── QA: return Wikipedia answer directly, skip model ──
    if is_direct:
        async def wiki_stream():
            yield "data: " + json.dumps({
                "type": "task", "task": detected_task, "used_wiki": True
            }) + "\n\n"

            # Stream word by word for natural feel
            words = prompt.split(" ")
            full_response = prompt
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield "data: " + json.dumps({"type": "token", "token": chunk}) + "\n\n"

            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

            if user and chat_id:
                save_to_db(db, chat_id, user, message,
                           full_response, detected_task, True)

        return StreamingResponse(wiki_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # ── Farmer/Story/Poetry: model generates ──────────────
    try:
        tokenizer = get_tokenizer()
        model     = get_model(detected_task)
    except Exception:
        async def fallback_stream():
            response = fallback_response(detected_task, message)
            yield "data: " + json.dumps({
                "type": "task", "task": detected_task, "used_wiki": used_wiki
            }) + "\n\n"
            for word in response.split(" "):
                yield "data: " + json.dumps({
                    "type": "token", "token": word + " "
                }) + "\n\n"
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        return StreamingResponse(fallback_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    async def model_stream():
        yield "data: " + json.dumps({
            "type": "task", "task": detected_task, "used_wiki": used_wiki
        }) + "\n\n"

        full_response = ""
        try:
            for token in generate_stream(prompt, model, tokenizer, detected_task):
                full_response += token
                yield "data: " + json.dumps({"type": "token", "token": token}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"
            return

        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        if user and chat_id:
            save_to_db(db, chat_id, user, message,
                       full_response, detected_task, used_wiki)

    return StreamingResponse(model_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})