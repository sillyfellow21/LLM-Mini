# ============================================================
# backend/middleware/auth.py — JWT Authentication Middleware
# ============================================================
import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.core.config import JWT_SECRET, JWT_ALGO
from backend.core.database import get_db
from backend.models.db_models import User


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    # Check header first, then fall back to query param
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        token = request.query_params.get("token", "")
    if not token:
        raise HTTPException(401, "Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        user    = db.query(User).filter(User.id == payload["sub"]).first()
        if not user:
            raise HTTPException(401, "User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except Exception:
        raise HTTPException(401, "Invalid token")


def get_optional_user(request: Request, db: Session = Depends(get_db)):
    try:
        return get_current_user(request, db)
    except Exception:
        return None
