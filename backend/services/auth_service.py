# ============================================================
# backend/services/auth_service.py — Auth Business Logic
# ============================================================
import bcrypt
import jwt
from datetime import datetime, timedelta
from backend.core.config import JWT_SECRET, JWT_ALGO


def hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_pw(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def make_token(user_id: str, token_type: str = "access", hours: int = 24) -> str:
    exp = datetime.utcnow() + timedelta(hours=hours)
    return jwt.encode(
        {"sub": user_id, "type": token_type, "exp": exp},
        JWT_SECRET, algorithm=JWT_ALGO
    )
