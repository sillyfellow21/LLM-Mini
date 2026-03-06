# ============================================================
# backend/controllers/auth_controller.py — Auth Logic
# ============================================================
import secrets
from datetime import datetime, timedelta
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from backend.models.db_models import User, EmailToken
from backend.models.schemas import RegisterReq, LoginReq, TokenReq, ForgotReq, ResetReq
from backend.services.auth_service import hash_pw, check_pw, make_token
from backend.services.email_service import send_verify_email, send_reset_email


def register(req: RegisterReq, bg: BackgroundTasks, db: Session):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(400, "Email already registered")
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(400, "Username already taken")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    user = User(email=req.email, username=req.username,
                password_hash=hash_pw(req.password))
    db.add(user); db.commit(); db.refresh(user)

    token = secrets.token_urlsafe(32)
    db.add(EmailToken(user_id=user.id, token=token, token_type="verify",
                      expires_at=datetime.utcnow() + timedelta(hours=24)))
    db.commit()
    bg.add_task(send_verify_email, user.email, token)
    return {"message": "Registered! Please check your email to verify your account."}


def verify_email(req: TokenReq, db: Session):
    t = db.query(EmailToken).filter(
        EmailToken.token == req.token,
        EmailToken.token_type == "verify",
        EmailToken.used == False
    ).first()
    if not t or t.expires_at < datetime.utcnow():
        raise HTTPException(400, "Invalid or expired token")
    t.used = True
    t.user.is_verified = True
    db.commit()
    return {"message": "Email verified! You can now login."}


def login(req: LoginReq, db: Session):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not check_pw(req.password, user.password_hash):
        raise HTTPException(400, "Invalid email or password")
    if not user.is_verified:
        raise HTTPException(400, "Please verify your email first")
    return {
        "access_token":  make_token(user.id, "access",  hours=24),
        "refresh_token": make_token(user.id, "refresh", hours=24 * 30),
        "user": {"id": user.id, "email": user.email, "username": user.username},
    }


def forgot_password(req: ForgotReq, bg: BackgroundTasks, db: Session):
    user = db.query(User).filter(User.email == req.email).first()
    if user:
        token = secrets.token_urlsafe(32)
        db.add(EmailToken(user_id=user.id, token=token, token_type="reset",
                          expires_at=datetime.utcnow() + timedelta(hours=1)))
        db.commit()
        bg.add_task(send_reset_email, user.email, token)
    return {"message": "If that email exists, a reset link has been sent."}


def reset_password(req: ResetReq, db: Session):
    t = db.query(EmailToken).filter(
        EmailToken.token == req.token,
        EmailToken.token_type == "reset",
        EmailToken.used == False
    ).first()
    if not t or t.expires_at < datetime.utcnow():
        raise HTTPException(400, "Invalid or expired token")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    t.used = True
    t.user.password_hash = hash_pw(req.password)
    db.commit()
    return {"message": "Password reset successfully. You can now login."}
