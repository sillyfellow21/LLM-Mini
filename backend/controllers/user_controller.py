# ============================================================
# backend/controllers/user_controller.py — User Logic
# ============================================================
import secrets
from datetime import datetime, timedelta
from fastapi import HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from backend.models.db_models import User, EmailToken
from backend.models.schemas import UpdateUsernameReq, ChangePasswordReq, ChangeEmailReq
from backend.services.auth_service import hash_pw, check_pw
from backend.services.email_service import send_verify_email


def get_profile(user: User):
    return {
        "id":          user.id,
        "email":       user.email,
        "username":    user.username,
        "is_verified": user.is_verified,
        "created_at":  str(user.created_at),
    }


def update_username(req: UpdateUsernameReq, db: Session, user: User):
    if db.query(User).filter(User.username == req.username, User.id != user.id).first():
        raise HTTPException(400, "Username already taken")
    user.username = req.username
    db.commit()
    return {"message": "Username updated", "username": user.username}


def change_password(req: ChangePasswordReq, db: Session, user: User):
    if not check_pw(req.old_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    user.password_hash = hash_pw(req.new_password)
    db.commit()
    return {"message": "Password changed successfully"}


def change_email(req: ChangeEmailReq, bg: BackgroundTasks, db: Session, user: User):
    if not check_pw(req.password, user.password_hash):
        raise HTTPException(400, "Password is incorrect")
    if db.query(User).filter(User.email == req.new_email, User.id != user.id).first():
        raise HTTPException(400, "Email already in use")
    user.email       = req.new_email
    user.is_verified = False
    db.commit()
    token = secrets.token_urlsafe(32)
    db.add(EmailToken(user_id=user.id, token=token, token_type="verify",
                      expires_at=datetime.utcnow() + timedelta(hours=24)))
    db.commit()
    bg.add_task(send_verify_email, user.email, token)
    return {"message": "Email updated. Please verify your new email."}


def delete_account(db: Session, user: User):
    db.delete(user)
    db.commit()
    return {"message": "Account deleted"}
