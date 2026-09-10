# ============================================================
# backend/routes/user.py — User Endpoints
# ============================================================
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.middleware.auth import get_current_user
from backend.models.db_models import User
from backend.models.schemas import UpdateUsernameReq, ChangePasswordReq, ChangeEmailReq
from backend.controllers import user_controller as ctrl

router = APIRouter()


@router.get("/profile")
async def get_profile(user: User = Depends(get_current_user)):
    return ctrl.get_profile(user)


@router.put("/username")
async def update_username(req: UpdateUsernameReq, db: Session = Depends(get_db),
                          user: User = Depends(get_current_user)):
    return ctrl.update_username(req, db, user)


@router.put("/password")
async def change_password(req: ChangePasswordReq, db: Session = Depends(get_db),
                          user: User = Depends(get_current_user)):
    return ctrl.change_password(req, db, user)


@router.put("/email")
async def change_email(req: ChangeEmailReq, bg: BackgroundTasks,
                       db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    return ctrl.change_email(req, bg, db, user)


@router.delete("/account")
async def delete_account(db: Session = Depends(get_db),
                         user: User = Depends(get_current_user)):
    return ctrl.delete_account(db, user)
