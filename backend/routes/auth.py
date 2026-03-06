# ============================================================
# backend/routes/auth.py — Auth Endpoints
# ============================================================
from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.models.schemas import RegisterReq, LoginReq, TokenReq, ForgotReq, ResetReq
from backend.controllers import auth_controller as ctrl

router = APIRouter()


@router.post("/register")
async def register(req: RegisterReq, bg: BackgroundTasks,
                   db: Session = Depends(get_db)):
    return ctrl.register(req, bg, db)


@router.post("/verify-email")
async def verify_email(req: TokenReq, db: Session = Depends(get_db)):
    return ctrl.verify_email(req, db)


@router.post("/login")
async def login(req: LoginReq, db: Session = Depends(get_db)):
    return ctrl.login(req, db)


@router.post("/forgot-password")
async def forgot_password(req: ForgotReq, bg: BackgroundTasks,
                          db: Session = Depends(get_db)):
    return ctrl.forgot_password(req, bg, db)


@router.post("/reset-password")
async def reset_password(req: ResetReq, db: Session = Depends(get_db)):
    return ctrl.reset_password(req, db)
