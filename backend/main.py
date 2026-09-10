# ============================================================
# main.py — FastAPI App Entry Point
# ============================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.database import Base, engine
from backend.routes import auth, user, chat

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="LLM-Mini API")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True)

# Register routers
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(user.router, prefix="/user", tags=["User"])
app.include_router(chat.router, tags=["Chat"])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup():
    print("[API] Starting LLM-Mini...")
    from backend.model_loader import preload_all
    preload_all()
    print("[API] Ready!")
