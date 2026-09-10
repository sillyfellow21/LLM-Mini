# ============================================================
# backend/core/database.py — Database Engine & Session
# ============================================================
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.core.config import DB_URL

engine   = create_engine(DB_URL)
Session_ = sessionmaker(bind=engine)
Base     = declarative_base()

def get_db():
    db = Session_()
    try:
        yield db
    finally:
        db.close()
