# ============================================================
# backend/core/config.py — Environment & App Config
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

DB_URL      = os.getenv("DATABASE_URL")
JWT_SECRET  = os.getenv("JWT_SECRET", "secret")
JWT_ALGO    = "HS256"
SMTP_HOST   = os.getenv("SMTP_HOST")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SMTP_EMAIL  = os.getenv("SMTP_EMAIL")
SMTP_PASS   = os.getenv("SMTP_PASS")
APP_URL     = os.getenv("APP_URL", "http://localhost:5173")
GUEST_LIMIT = 50
