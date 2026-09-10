# ============================================================
# backend/services/email_service.py — Email Sending Logic
# ============================================================
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from backend.core.config import SMTP_HOST, SMTP_PORT, SMTP_EMAIL, SMTP_PASS, APP_URL


def send_email(to: str, subject: str, html: str):
    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_EMAIL
        msg["To"]      = to
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_EMAIL, SMTP_PASS)
            s.sendmail(SMTP_EMAIL, to, msg.as_string())
    except Exception as e:
        print(f"[Email] Failed: {e}")


def send_verify_email(email: str, token: str):
    url  = f"{APP_URL}/verify-email?token={token}"
    html = f"""
    <div style="font-family:sans-serif;max-width:500px;margin:auto;padding:32px">
        <h2 style="color:#10a37f">Verify your email</h2>
        <p>Click the button below to verify your LLM-Mini account.</p>
        <a href="{url}" style="display:inline-block;margin:16px 0;padding:12px 24px;
            background:#10a37f;color:white;text-decoration:none;border-radius:8px">
            Verify Email
        </a>
        <p style="color:#999;font-size:12px">Link expires in 24 hours.</p>
    </div>"""
    send_email(email, "Verify your LLM-Mini account", html)


def send_reset_email(email: str, token: str):
    url  = f"{APP_URL}/reset-password?token={token}"
    html = f"""
    <div style="font-family:sans-serif;max-width:500px;margin:auto;padding:32px">
        <h2 style="color:#10a37f">Reset your password</h2>
        <p>Click below to reset your password.</p>
        <a href="{url}" style="display:inline-block;margin:16px 0;padding:12px 24px;
            background:#10a37f;color:white;text-decoration:none;border-radius:8px">
            Reset Password
        </a>
        <p style="color:#999;font-size:12px">Link expires in 1 hour.</p>
    </div>"""
    send_email(email, "Reset your LLM-Mini password", html)
