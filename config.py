import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUT_JDS    = BASE_DIR / "output" / "jds"
OUTPUT_DRAFTS = BASE_DIR / "output" / "drafts"
OUTPUT_JDS.mkdir(parents=True, exist_ok=True)
OUTPUT_DRAFTS.mkdir(parents=True, exist_ok=True)

# ── Groq ──────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
LLM_MODEL: str    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ── LangSmith ────────────────────────────────────────────────────
# Variables read directly by the langsmith / langchain SDK at import time.
# Setting them here (from .env) ensures they're present before first import.
os.environ.setdefault("LANGCHAIN_TRACING_V2",  os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY",     os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT",     os.getenv("LANGCHAIN_PROJECT", "pod-jd-intelligence"))
os.environ.setdefault("LANGCHAIN_ENDPOINT",    os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))

# ── Email (IMAP) ──────────────────────────────────────────────────
IMAP_HOST:     str = os.getenv("IMAP_HOST", "outlook.office365.com")
IMAP_PORT:     int = int(os.getenv("IMAP_PORT", "993"))
IMAP_USER:     str = os.environ["IMAP_USER"]
IMAP_PASSWORD: str = os.environ["IMAP_PASSWORD"]
IMAP_FOLDER:   str = os.getenv("IMAP_FOLDER", "INBOX")

# ── Email (SMTP) ──────────────────────────────────────────────────
SMTP_HOST:     str = os.getenv("SMTP_HOST", "smtp.office365.com")
SMTP_PORT:     int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER:     str = os.getenv("SMTP_USER", IMAP_USER)
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", IMAP_PASSWORD)

# ── Recipients ────────────────────────────────────────────────────
STUDENT_LIST_EMAIL: str = os.getenv("STUDENT_LIST_EMAIL", "aiml-students@mitaoe.ac.in")
FACULTY_LIST_EMAIL: str = os.getenv("FACULTY_LIST_EMAIL", "aiml-faculty@mitaoe.ac.in")

# ── Guard-rails ───────────────────────────────────────────────────
POD_ALLOWED_SENDERS: set[str] = {
    s.strip().lower()
    for s in os.getenv("POD_ALLOWED_SENDERS", "placement@mitaoe.ac.in").split(",")
    if s.strip()
}

DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"

JD_SUBJECT_KEYWORDS: list[str] = [
    kw.strip().lower()
    for kw in os.getenv(
        "JD_SUBJECT_KEYWORDS",
        "placement,job opening,recruitment,hiring,campus drive,internship",
    ).split(",")
    if kw.strip()
]
