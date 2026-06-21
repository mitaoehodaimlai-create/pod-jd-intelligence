"""
CONFIG
======
Central place that loads all settings from the .env file.

Every other file in this project imports from here — so if you need
to change a setting, you only change it in ONE place (.env or here).

Required .env keys:
  GROQ_API_KEY       — Groq LLM API key
  LANGCHAIN_API_KEY  — LangSmith monitoring API key
  IMAP_USER          — Email address to read from
  IMAP_PASSWORD      — App password (NOT login password)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()


# ── Output directories ────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
OUTPUT_JDS    = BASE_DIR / "output" / "jds"      # parsed JD JSON files go here
OUTPUT_DRAFTS = BASE_DIR / "output" / "drafts"   # draft email JSON files go here

# RAG vector database — ChromaDB stores JD embeddings here.
# Persists across runs so historical JDs are always available as context.
RAG_DB_PATH   = BASE_DIR / "output" / "rag_db"

# Create them if they don't exist yet
OUTPUT_JDS.mkdir(parents=True, exist_ok=True)
OUTPUT_DRAFTS.mkdir(parents=True, exist_ok=True)
RAG_DB_PATH.mkdir(parents=True, exist_ok=True)


# ── Groq LLM ─────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]        # required — raises error if missing
LLM_MODEL: str    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ── LangSmith (monitoring + tracing) ─────────────────────────────────────────
# LangChain reads these env vars automatically when any chain is invoked.
# Setting them here (from .env) ensures they're loaded before first use.
LANGCHAIN_API_KEY:      str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2:   str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT:      str = os.getenv("LANGCHAIN_PROJECT", "pod-jd-intelligence")
LANGCHAIN_ENDPOINT:     str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Push LangSmith settings into os.environ so LangChain SDK picks them up
os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"]   = LANGCHAIN_ENDPOINT


# ── Email — IMAP (read) ───────────────────────────────────────────────────────
IMAP_HOST:     str = os.getenv("IMAP_HOST", "outlook.office365.com")
IMAP_PORT:     int = int(os.getenv("IMAP_PORT", "993"))
IMAP_USER:     str = os.environ["IMAP_USER"]           # required
IMAP_PASSWORD: str = os.environ["IMAP_PASSWORD"]       # required
IMAP_FOLDER:   str = os.getenv("IMAP_FOLDER", "INBOX")


# ── Email — SMTP (send) ───────────────────────────────────────────────────────
SMTP_HOST:     str = os.getenv("SMTP_HOST", "smtp.office365.com")
SMTP_PORT:     int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER:     str = os.getenv("SMTP_USER", IMAP_USER)
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", IMAP_PASSWORD)


# ── Recipients ────────────────────────────────────────────────────────────────
STUDENT_LIST_EMAIL: str = os.getenv("STUDENT_LIST_EMAIL", "aiml-students@mitaoe.ac.in")
FACULTY_LIST_EMAIL: str = os.getenv("FACULTY_LIST_EMAIL", "aiml-faculty@mitaoe.ac.in")


# ── Security guard-rails ──────────────────────────────────────────────────────
# Only emails from these senders are processed — all others are skipped
POD_ALLOWED_SENDERS: set[str] = {
    addr.strip().lower()
    for addr in os.getenv("POD_ALLOWED_SENDERS", "placement@mitaoe.ac.in").split(",")
    if addr.strip()
}

# Email subject must contain at least one of these words
JD_SUBJECT_KEYWORDS: list[str] = [
    kw.strip().lower()
    for kw in os.getenv(
        "JD_SUBJECT_KEYWORDS",
        "placement,job opening,recruitment,hiring,campus drive,internship"
    ).split(",")
    if kw.strip()
]


# ── Pipeline behaviour ────────────────────────────────────────────────────────
# True  = save drafts only (safe default — nothing is emailed automatically)
# False = send emails immediately after pipeline
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"
