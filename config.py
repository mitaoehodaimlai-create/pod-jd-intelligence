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
# Supports both key naming conventions:
#   LANGSMITH_*       — new convention (used in .env)
#   LANGCHAIN_*       — old convention (still accepted by LangChain SDK)
_ls_api_key  = os.getenv("LANGSMITH_API_KEY")  or os.getenv("LANGCHAIN_API_KEY",  "")
_ls_tracing  = os.getenv("LANGSMITH_TRACING")  or os.getenv("LANGCHAIN_TRACING_V2", "false")
_ls_project  = os.getenv("LANGSMITH_PROJECT")  or os.getenv("LANGCHAIN_PROJECT",  "pod-jd-intelligence")
_ls_endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Expose as canonical names (used elsewhere in this project)
LANGCHAIN_API_KEY:    str = _ls_api_key
LANGCHAIN_TRACING_V2: str = _ls_tracing
LANGCHAIN_PROJECT:    str = _ls_project
LANGCHAIN_ENDPOINT:   str = _ls_endpoint

# Push both naming conventions into os.environ so LangChain SDK picks them up
os.environ["LANGSMITH_API_KEY"]    = _ls_api_key
os.environ["LANGSMITH_TRACING"]    = _ls_tracing
os.environ["LANGCHAIN_API_KEY"]    = _ls_api_key
os.environ["LANGCHAIN_TRACING_V2"] = _ls_tracing
os.environ["LANGCHAIN_PROJECT"]    = _ls_project
os.environ["LANGCHAIN_ENDPOINT"]   = _ls_endpoint


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
