"""
MAIN APPLICATION — POD-JD Intelligence
=======================================

HOW THE PIPELINE WORKS (step by step):

  Step 1  → email_service  reads unread emails from POD inbox (IMAP)
  Step 2  → pdf_service    extracts text from PDF attachments
  Step 3  → email_agent    LangChain+Groq parses email+PDF text → structured JD
  Step 4  → rag_store      embeds JD → ChromaDB; retrieves similar past JDs
  Step 5  → student_agent  LangChain+Groq generates student interview prep brief
            (with RAG context from similar past JDs)
  Step 6  → teacher_agent  LangChain+Groq generates faculty curriculum reco
            (with RAG context from similar past JDs)
  Step 7  → draft_service  saves both outputs as local draft files (NOT auto-sent)
  Step 8  → human approval → run  python main.py approve <path>  to send

RAG EXPLAINED:
  R — Retrieve: past JDs are stored as vectors in ChromaDB (output/rag_db/)
  A — Augment:  top-3 similar JDs are added to the agent's LLM prompt
  G — Generate: LLM sees current JD + patterns → better recommendations

PIPELINE FLOW:

  POD Inbox (IMAP)
       ↓
  email_service.get_pod_emails()
       ↓   list of email dicts
  FOR EACH EMAIL:
       ↓
  pdf_service.extract_text_from_pdf()   ← reads PDF attachments
       ↓   combined text
  email_agent.parse_jd()                ← LangChain | ChatGroq | JsonOutputParser
       ↓   jd dict
  rag_store.add_jd(jd)                  ← embed + store in ChromaDB
  rag_store.get_similar_jds(jd)         ← semantic search → similar past JDs
       ↓   rag_context string
  student_agent.run(jd, rag_context)    ← LangChain | ChatGroq | StrOutputParser
  teacher_agent.run(jd, rag_context)    ← LangChain | ChatGroq | StrOutputParser
       ↓   two Markdown strings
  draft_service.save_drafts()           ← saves to output/drafts/*.json

ALL LLM CALLS ARE TRACED IN LANGSMITH:
  Open https://smith.langchain.com → project "pod-jd-intelligence"

CLI COMMANDS:
  python main.py run                       → process new emails (safe, drafts only)
  python main.py run --send                → process + send emails immediately
  python main.py list-jds                  → show all parsed JDs
  python main.py list-drafts              → show all saved drafts
  python main.py approve <path>           → send one specific draft
  python main.py status                   → pipeline stats + RAG store info
  python main.py serve                    → start FastAPI REST server (port 8000)
  python main.py serve --port=9000        → custom port
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Agents — do the LLM work via LangChain + Groq
from agents import email_agent, student_agent, teacher_agent, skill_upgrade_agent, reminder_agent

# Services — handle email I/O, PDF extraction, draft file management
from services.email_service import get_pod_emails
from services.pdf_service import extract_text_from_pdf
from services import draft_service

import config

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# RAG STORE — Retrieval-Augmented Generation (inline, no separate rag/ package)
# ─────────────────────────────────────────────────────────────────────────────
#
# RAG in one line: "give the LLM relevant past JDs so it generates better output"
#
# HOW IT WORKS:
#   R — Retrieve : past JDs stored as 384-number vectors in ChromaDB on disk
#   A — Augment  : top-3 similar JDs appended to the agent's LLM prompt
#   G — Generate : LLM sees current JD + history → richer recommendations
#
# WITHOUT RAG: AI sees only THIS JD → standard recommendations
# WITH RAG:    AI spots patterns like "Python in 90% of similar JDs → top priority"
#              or "DSA gap appeared 4 times → escalate to curriculum board"
#
# EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2
#   - Free and local (no API key), downloads ~90MB once to ~/.cache/huggingface/
#   - Runs on CPU, fast (~5ms per JD)
#   - Converts text → 384 numbers that capture semantic meaning
#
# CHROMADB: local vector database — stores and searches those 384-dim vectors
#   - Persists to output/rag_db/ so JDs accumulate across runs
#   - Cosine similarity search finds "similar" JDs in milliseconds
#
# LANGCHAIN COMPONENTS USED:
#   HuggingFaceEmbeddings   text → 384-dim vector (wraps sentence-transformers)
#   Chroma                  stores/searches vectors on disk (wraps ChromaDB)
#   Document                standard type: page_content (embedded) + metadata (stored as-is)

_RAG_AVAILABLE = False
_chroma_client: Optional[object] = None
_chroma_collection: Optional[object] = None

try:
    import chromadb as _chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as _ChromaEF
    _RAG_AVAILABLE = True
except ImportError:
    pass   # RAG is optional — pipeline works without it


def _rag_get_collection():
    """
    Initialize (or return cached) ChromaDB persistent collection.

    Uses ChromaDB's built-in ONNX embedding (all-MiniLM-L6-v2, 384-dim).
    No sentence-transformers / torch / torchvision required.
    First call downloads ~79MB ONNX model once to ~/.cache/chroma/
    """
    global _chroma_client, _chroma_collection

    if _chroma_collection is not None:
        return _chroma_collection

    print("RAG: Initialising ChromaDB (ONNX all-MiniLM-L6-v2)...")
    _chroma_client = _chromadb.PersistentClient(path=str(config.RAG_DB_PATH))
    _chroma_collection = _chroma_client.get_or_create_collection(
        name               = "jds",
        embedding_function = _ChromaEF(),
        metadata           = {"hnsw:space": "cosine"},
    )
    count = _chroma_collection.count()
    print(f"RAG: Ready — {count} JD(s) in store at {config.RAG_DB_PATH.name}/")
    return _chroma_collection


def _rag_jd_to_text(jd: dict) -> str:
    """
    Convert a JD dict to a single text string for embedding.

    Only the semantically meaningful fields are included in the vector
    (role, skills, tech). Company/location/CTC go into metadata only.
    """
    parts = [
        f"Role: {jd.get('role', '')}",
        f"Skills: {', '.join(jd.get('required_skills', []))}",
        f"Technologies: {', '.join(jd.get('tools_tech', []))}",
        f"Responsibilities: {'; '.join(jd.get('responsibilities', [])[:3])}",
        f"Nice to have: {', '.join(jd.get('nice_to_have', []))}",
        f"Eligibility: {jd.get('eligibility', '')}",
    ]
    return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())


def rag_add_jd(jd: dict) -> bool:
    """
    STORE STEP — embed a parsed JD and save to ChromaDB.
    Returns True on success, False on failure (non-fatal).
    """
    if not _RAG_AVAILABLE:
        return False
    try:
        col     = _rag_get_collection()
        jd_text = _rag_jd_to_text(jd)
        if not jd_text.strip():
            return False

        doc_id = jd.get("message_id", "") or f"{jd.get('company','')}-{jd.get('role','')}"
        doc_id = doc_id.replace(" ", "_")[:64]

        col.upsert(
            ids       = [doc_id],
            documents = [jd_text],
            metadatas = [{
                "company":  str(jd.get("company", "Unknown")),
                "role":     str(jd.get("role", "Unknown")),
                "location": str(jd.get("location", "")),
                "deadline": str(jd.get("deadline", "")),
                "ctc":      str(jd.get("ctc", "")),
                "jd_json":  json.dumps(jd),
            }],
        )
        print(f"RAG: Stored → {jd.get('company')} / {jd.get('role')}  (total: {col.count()})")
        return True
    except Exception as e:
        print(f"RAG: add failed — {e}")
        return False


def rag_get_similar_jds(jd: dict, k: int = 3) -> list:
    """
    RETRIEVE STEP — find k semantically similar past JDs via cosine similarity.
    Returns empty list if store is empty or on error (non-fatal).
    """
    if not _RAG_AVAILABLE:
        return []
    try:
        col = _rag_get_collection()
        if col.count() == 0:
            return []
        query_text = _rag_jd_to_text(jd)
        if not query_text.strip():
            return []

        results = col.query(query_texts=[query_text], n_results=min(k, col.count()))
        similar = []
        for meta in (results.get("metadatas") or [[]])[0]:
            jd_json_str = meta.get("jd_json", "")
            if jd_json_str:
                try:
                    similar.append(json.loads(jd_json_str))
                except json.JSONDecodeError:
                    pass
        return similar
    except Exception as e:
        print(f"RAG: retrieve failed — {e}")
        return []


def rag_format_context(similar_jds: list) -> str:
    """
    AUGMENT STEP — format retrieved JDs as plain text for the LLM prompt.
    Returns empty string if no similar JDs (agents work fine without it).
    """
    if not similar_jds:
        return ""
    lines = [f"[{len(similar_jds)} SIMILAR PAST JD(s) FROM THIS SEMESTER FOR CONTEXT]"]
    for i, past_jd in enumerate(similar_jds, 1):
        lines.append(f"\nPast JD #{i}:")
        lines.append(f"  Company : {past_jd.get('company', '?')}")
        lines.append(f"  Role    : {past_jd.get('role', '?')}")
        lines.append(f"  Skills  : {', '.join(past_jd.get('required_skills', []))}")
        lines.append(f"  Tech    : {', '.join(past_jd.get('tools_tech', []))}")
        if past_jd.get("ctc"):
            lines.append(f"  CTC     : {past_jd.get('ctc')}")
    return "\n".join(lines)


def rag_get_skill_trends(top_n: int = 10) -> dict:
    """
    Count skill frequency across ALL stored JDs.
    Returns {skill: count} sorted descending — useful for curriculum planning.
    """
    if not _RAG_AVAILABLE:
        return {}
    try:
        col = _rag_get_collection()
        if col.count() == 0:
            return {}
        all_docs     = col.get(include=["metadatas"])
        skill_counts: dict = {}
        for meta in all_docs.get("metadatas", []):
            jd_json_str = meta.get("jd_json", "")
            if not jd_json_str:
                continue
            try:
                past_jd = json.loads(jd_json_str)
                for skill in past_jd.get("required_skills", []) + past_jd.get("tools_tech", []):
                    key = skill.strip().lower()
                    if key:
                        skill_counts[key] = skill_counts.get(key, 0) + 1
            except Exception:
                pass
        return dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])
    except Exception as e:
        print(f"RAG: skill_trends failed — {e}")
        return {}


def rag_get_store_stats() -> dict:
    """Return count, companies, top skills — shown in `python main.py status`."""
    if not _RAG_AVAILABLE:
        return {"total_jds": 0, "companies": [], "top_skills": {}}
    try:
        col   = _rag_get_collection()
        total = col.count()
        if total == 0:
            return {"total_jds": 0, "companies": [], "top_skills": {}}
        all_docs  = col.get(include=["metadatas"])
        companies = sorted({
            m.get("company", "") for m in all_docs.get("metadatas", []) if m.get("company")
        })
        return {"total_jds": total, "companies": companies, "top_skills": rag_get_skill_trends(5)}
    except Exception as e:
        return {"total_jds": 0, "companies": [], "top_skills": {}, "error": str(e)}


# ── FASTAPI REST SERVER (started by `python main.py serve`) ──────────────────
# api.py is merged here so the project has a single entry point.
# FastAPI is optional — CLI and MCP server work without it.
# Install: pip install fastapi "uvicorn[standard]"

_FASTAPI_AVAILABLE = False
try:
    from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
    from pydantic import BaseModel as _PydanticModel
    import uvicorn as _uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    pass

if _FASTAPI_AVAILABLE:
    # FastAPI app — all endpoints defined here, started only by cmd_serve()
    _api_app = FastAPI(
        title       = "POD-JD Intelligence",
        description = "POD email → JD parser → Student prep + Faculty curriculum briefs",
        version     = "1.0.0",
    )

    _last_run_state: dict = {"status": "never_run", "processed": []}

    class _RunRequest(_PydanticModel):
        dry_run: bool = True      # True = save drafts, don't send (safe default)

    class _ApproveRequest(_PydanticModel):
        draft_path: str           # Full path from GET /drafts → "path" field

    def _api_do_run(dry_run: bool) -> None:
        """Background task: run the pipeline and update _last_run_state."""
        global _last_run_state
        _last_run_state = {"status": "running", "dry_run": dry_run, "processed": []}
        try:
            run_pipeline(dry_run=dry_run)
            _last_run_state["status"] = "ok"
        except Exception as exc:
            _last_run_state = {"status": "error", "error": str(exc), "processed": []}

    # ── System endpoints ─────────────────────────────────────────────────────

    @_api_app.get("/health", tags=["System"])
    def _api_health():
        """Liveness check — confirms the server is running."""
        return {"status": "ok", "model": config.LLM_MODEL, "dry_run_default": config.DRY_RUN}

    @_api_app.get("/", tags=["System"])
    def _api_root():
        """Overview — JD counts, draft counts, last run status."""
        draft_files = list(config.OUTPUT_DRAFTS.glob("*.json"))
        sent_count  = sum(
            1 for f in draft_files
            if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
        )
        return {
            "service":         "POD-JD Intelligence",
            "jds_processed":   len(list(config.OUTPUT_JDS.glob("*.json"))),
            "drafts_total":    len(draft_files),
            "drafts_sent":     sent_count,
            "drafts_pending":  len(draft_files) - sent_count,
            "last_run_status": _last_run_state.get("status"),
            "rag_available":   _RAG_AVAILABLE,
            "model":           config.LLM_MODEL,
            "docs":            "/docs",
        }

    # ── Pipeline endpoints ───────────────────────────────────────────────────

    @_api_app.post("/pipeline/run", tags=["Pipeline"])
    def _api_run_pipeline(req: _RunRequest, bg: BackgroundTasks):
        """Trigger the pipeline in the background. Poll /pipeline/status for result."""
        bg.add_task(_api_do_run, req.dry_run)
        return {"status": "started", "dry_run": req.dry_run}

    @_api_app.get("/pipeline/status", tags=["Pipeline"])
    def _api_pipeline_status():
        """Return the result of the last pipeline run."""
        return _last_run_state

    @_api_app.post("/pipeline/upload", tags=["Pipeline"])
    async def _api_upload_pdf(
        file: UploadFile = File(..., description="PDF file containing the Job Description"),
        dry_run: bool = True,
    ):
        """
        Upload a PDF directly and run the full pipeline — no email needed.

        Use this to test the pipeline from Swagger UI:
        1. Click 'Try it out'
        2. Choose a JD PDF file
        3. Click Execute — get student brief + teacher reco saved as drafts
        """
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(pdf_bytes)
        if not pdf_text:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        combined = f"Subject: {file.filename}\n\nPDF Content:\n{pdf_text}"

        # Agent 1 — parse JD
        import uuid as _uuid
        message_id = f"upload-{_uuid.uuid4().hex[:8]}"
        jd = email_agent.parse_jd(combined, message_id)
        if not jd:
            raise HTTPException(status_code=422, detail="LLM could not parse a Job Description from this PDF.")

        _save_jd(jd)

        # RAG context (if available)
        rag_context = ""
        if _RAG_AVAILABLE:
            rag_add_jd(jd)
            similar     = rag_get_similar_jds(jd)
            rag_context = rag_format_context(similar)

        # Agent 2 — student brief
        student_brief = student_agent.run(jd, rag_context=rag_context)

        # Agent 3 — teacher reco
        teacher_reco = teacher_agent.run(jd, rag_context=rag_context)

        # Save drafts
        drafts = draft_service.save_drafts(jd, student_brief, teacher_reco)

        if not dry_run:
            for d in drafts:
                draft_service.send_approved_draft(d["path"])

        return {
            "status":   "ok",
            "dry_run":  dry_run,
            "filename": file.filename,
            "jd": {
                "company":         jd.get("company"),
                "role":            jd.get("role"),
                "deadline":        jd.get("deadline"),
                "required_skills": jd.get("required_skills", []),
            },
            "drafts": [
                {"draft_id": d["draft_id"], "recipient": d["recipient"], "path": d["path"]}
                for d in drafts
            ],
        }

    # ── JD endpoints ─────────────────────────────────────────────────────────

    @_api_app.get("/jds", tags=["JDs"])
    def _api_list_jds():
        """List all parsed JD JSON files in output/jds/."""
        results = []
        for f in sorted(config.OUTPUT_JDS.glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                results.append({
                    "filename":        f.name,
                    "company":         d.get("company"),
                    "role":            d.get("role"),
                    "deadline":        d.get("deadline"),
                    "required_skills": d.get("required_skills", []),
                    "path":            str(f),
                })
            except Exception:
                pass
        return results

    @_api_app.get("/jds/{filename}", tags=["JDs"])
    def _api_get_jd(filename: str):
        """Return the full JD JSON for a given filename."""
        path = config.OUTPUT_JDS / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    # ── Draft endpoints ───────────────────────────────────────────────────────

    @_api_app.get("/drafts", tags=["Drafts"])
    def _api_list_drafts(pending_only: bool = False):
        """List saved draft emails. Add ?pending_only=true for unsent only."""
        return draft_service.list_all_drafts(pending_only=pending_only)

    @_api_app.get("/drafts/{draft_id}", tags=["Drafts"])
    def _api_get_draft(draft_id: str):
        """Return full draft content (body text included) for one draft ID."""
        matches = list(config.OUTPUT_DRAFTS.glob(f"{draft_id}.json"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")
        return json.loads(matches[0].read_text(encoding="utf-8"))

    @_api_app.post("/drafts/approve", tags=["Drafts"])
    def _api_approve_draft(req: _ApproveRequest):
        """Send one approved draft via SMTP. Get draft_path from GET /drafts."""
        if not Path(req.draft_path).exists():
            raise HTTPException(status_code=404, detail="Draft file not found")
        ok = draft_service.send_approved_draft(req.draft_path)
        if not ok:
            raise HTTPException(status_code=500, detail="SMTP send failed — check .env")
        return {"success": True, "draft_path": req.draft_path}

    # ── RAG endpoint ──────────────────────────────────────────────────────────

    @_api_app.get("/rag/stats", tags=["RAG"])
    def _api_rag_stats():
        """Return RAG store stats: total JDs, companies, top skills."""
        if not _RAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="RAG not available (chromadb not installed)")
        return rag_get_store_stats()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — core logic
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(dry_run: bool = True) -> None:
    """
    Run the full pipeline for all new POD emails.

    No LangGraph needed — simple for-loop calling each agent in sequence.

    Args:
        dry_run: True  = save drafts only (safe default — nothing is sent)
                 False = save drafts AND send via SMTP immediately
    """
    mode = "[yellow]Dry Run — drafts only[/yellow]" if dry_run \
           else "[green]Live — emails will be sent[/green]"

    console.print(Panel(
        f"[bold cyan]POD-JD Intelligence[/bold cyan]\n"
        f"LLM: LangChain + Groq ({config.LLM_MODEL})\n"
        f"RAG: {'ChromaDB vector store (ON)' if _RAG_AVAILABLE else 'not installed (OFF)'}\n"
        f"Mode: {mode}",
        border_style="blue",
    ))

    # ── STEP 1: Fetch emails ──────────────────────────────────────────────────
    emails = get_pod_emails()
    if not emails:
        console.print("[yellow]No new POD emails found.[/yellow]")
        return

    for i, email in enumerate(emails, start=1):
        console.rule(f"Email {i}/{len(emails)} — {email['subject'][:65]}")

        # ── STEP 2: Extract PDF text ──────────────────────────────────────────
        # pdfplumber primary, pypdf fallback — see services/pdf_service.py
        pdf_texts = []
        for filename, pdf_bytes in email["attachments"].items():
            text = extract_text_from_pdf(pdf_bytes)
            if text:
                pdf_texts.append(f"--- PDF: {filename} ---\n{text}")
                console.print(f"  PDF: {len(text)} chars extracted from [cyan]{filename}[/cyan]")

        combined = f"Subject: {email['subject']}\n\nEmail Body:\n{email['body']}"
        if pdf_texts:
            combined += "\n\n" + "\n\n".join(pdf_texts)

        # ── STEP 3: Agent 1 — Parse JD (LangChain + Groq) ────────────────────
        # LLM chain: ChatPromptTemplate → ChatGroq → JsonOutputParser
        # Converts unstructured email/PDF text into a structured Python dict.
        # LangSmith traces this call — see input/output at smith.langchain.com
        with console.status("Agent 1 (email_agent): parsing JD with Groq LLM..."):
            jd = email_agent.parse_jd(combined, email["message_id"])

        if not jd:
            console.print("[red]Could not parse JD. Skipping this email.[/red]")
            continue

        jd_path = _save_jd(jd)

        console.print(Panel(
            f"[bold]{jd.get('company')}[/bold] — {jd.get('role')}\n"
            f"Location : {jd.get('location')}     CTC : {jd.get('ctc')}\n"
            f"Deadline : {jd.get('deadline')}\n"
            f"Skills   : {', '.join(jd.get('required_skills', [])[:4])}",
            title="Parsed JD",
            border_style="green",
        ))

        # ── STEP 4: RAG — Store + Retrieve ───────────────────────────────────
        # STORE:    Embed the current JD (384-dim vector) into ChromaDB.
        #           Future runs will find this JD when searching for similar roles.
        # RETRIEVE: Find top-3 JDs already in the store that are semantically
        #           similar to the current one (same skill domain, similar role).
        # AUGMENT:  Format those JDs as a context string to inject into agent prompts.
        rag_context = ""
        if _RAG_AVAILABLE:
            rag_add_jd(jd)                              # embed + persist to disk
            similar     = rag_get_similar_jds(jd)       # semantic search in ChromaDB
            rag_context = rag_format_context(similar)   # format for LLM prompt
            if similar:
                console.print(f"  RAG: [cyan]{len(similar)} similar past JD(s)[/cyan] added to agent context")

        # ── STEP 5: Agent 2 — Student Prep Brief ─────────────────────────────
        with console.status("Agent 2 (student_agent): generating prep brief..."):
            student_brief = student_agent.run(jd, rag_context=rag_context)

        # ── STEP 6: Agent 3 — Teacher Curriculum Reco ────────────────────────
        with console.status("Agent 3 (teacher_agent): generating curriculum reco..."):
            teacher_reco = teacher_agent.run(jd, rag_context=rag_context)

        # ── STEP 7: Agent 4 — Skill Upgrade Analysis ─────────────────────────
        # Compares JD requirements vs AIML curriculum → prioritised upgrade plan
        # RAG context highlights skills that recur across similar JDs (critical gaps)
        with console.status("Agent 4 (skill_upgrade_agent): analysing skill gaps..."):
            skill_upgrade = skill_upgrade_agent.run(jd, rag_context=rag_context)

        # ── STEP 8: Agent 5 — Application Reminder + Deadline Plan ──────────
        # Generates deadline-aware day-by-day prep plan (7/14/30-day adaptive)
        with console.status("Agent 5 (reminder_agent): building deadline prep plan..."):
            reminder = reminder_agent.run(jd)

        # ── STEP 9: Save draft emails ─────────────────────────────────────────
        drafts = draft_service.save_drafts(
            jd, student_brief, teacher_reco,
            skill_upgrade=skill_upgrade,
            reminder=reminder,
        )

        # ── STEP 10: Send (only if not dry run) ──────────────────────────────
        if not dry_run:
            for draft in drafts:
                draft_service.send_approved_draft(draft["path"])

        for draft in drafts:
            status = "[green]SENT[/green]" if draft.get("sent") \
                     else "[yellow]Saved — pending approval[/yellow]"
            console.print(f"  Draft → {draft['recipient']} | {status}")
            console.print(f"  File  : [dim]{Path(draft['path']).name}[/dim]")

    if dry_run:
        console.print(
            "\n[dim]Use [bold]python main.py list-drafts[/bold] to see drafts.  "
            "Use [bold]python main.py approve <path>[/bold] to send one.[/dim]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_list_jds() -> None:
    """Show a table of all saved JD JSON files."""
    files = sorted(config.OUTPUT_JDS.glob("*.json"))
    if not files:
        console.print("[yellow]No JDs found in output/jds/[/yellow]")
        return

    table = Table(title="Parsed Job Descriptions", show_lines=True)
    table.add_column("#",        style="dim", width=4)
    table.add_column("Company",  style="bold cyan")
    table.add_column("Role")
    table.add_column("CTC")
    table.add_column("Deadline")
    table.add_column("Top Skills", style="dim")

    for i, f in enumerate(files, 1):
        d = json.loads(f.read_text(encoding="utf-8"))
        table.add_row(
            str(i),
            d.get("company", "?"),
            d.get("role", "?"),
            d.get("ctc", "?"),
            d.get("deadline", "?"),
            ", ".join(d.get("required_skills", [])[:3]),
        )
    console.print(table)


def cmd_list_drafts() -> None:
    """Show a table of all saved draft emails."""
    drafts = draft_service.list_all_drafts()
    if not drafts:
        console.print("[yellow]No drafts found in output/drafts/[/yellow]")
        return

    table = Table(title="Draft Emails", show_lines=True)
    table.add_column("Draft ID",  style="dim")
    table.add_column("Recipient", style="cyan")
    table.add_column("Subject")
    table.add_column("Sent?", justify="center")
    table.add_column("File", style="dim")

    for d in drafts:
        sent = "[green]Yes[/green]" if d["sent"] else "[yellow]Pending[/yellow]"
        subj = d["subject"]
        table.add_row(
            d["draft_id"],
            d["recipient"],
            subj[:55] + "…" if len(subj) > 55 else subj,
            sent,
            Path(d["path"]).name,
        )
    console.print(table)


def cmd_approve(draft_path: str) -> None:
    """Send one specific draft via SMTP."""
    console.print(f"Sending: [cyan]{draft_path}[/cyan]")
    ok = draft_service.send_approved_draft(draft_path)
    console.print("[green]✓ Email sent.[/green]" if ok
                  else "[red]✗ Failed — check SMTP credentials in .env[/red]")


def cmd_status() -> None:
    """Show pipeline stats, config, and RAG store info."""
    all_drafts = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )
    tracing = config.LANGCHAIN_TRACING_V2.lower() == "true"

    # RAG store stats (if available)
    rag_line = "[dim]not installed — pip install chromadb langchain-community sentence-transformers[/dim]"
    if _RAG_AVAILABLE:
        stats = rag_get_store_stats()
        top_skills = ", ".join(
            f"{k}({v})" for k, v in list(stats.get("top_skills", {}).items())[:5]
        )
        rag_line = (
            f"[green]{stats['total_jds']} JDs indexed[/green]\n"
            f"  Companies : {', '.join(stats.get('companies', []))}\n"
            f"  Top skills: {top_skills or '(none yet)'}"
        )

    console.print(Panel(
        f"JDs Parsed        : [bold]{len(list(config.OUTPUT_JDS.glob('*.json')))}[/bold]\n"
        f"Drafts Total      : {len(all_drafts)}\n"
        f"Drafts Sent       : [green]{sent_count}[/green]\n"
        f"Drafts Pending    : [yellow]{len(all_drafts) - sent_count}[/yellow]\n\n"
        f"LLM               : {config.LLM_MODEL}  (LangChain + Groq)\n"
        f"LangSmith Tracing : {'[green]ON[/green] — ' + config.LANGCHAIN_PROJECT if tracing else '[dim]off[/dim]'}\n"
        f"Dry Run Default   : {config.DRY_RUN}\n"
        f"POD Senders       : {', '.join(config.POD_ALLOWED_SENDERS)}\n\n"
        f"RAG Store         : {rag_line}",
        title="Pipeline Status",
        border_style="blue",
    ))


def cmd_serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the FastAPI REST server (api.py merged here).

    Endpoints available at http://localhost:8000/docs (Swagger UI):
      GET  /health              → liveness check
      GET  /                    → pipeline overview
      POST /pipeline/run        → trigger pipeline (background)
      GET  /pipeline/status     → last run result
      GET  /jds                 → list parsed JDs
      GET  /jds/{filename}      → get one JD
      GET  /drafts              → list draft emails
      GET  /drafts/{draft_id}   → get one draft
      POST /drafts/approve      → send approved draft
      GET  /rag/stats           → RAG store statistics
    """
    if not _FASTAPI_AVAILABLE:
        console.print("[red]FastAPI or uvicorn not installed.[/red]")
        console.print("Install: [bold]pip install fastapi 'uvicorn[standard]'[/bold]")
        return

    console.print(Panel(
        f"FastAPI server starting\n"
        f"  URL     : [cyan]http://{host}:{port}[/cyan]\n"
        f"  Swagger : [cyan]http://{host}:{port}/docs[/cyan]\n"
        f"  ReDoc   : [cyan]http://{host}:{port}/redoc[/cyan]\n"
        f"  Ctrl+C  : stop server",
        title="REST API Server",
        border_style="green",
    ))
    _uvicorn.run(_api_app, host=host, port=port)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save_jd(jd: dict) -> str:
    """Save a parsed JD dict as a timestamped JSON file in output/jds/."""
    company   = jd.get("company", "unknown").replace(" ", "_").lower()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path      = config.OUTPUT_JDS / f"{company}_{timestamp}.json"
    path.write_text(json.dumps(jd, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"  ✓ JD saved → [dim]{path.name}[/dim]")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "run":
        run_pipeline(dry_run=("--send" not in args))
    elif args[0] == "list-jds":
        cmd_list_jds()
    elif args[0] == "list-drafts":
        cmd_list_drafts()
    elif args[0] == "approve" and len(args) > 1:
        cmd_approve(args[1])
    elif args[0] == "status":
        cmd_status()
    elif args[0] == "serve":
        host = "0.0.0.0"
        port = 8000
        for arg in args[1:]:
            if arg.startswith("--port="):
                port = int(arg.split("=")[1])
            elif arg.startswith("--host="):
                host = arg.split("=")[1]
        cmd_serve(host, port)
    else:
        console.print(__doc__)


if __name__ == "__main__":
    main()
