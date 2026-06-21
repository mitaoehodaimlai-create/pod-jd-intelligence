"""
FASTAPI SERVER — POD-JD Intelligence
======================================
REST API that exposes the same pipeline as HTTP endpoints.
Useful for web clients, Postman, curl, or any frontend integration.

No LangGraph — agents are called directly in sequence (same as main.py):
  email_agent → student_agent → teacher_agent → draft_service

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Auto-generated docs:
    http://localhost:8000/docs     ← Swagger UI (try all endpoints here)
    http://localhost:8000/redoc    ← ReDoc
"""

import json
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

import config
from services import draft_service

app = FastAPI(
    title="POD-JD Intelligence",
    description="POD email → JD parser → Student prep + Faculty curriculum briefs",
    version="1.0.0",
)


# ── Request models ────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    dry_run: bool = True       # True = save drafts only (default)

class ApproveRequest(BaseModel):
    draft_path: str            # Full path to the draft JSON file


# ── In-memory state for last pipeline run ─────────────────────────────────────
# Simple approach — good enough for a single-instance deployment.
# For production with multiple workers, use Redis or a database instead.

_last_run: dict = {"status": "never_run", "processed": []}


def _do_run(dry_run: bool) -> None:
    """
    Background function that runs the full pipeline.
    Called by POST /pipeline/run via FastAPI BackgroundTasks.

    Calls agents directly — no LangGraph needed for this linear pipeline:
      email_agent.parse_jd()  →  student_agent.run()  →  teacher_agent.run()
      →  draft_service.save_drafts()
    """
    global _last_run
    _last_run = {"status": "running", "dry_run": dry_run, "processed": []}

    try:
        from services.email_service import get_pod_emails
        from services.pdf_service import extract_text_from_pdf
        from agents import email_agent, student_agent, teacher_agent
        from main import _save_jd

        # Step 1: Fetch unread POD emails
        emails = get_pod_emails()
        if not emails:
            _last_run = {"status": "no_new_emails", "dry_run": dry_run, "processed": []}
            return

        summary = []
        for email in emails:
            # Step 2: Extract PDF text
            pdf_texts = []
            for filename, pdf_bytes in email["attachments"].items():
                text = extract_text_from_pdf(pdf_bytes)
                if text:
                    pdf_texts.append(f"--- PDF: {filename} ---\n{text}")

            # Step 3: Combine text for LLM
            combined = f"Subject: {email['subject']}\n\nEmail Body:\n{email['body']}"
            if pdf_texts:
                combined += "\n\n" + "\n\n".join(pdf_texts)

            # Step 4: Agent 1 — parse JD (LangChain + Groq)
            jd = email_agent.parse_jd(combined, email["message_id"])
            if not jd:
                continue

            _save_jd(jd)

            # Step 5: Agent 2 — student brief (LangChain + Groq)
            student_brief = student_agent.run(jd)

            # Step 6: Agent 3 — teacher reco (LangChain + Groq)
            teacher_reco = teacher_agent.run(jd)

            # Step 7: Save drafts
            drafts = draft_service.save_drafts(jd, student_brief, teacher_reco)

            if not dry_run:
                for draft in drafts:
                    draft_service.send_approved_draft(draft["path"])

            summary.append({
                "company": jd.get("company", "?"),
                "role":    jd.get("role", "?"),
                "drafts": [
                    {"draft_id": d["draft_id"], "recipient": d["recipient"],
                     "sent": d.get("sent", False), "path": d["path"]}
                    for d in drafts
                ],
            })

        _last_run = {"status": "ok", "dry_run": dry_run, "processed": summary}

    except Exception as exc:
        _last_run = {"status": "error", "error": str(exc), "processed": []}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Quick liveness check — confirms the server is running."""
    return {"status": "ok", "model": config.LLM_MODEL, "dry_run_default": config.DRY_RUN}


@app.get("/", tags=["System"])
def root():
    """Pipeline overview — JD counts, draft counts, last run status."""
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
        "last_run_status": _last_run.get("status"),
        "model":           config.LLM_MODEL,
        "docs":            "/docs",
    }


@app.post("/pipeline/run", tags=["Pipeline"])
def run_pipeline(req: RunRequest, background_tasks: BackgroundTasks):
    """
    Trigger the full pipeline in the background.
    Returns immediately with {"status": "started"}.
    Poll GET /pipeline/status to check progress.
    """
    background_tasks.add_task(_do_run, req.dry_run)
    return {"status": "started", "dry_run": req.dry_run}


@app.get("/pipeline/status", tags=["Pipeline"])
def pipeline_status():
    """Return the result of the most recent pipeline run."""
    return _last_run


@app.get("/jds", tags=["JDs"])
def list_jds():
    """List all parsed JD JSON files saved in output/jds/."""
    files   = sorted(config.OUTPUT_JDS.glob("*.json"))
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "filename":        f.name,
                "company":         data.get("company"),
                "role":            data.get("role"),
                "deadline":        data.get("deadline"),
                "required_skills": data.get("required_skills", []),
                "path":            str(f),
            })
        except Exception:
            pass
    return results


@app.get("/jds/{filename}", tags=["JDs"])
def get_jd(filename: str):
    """Return the full parsed JD JSON for a given filename."""
    path = config.OUTPUT_JDS / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/drafts", tags=["Drafts"])
def list_drafts(pending_only: bool = False):
    """
    List all saved draft emails.
    Add ?pending_only=true to show only unsent drafts.
    """
    return draft_service.list_all_drafts(pending_only=pending_only)


@app.get("/drafts/{draft_id}", tags=["Drafts"])
def get_draft(draft_id: str):
    """Return full draft content (including body text) for one draft ID."""
    matches = list(config.OUTPUT_DRAFTS.glob(f"{draft_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")
    return json.loads(matches[0].read_text(encoding="utf-8"))


@app.post("/drafts/approve", tags=["Drafts"])
def approve_draft(req: ApproveRequest):
    """
    Send an approved draft email via SMTP.
    Get the draft_path from GET /drafts → "path" field.
    """
    if not Path(req.draft_path).exists():
        raise HTTPException(status_code=404, detail="Draft file not found")

    ok = draft_service.send_approved_draft(req.draft_path)
    if not ok:
        raise HTTPException(status_code=500, detail="SMTP send failed — check credentials in .env")

    return {"success": True, "draft_path": req.draft_path}
