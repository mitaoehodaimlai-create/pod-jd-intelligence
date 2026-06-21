"""
POD-JD Intelligence — FastAPI REST server

Exposes the same pipeline as HTTP endpoints so any web client,
Postman, or curl can trigger runs and inspect results.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Docs (auto-generated):
    http://localhost:8000/docs     ← Swagger UI
    http://localhost:8000/redoc    ← ReDoc
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config

app = FastAPI(
    title="POD-JD Intelligence",
    description="Multi-agent pipeline: POD emails → JD analysis → Student prep + Faculty curriculum briefs",
    version="1.0.0",
)


# ── request / response models ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    dry_run: bool = True


class ApproveRequest(BaseModel):
    draft_path: str


class RunResult(BaseModel):
    status: str
    dry_run: bool
    processed: list[dict]


# ── background state (simple in-memory; replace with Redis for prod) ──────────

_last_run: dict = {"status": "never_run", "processed": []}


def _do_run(dry_run: bool) -> None:
    global _last_run
    _last_run = {"status": "running", "dry_run": dry_run, "processed": []}
    try:
        from tools.email_reader import fetch_pod_emails
        from workflow.graph import process_email

        emails = fetch_pod_emails(mark_seen=False)
        if not emails:
            _last_run = {"status": "no_new_emails", "dry_run": dry_run, "processed": []}
            return

        summary = []
        for email in emails:
            result = process_email(email, dry_run=dry_run)
            jd = result.get("jd") or {}
            summary.append(
                {
                    "email_uid": email.uid,
                    "company": jd.get("company", "?"),
                    "role": jd.get("role", "?"),
                    "error": result.get("error"),
                    "drafts": [
                        {
                            "draft_id": d["draft_id"],
                            "recipient": d["recipient"],
                            "sent": d["sent"],
                            "path": d["path"],
                        }
                        for d in result.get("drafts", [])
                    ],
                }
            )

        _last_run = {"status": "ok", "dry_run": dry_run, "processed": summary}
    except Exception as exc:
        _last_run = {"status": "error", "error": str(exc), "processed": []}


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Quick liveness check."""
    return {"status": "ok", "model": config.LLM_MODEL, "dry_run_default": config.DRY_RUN}


@app.get("/", tags=["System"])
def root():
    """Pipeline status + counts."""
    jd_count     = len(list(config.OUTPUT_JDS.glob("*.json")))
    draft_files  = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count   = sum(
        1 for f in draft_files
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )
    return {
        "service": "POD-JD Intelligence",
        "jds_processed": jd_count,
        "drafts_total": len(draft_files),
        "drafts_sent": sent_count,
        "drafts_pending": len(draft_files) - sent_count,
        "last_run_status": _last_run.get("status"),
        "model": config.LLM_MODEL,
        "pod_senders": list(config.POD_ALLOWED_SENDERS),
        "docs": "/docs",
    }


@app.post("/pipeline/run", tags=["Pipeline"])
def run_pipeline(req: RunRequest, background_tasks: BackgroundTasks):
    """
    Trigger the full pipeline in the background.
    Returns immediately with `{"status": "started"}`.
    Poll `GET /pipeline/status` to see when it finishes.
    """
    background_tasks.add_task(_do_run, req.dry_run)
    return {"status": "started", "dry_run": req.dry_run}


@app.get("/pipeline/status", tags=["Pipeline"])
def pipeline_status():
    """Return the result of the most recent pipeline run."""
    return _last_run


@app.get("/jds", tags=["JDs"])
def list_jds():
    """List all parsed JD JSON files."""
    files = sorted(config.OUTPUT_JDS.glob("*.json"))
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(
                {
                    "filename": f.name,
                    "company": data.get("company"),
                    "role": data.get("role"),
                    "deadline": data.get("deadline"),
                    "required_skills": data.get("required_skills", []),
                    "path": str(f),
                }
            )
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
    List all draft emails.
    Set `pending_only=true` to filter only unsent drafts.
    """
    files = sorted(config.OUTPUT_DRAFTS.glob("*.json"))
    drafts = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if pending_only and data.get("sent", False):
                continue
            drafts.append(
                {
                    "draft_id": data["draft_id"],
                    "recipient": data["to"],
                    "subject": data["subject"],
                    "sent": data.get("sent", False),
                    "path": str(f),
                }
            )
        except Exception:
            pass
    return drafts


@app.get("/drafts/{draft_id}", tags=["Drafts"])
def get_draft(draft_id: str):
    """Return full content of a draft (including body text)."""
    matches = list(config.OUTPUT_DRAFTS.glob(f"{draft_id}.json"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id} not found")
    return json.loads(matches[0].read_text(encoding="utf-8"))


@app.post("/drafts/approve", tags=["Drafts"])
def approve_draft(req: ApproveRequest):
    """
    Send an approved draft via SMTP.
    Pass the full file path from `GET /drafts`.
    """
    from workflow.nodes.notify import approve_and_send

    if not Path(req.draft_path).exists():
        raise HTTPException(status_code=404, detail="Draft file not found")

    ok = approve_and_send(req.draft_path)
    if not ok:
        raise HTTPException(status_code=500, detail="SMTP send failed — check credentials and server logs")
    return {"success": True, "draft_path": req.draft_path}
