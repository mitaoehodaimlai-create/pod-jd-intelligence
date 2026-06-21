"""Notify node — composes draft emails and saves them locally (or sends via SMTP)."""
from __future__ import annotations

import json
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import config
from workflow.state import DraftRecord, PipelineState

OUTPUT_DRAFTS = config.OUTPUT_DRAFTS


def _build_mime(to: str, subject: str, body: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["From"] = config.SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg


def _save_draft(draft_id: str, to: str, subject: str, body: str) -> str:
    path = OUTPUT_DRAFTS / f"{draft_id}.json"
    path.write_text(
        json.dumps(
            {"draft_id": draft_id, "to": to, "subject": subject, "body": body, "sent": False},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(path)


def _send_smtp(to: str, subject: str, body: str) -> None:
    msg = _build_mime(to, subject, body)
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.sendmail(config.SMTP_USER, [to], msg.as_string())


def notify_node(state: PipelineState) -> dict:
    jd = state.get("jd") or {}
    company = jd.get("company", "Company")
    role = jd.get("role", "Role")
    base_subject = f"[Placement] {company} — {role}"

    drafts: list[DraftRecord] = []

    for recipient, body, label in [
        (config.STUDENT_LIST_EMAIL, state.get("student_brief", ""), "student"),
        (config.FACULTY_LIST_EMAIL, state.get("faculty_reco", ""), "faculty"),
    ]:
        if not body:
            continue

        subject = f"{base_subject} | {'Prep Brief' if label == 'student' else 'Curriculum Reco'}"
        draft_id = f"{label}_{uuid.uuid4().hex[:8]}"
        path = _save_draft(draft_id, recipient, subject, body)

        if not state.get("dry_run", True):
            try:
                _send_smtp(recipient, subject, body)
                sent = True
            except Exception:
                sent = False
        else:
            sent = False

        drafts.append(
            DraftRecord(
                draft_id=draft_id,
                recipient=recipient,
                subject=subject,
                body=body,
                sent=sent,
                path=path,
            )
        )

    return {"drafts": drafts}


def approve_and_send(draft_path: str) -> bool:
    """Load a saved draft JSON and send it via SMTP. Called from CLI / MCP."""
    data = json.loads(Path(draft_path).read_text(encoding="utf-8"))
    try:
        _send_smtp(data["to"], data["subject"], data["body"])
        data["sent"] = True
        Path(draft_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        return False
