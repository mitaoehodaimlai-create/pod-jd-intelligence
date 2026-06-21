"""
DRAFT SERVICE
=============
Handles saving and sending notification emails.

IMPORTANT: Emails are NEVER sent automatically.
  - The pipeline saves drafts as JSON files in output/drafts/
  - A human must run `python main.py approve <path>` to actually send

This prevents accidental bulk emails to students/faculty.

Draft JSON format:
{
    "draft_id":  "student_a1b2c3",
    "to":        "aiml-students@mitaoe.ac.in",
    "subject":   "[Placement] TCS – Software Engineer | Prep Brief",
    "body":      "... full email text ...",
    "sent":      false
}
"""

import json
import uuid
from pathlib import Path
from datetime import datetime

import config
from services.email_service import send_email


# ------------------------------------------------------------------
# SAVE  (always runs — no emails sent here)
# ------------------------------------------------------------------

def save_drafts(jd, student_brief, teacher_reco):
    """
    Save two draft emails to output/drafts/ as JSON files.
    Returns a list of draft info dicts so main.py can print them.

    Args:
        jd:            parsed JD dict (company, role, etc.)
        student_brief: text output from the Student Agent
        teacher_reco:  text output from the Teacher Agent
    """
    company = jd.get("company", "Company")
    role    = jd.get("role", "Role")

    # Base subject prefix so recipients know this came from the POD pipeline
    base_subject = f"[Placement] {company} – {role}"

    saved_drafts = []

    # --- Draft 1: Student notification ---
    student_draft = _create_draft(
        to      = config.STUDENT_LIST_EMAIL,
        subject = f"{base_subject} | Interview Prep Brief",
        body    = student_brief,
        label   = "student"
    )
    saved_drafts.append(student_draft)

    # --- Draft 2: Faculty notification ---
    teacher_draft = _create_draft(
        to      = config.FACULTY_LIST_EMAIL,
        subject = f"{base_subject} | Curriculum Recommendation",
        body    = teacher_reco,
        label   = "teacher"
    )
    saved_drafts.append(teacher_draft)

    return saved_drafts


# ------------------------------------------------------------------
# SEND  (only called when human runs `approve` command)
# ------------------------------------------------------------------

def send_approved_draft(draft_path):
    """
    Load a saved draft JSON file and send it via SMTP.

    Args:
        draft_path: full path to the draft .json file

    Returns:
        True if sent successfully, False otherwise.
    """
    path = Path(draft_path)

    if not path.exists():
        print(f"Draft file not found: {draft_path}")
        return False

    # Load the draft
    draft = json.loads(path.read_text(encoding="utf-8"))

    if draft.get("sent"):
        print(f"This draft was already sent. Skipping.")
        return False

    try:
        # Actually send the email
        send_email(draft["to"], draft["subject"], draft["body"])

        # Mark as sent and save back
        draft["sent"] = True
        draft["sent_at"] = datetime.utcnow().isoformat()
        path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")

        return True

    except Exception as e:
        print(f"Failed to send draft: {e}")
        return False


# ------------------------------------------------------------------
# LIST  (used by main.py and mcp_server.py)
# ------------------------------------------------------------------

def list_all_drafts(pending_only=False):
    """
    Return a list of all saved draft info dicts.

    Args:
        pending_only: if True, return only drafts that haven't been sent
    """
    drafts = []
    for f in sorted(config.OUTPUT_DRAFTS.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if pending_only and data.get("sent", False):
                continue
            drafts.append({
                "draft_id":  data["draft_id"],
                "recipient": data["to"],
                "subject":   data["subject"],
                "sent":      data.get("sent", False),
                "path":      str(f),
            })
        except Exception:
            pass
    return drafts


# ------------------------------------------------------------------
# PRIVATE HELPER
# ------------------------------------------------------------------

def _create_draft(to, subject, body, label):
    """
    Build a draft dict, save it to output/drafts/, and return its info.
    """
    # Unique ID: label + 8 random hex chars, e.g. "student_3f8a1b2c"
    draft_id = f"{label}_{uuid.uuid4().hex[:8]}"
    file_path = config.OUTPUT_DRAFTS / f"{draft_id}.json"

    draft_data = {
        "draft_id": draft_id,
        "to":       to,
        "subject":  subject,
        "body":     body,
        "sent":     False,
        "created":  datetime.utcnow().isoformat(),
    }

    # Save to file
    file_path.write_text(
        json.dumps(draft_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"  ✓ Draft saved → {file_path.name}  (recipient: {to})")

    return {
        "draft_id":  draft_id,
        "recipient": to,
        "subject":   subject,
        "path":      str(file_path),
    }
