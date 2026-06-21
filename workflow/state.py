"""
LANGGRAPH PIPELINE STATE
========================
This file defines the "state" that flows through the LangGraph pipeline.

Think of the state like a shared whiteboard:
  - Each node (agent) READS what it needs from the whiteboard
  - Each node WRITES its output back to the whiteboard
  - The next node picks up from where the previous one left off

State flow through the graph:
  ┌─────────────────────────────────────────────┐
  │  START                                      │
  │    email_uid, message_id, sender,           │  ← filled before graph starts
  │    subject, email_body, pdf_texts,          │
  │    dry_run                                  │
  │                                             │
  │  After parse_jd node:                       │
  │    + jd  (structured Job Description dict)  │  ← filled by email_agent
  │                                             │
  │  After student_prep node:                   │
  │    + student_brief  (markdown text)         │  ← filled by student_agent
  │                                             │
  │  After teacher_reco node:                   │
  │    + teacher_reco  (markdown text)          │  ← filled by teacher_agent
  │                                             │
  │  After notify node:                         │
  │    + drafts  (list of saved draft files)    │  ← filled by draft_service
  │                                             │
  │  END                                        │
  └─────────────────────────────────────────────┘
"""

from typing import Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    """
    All data that moves through the LangGraph pipeline for ONE email.

    Fields marked with (input) are set before the graph runs.
    Fields marked with (output) are filled in by the nodes as they run.
    """

    # ── Email data (input) ──────────────────────────────────────
    email_uid:   str          # unique ID of the email in the inbox
    message_id:  str          # Message-ID header (used for deduplication)
    sender:      str          # sender's email address
    subject:     str          # email subject line
    email_body:  str          # plain text body of the email
    pdf_texts:   list[str]    # extracted text from each PDF attachment

    # ── Parsed Job Description (filled by email_agent node) ─────
    jd: Optional[dict]        # structured JD dict (company, role, skills, etc.)

    # ── Agent outputs (filled by student/teacher agent nodes) ────
    student_brief: str        # interview prep guide for students
    teacher_reco:  str        # curriculum recommendations for faculty

    # ── Notification results (filled by notify node) ─────────────
    drafts: list[dict]        # list of saved draft info dicts

    # ── Control flags ────────────────────────────────────────────
    dry_run: bool             # True = save drafts only, False = send emails
    error:   Optional[str]    # error message if something went wrong
