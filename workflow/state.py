"""LangGraph pipeline state definition."""
from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict


class JDStructured(TypedDict):
    company: str
    role: str
    location: str
    eligibility: str
    required_skills: list[str]
    tools_tech: list[str]
    responsibilities: list[str]
    nice_to_have: list[str]
    ctc: str
    deadline: str
    source_email_id: str


class DraftRecord(TypedDict):
    draft_id: str
    recipient: str
    subject: str
    body: str
    sent: bool
    path: str


class PipelineState(TypedDict):
    # email being processed
    email_uid: str
    message_id: str
    sender: str
    subject: str
    email_body: str
    pdf_texts: list[str]

    # parsed JD
    jd: Optional[JDStructured]

    # agent outputs
    student_brief: str
    faculty_reco: str

    # drafts
    drafts: list[DraftRecord]

    # control
    dry_run: bool
    error: Optional[str]
