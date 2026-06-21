"""LangGraph StateGraph — POD-JD Intelligence pipeline."""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from langgraph.graph import StateGraph, START, END

import config
from tools.email_reader import RawEmail
from workflow.nodes.ingest import ingest_node
from workflow.nodes.parse_jd import parse_jd_node
from workflow.nodes.student_prep import student_prep_node
from workflow.nodes.faculty_reco import faculty_reco_node
from workflow.nodes.notify import notify_node
from workflow.state import PipelineState


# ── node wrappers (LangGraph nodes receive & return state dicts) ──────────────

def node_parse_jd(state: PipelineState) -> dict:
    return parse_jd_node(state)


def node_student_prep(state: PipelineState) -> dict:
    return student_prep_node(state)


def node_faculty_reco(state: PipelineState) -> dict:
    return faculty_reco_node(state)


def node_notify(state: PipelineState) -> dict:
    return notify_node(state)


# ── build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)
    g.add_node("parse_jd", node_parse_jd)
    g.add_node("student_prep", node_student_prep)
    g.add_node("faculty_reco", node_faculty_reco)
    g.add_node("notify", node_notify)

    g.add_edge(START, "parse_jd")
    g.add_edge("parse_jd", "student_prep")
    g.add_edge("student_prep", "faculty_reco")
    g.add_edge("faculty_reco", "notify")
    g.add_edge("notify", END)

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── high-level runner ─────────────────────────────────────────────────────────

def process_email(email: RawEmail, dry_run: bool = True) -> dict:
    from workflow.nodes.ingest import ingest_node as _ingest

    ingest_state = _ingest(
        {"subject": email.subject, "email_body": email.body, "pdf_texts": []},
        email.attachment_bytes,
    )

    initial_state: PipelineState = {
        "email_uid": email.uid,
        "message_id": email.message_id,
        "sender": email.sender,
        "subject": email.subject,
        "email_body": email.body,
        "pdf_texts": ingest_state["pdf_texts"],
        "jd": None,
        "student_brief": "",
        "faculty_reco": "",
        "drafts": [],
        "dry_run": dry_run,
        "error": None,
    }

    result = get_graph().invoke(initial_state)

    if result.get("jd"):
        _save_jd(result["jd"])

    return result


def _save_jd(jd: dict) -> None:
    company = jd.get("company", "unknown").replace(" ", "_").lower()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = config.OUTPUT_JDS / f"{company}_{ts}.json"
    path.write_text(json.dumps(jd, indent=2, ensure_ascii=False), encoding="utf-8")
