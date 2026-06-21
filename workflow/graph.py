"""
LANGGRAPH PIPELINE GRAPH
=========================
This file wires the three AI agents into a LangGraph StateGraph.

What is LangGraph?
  LangGraph lets you define an AI pipeline as a directed graph where:
    - Each NODE is a step (parse JD, generate student brief, etc.)
    - Each EDGE defines the order (parse → student → teacher → notify)
    - The STATE flows through every node, getting richer at each step

Graph structure:
  START
    │
    ▼
  [parse_jd]      ← email_agent: extract structured JD from email+PDF
    │
    ▼
  [student_prep]  ← student_agent: generate interview prep brief
    │
    ▼
  [teacher_reco]  ← teacher_agent: generate curriculum recommendations
    │
    ▼
  [notify]        ← draft_service: save email drafts to output/drafts/
    │
    ▼
  END

LangSmith tracing:
  When LANGCHAIN_TRACING_V2=true, every node execution is automatically
  traced and visible at https://smith.langchain.com
"""

from langgraph.graph import StateGraph, START, END

from workflow.state import PipelineState
from services import draft_service


# ── NODE FUNCTIONS ────────────────────────────────────────────────────────────
# Each node function receives the full state dict and returns a dict
# containing only the fields it updated (LangGraph merges it back).

def parse_jd_node(state: PipelineState) -> dict:
    """
    NODE 1: Parse Job Description
    Calls email_agent to extract the structured JD from email body + PDF text.
    Writes 'jd' field into state.
    """
    # Import here to avoid circular imports
    from agents.email_agent import parse_jd

    print("\n[Graph] Node: parse_jd")

    # Combine email body and all PDF text into one block
    combined = f"Subject: {state['subject']}\n\nEmail Body:\n{state['email_body']}"
    for pdf_chunk in state.get("pdf_texts", []):
        combined += f"\n\n{pdf_chunk}"

    # Call the agent function to parse the JD
    jd = parse_jd(combined, state["message_id"])

    if jd:
        print(f"[Graph] JD parsed: {jd.get('company')} — {jd.get('role')}")
        return {"jd": jd, "error": None}
    else:
        print("[Graph] Failed to parse JD.")
        return {"jd": None, "error": "JD parsing failed"}


def student_prep_node(state: PipelineState) -> dict:
    """
    NODE 2: Student Prep Brief
    Calls student_agent to generate an interview prep guide.
    Writes 'student_brief' field into state.
    Skipped if jd is None (parse failed).
    """
    from agents.student_agent import run as student_run

    print("\n[Graph] Node: student_prep")

    if not state.get("jd"):
        print("[Graph] Skipping student_prep — no JD available.")
        return {"student_brief": ""}

    brief = student_run(state["jd"])
    return {"student_brief": brief}


def teacher_reco_node(state: PipelineState) -> dict:
    """
    NODE 3: Teacher Curriculum Recommendation
    Calls teacher_agent to generate curriculum update suggestions.
    Writes 'teacher_reco' field into state.
    Skipped if jd is None.
    """
    from agents.teacher_agent import run as teacher_run

    print("\n[Graph] Node: teacher_reco")

    if not state.get("jd"):
        print("[Graph] Skipping teacher_reco — no JD available.")
        return {"teacher_reco": ""}

    reco = teacher_run(state["jd"])
    return {"teacher_reco": reco}


def notify_node(state: PipelineState) -> dict:
    """
    NODE 4: Save Notification Drafts
    Saves student brief + teacher reco as email draft files.
    Writes 'drafts' field into state.
    Sends emails immediately only if dry_run=False.
    """
    print("\n[Graph] Node: notify")

    if not state.get("jd"):
        print("[Graph] Skipping notify — no JD to notify about.")
        return {"drafts": []}

    # Save both drafts to output/drafts/
    drafts = draft_service.save_drafts(
        jd            = state["jd"],
        student_brief = state.get("student_brief", ""),
        teacher_reco  = state.get("teacher_reco", ""),
    )

    # Send immediately only if explicitly requested (not dry run)
    if not state.get("dry_run", True):
        for draft in drafts:
            draft_service.send_approved_draft(draft["path"])

    return {"drafts": drafts}


# ── BUILD GRAPH ───────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph StateGraph.
    Call this once at startup; reuse the compiled graph for all emails.
    """
    # Create a new graph that uses PipelineState as its state schema
    workflow = StateGraph(PipelineState)

    # Add each node (step) to the graph
    workflow.add_node("parse_jd",     parse_jd_node)
    workflow.add_node("student_prep", student_prep_node)
    workflow.add_node("teacher_reco", teacher_reco_node)
    workflow.add_node("notify",       notify_node)

    # Define the order of execution (edges)
    workflow.add_edge(START,          "parse_jd")      # start → parse JD
    workflow.add_edge("parse_jd",     "student_prep")  # parse → student brief
    workflow.add_edge("student_prep", "teacher_reco")  # student → teacher reco
    workflow.add_edge("teacher_reco", "notify")        # teacher → save drafts
    workflow.add_edge("notify",       END)             # save → done

    # Compile the graph (validates structure + enables LangSmith tracing)
    return workflow.compile()


# ── SINGLETON GRAPH (created once, reused) ───────────────────────────────────
_graph = None

def get_graph():
    """Return the compiled graph, creating it once if needed."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── CONVENIENCE RUNNER ────────────────────────────────────────────────────────

def run_for_email(email: dict, dry_run: bool = True) -> dict:
    """
    Run the full LangGraph pipeline for one email dict.

    Args:
        email:    dict with keys: uid, message_id, sender, subject,
                  body, attachments (from email_service.get_pod_emails)
        dry_run:  True = save drafts only, False = also send via SMTP

    Returns:
        Final state dict with jd, student_brief, teacher_reco, drafts.
    """
    from services.pdf_service import extract_text_from_pdf

    # Extract PDF text from attachments before entering the graph
    pdf_texts = []
    for filename, pdf_bytes in email.get("attachments", {}).items():
        text = extract_text_from_pdf(pdf_bytes)
        if text:
            pdf_texts.append(f"--- PDF: {filename} ---\n{text}")

    # Build the initial state for this email
    initial_state: PipelineState = {
        "email_uid":    email["uid"],
        "message_id":   email["message_id"],
        "sender":       email["sender"],
        "subject":      email["subject"],
        "email_body":   email["body"],
        "pdf_texts":    pdf_texts,
        "jd":           None,
        "student_brief": "",
        "teacher_reco": "",
        "drafts":       [],
        "dry_run":      dry_run,
        "error":        None,
    }

    # Run the graph — LangSmith traces every node automatically
    graph   = get_graph()
    result  = graph.invoke(initial_state)

    return result
