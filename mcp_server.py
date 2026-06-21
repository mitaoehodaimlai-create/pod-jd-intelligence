"""
MCP SERVER — POD-JD Intelligence
==================================
Exposes the pipeline as 5 tools that Claude Code can call directly
from the chat — no terminal needed.

No LangGraph — agents are called directly in sequence:
  email_agent.parse_jd()  →  student_agent.run()  →  teacher_agent.run()
  →  draft_service.save_drafts()

All LLM calls go through LangChain + Groq and are traced in LangSmith.

TOOLS:
  run_pipeline(dry_run)   → fetch POD emails, run all agents, save drafts
  list_jds()              → show all parsed Job Descriptions
  list_drafts()           → show drafts waiting for approval
  approve_draft(path)     → send one approved draft via SMTP
  get_status()            → pipeline stats and config

SETUP (one-time, add to ~/.claude/mcp_servers.json):
  {
    "mcpServers": {
      "pod-jd": {
        "command": "python",
        "args": ["/full/path/to/pod_jd_intelligence/mcp_server.py"],
        "env": { "PYTHONPATH": "/full/path/to/pod_jd_intelligence" }
      }
    }
  }

RUN:
  python mcp_server.py
"""

import json
import sys
from pathlib import Path

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

import config
from services import draft_service

mcp = FastMCP("pod-jd-intelligence")


# ── TOOL 1: Run the full pipeline ────────────────────────────────────────────

@mcp.tool()
def run_pipeline(dry_run: bool = True) -> str:
    """
    Fetch new POD emails and run all three agents in sequence:
      1. email_agent.parse_jd()  → structured Job Description
      2. student_agent.run()     → student interview prep brief
      3. teacher_agent.run()     → faculty curriculum recommendations
      4. draft_service           → save draft emails to output/drafts/

    All LLM calls are traced in LangSmith automatically.

    Args:
        dry_run: True  = save drafts only, do NOT send (default — safe)
                 False = save drafts AND send via SMTP

    Returns:
        JSON summary of all processed JDs and draft file paths.
    """
    from services.email_service import get_pod_emails
    from services.pdf_service import extract_text_from_pdf
    from agents import email_agent, student_agent, teacher_agent
    from main import _save_jd

    # RAG functions are defined inline in main.py (no separate rag/ package)
    from main import _RAG_AVAILABLE as _rag_ok, rag_add_jd, rag_get_similar_jds, rag_format_context

    # Step 1: Fetch unread POD emails
    emails = get_pod_emails()
    if not emails:
        return json.dumps({"status": "no_new_emails", "processed": []})

    summary = []

    for email in emails:
        # Step 2: Extract PDF attachment text
        pdf_texts = []
        for filename, pdf_bytes in email["attachments"].items():
            text = extract_text_from_pdf(pdf_bytes)
            if text:
                pdf_texts.append(f"--- PDF: {filename} ---\n{text}")

        # Step 3: Build combined text (email + PDFs)
        combined = f"Subject: {email['subject']}\n\nEmail Body:\n{email['body']}"
        if pdf_texts:
            combined += "\n\n" + "\n\n".join(pdf_texts)

        # Step 4: Agent 1 — Parse JD (LangChain + Groq)
        jd = email_agent.parse_jd(combined, email["message_id"])
        if not jd:
            continue

        _save_jd(jd)

        # Step 4b: RAG — store JD + retrieve similar past JDs for context
        # Stored JDs are embedded as vectors in ChromaDB and retrieved by
        # semantic similarity so agents can spot semester-wide skill trends.
        rag_context = ""
        if _rag_ok:
            rag_add_jd(jd)
            similar     = rag_get_similar_jds(jd)
            rag_context = rag_format_context(similar)

        # Step 5: Agent 2 — Student brief (LangChain + Groq + RAG context)
        student_brief = student_agent.run(jd, rag_context=rag_context)

        # Step 6: Agent 3 — Teacher reco (LangChain + Groq + RAG context)
        teacher_reco = teacher_agent.run(jd, rag_context=rag_context)

        # Step 7: Save drafts
        drafts = draft_service.save_drafts(jd, student_brief, teacher_reco)

        # Send immediately if not dry run
        if not dry_run:
            for draft in drafts:
                draft_service.send_approved_draft(draft["path"])

        summary.append({
            "company": jd.get("company"),
            "role":    jd.get("role"),
            "drafts": [
                {"draft_id": d["draft_id"], "recipient": d["recipient"], "path": d["path"]}
                for d in drafts
            ],
        })

    return json.dumps({"status": "ok", "dry_run": dry_run, "processed": summary}, indent=2)


# ── TOOL 2: List all parsed JDs ──────────────────────────────────────────────

@mcp.tool()
def list_jds() -> str:
    """
    List all Job Descriptions that have been parsed and saved in output/jds/.

    Returns:
        JSON array — file, company, role, deadline, required_skills.
    """
    files   = sorted(config.OUTPUT_JDS.glob("*.json"))
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "file":            f.name,
                "company":         data.get("company"),
                "role":            data.get("role"),
                "deadline":        data.get("deadline"),
                "required_skills": data.get("required_skills", []),
            })
        except Exception:
            pass
    return json.dumps(results, indent=2)


# ── TOOL 3: List draft emails ─────────────────────────────────────────────────

@mcp.tool()
def list_drafts(pending_only: bool = True) -> str:
    """
    List saved draft emails.

    Args:
        pending_only: True = only unsent drafts (default), False = all drafts

    Returns:
        JSON array — draft_id, recipient, subject, sent, path.
    """
    return json.dumps(draft_service.list_all_drafts(pending_only=pending_only), indent=2)


# ── TOOL 4: Approve and send a draft ─────────────────────────────────────────

@mcp.tool()
def approve_draft(draft_path: str) -> str:
    """
    Send one approved draft email via SMTP.
    Nothing is ever sent automatically — this is the only send path.

    Args:
        draft_path: Full path to the draft JSON file (from list_drafts → "path").

    Returns:
        JSON with success status and message.
    """
    ok = draft_service.send_approved_draft(draft_path)
    return json.dumps({
        "success":    ok,
        "draft_path": draft_path,
        "message":    "Email sent." if ok else "Send failed — check SMTP in .env",
    })


# ── TOOL 5: Pipeline status ───────────────────────────────────────────────────

@mcp.tool()
def get_status() -> str:
    """
    Return pipeline stats and current configuration.

    Returns:
        JSON with JD count, draft counts, model info, LangSmith status.
    """
    all_drafts = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )
    return json.dumps({
        "jds_parsed":         len(list(config.OUTPUT_JDS.glob("*.json"))),
        "drafts_total":       len(all_drafts),
        "drafts_sent":        sent_count,
        "drafts_pending":     len(all_drafts) - sent_count,
        "llm_model":          config.LLM_MODEL,
        "langsmith_tracing":  config.LANGCHAIN_TRACING_V2,
        "langsmith_project":  config.LANGCHAIN_PROJECT,
        "dry_run_default":    config.DRY_RUN,
        "pod_senders":        list(config.POD_ALLOWED_SENDERS),
    }, indent=2)


# ── START ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("POD-JD MCP Server running...")
    print(f"  Model     : {config.LLM_MODEL}")
    print(f"  LangSmith : {'ON — ' + config.LANGCHAIN_PROJECT if config.LANGCHAIN_TRACING_V2.lower() == 'true' else 'off'}")
    print("Waiting for Claude Code tool calls (stdio)...\n")
    mcp.run()
