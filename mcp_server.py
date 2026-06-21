"""
MCP SERVER — POD-JD Intelligence
==================================
Exposes the LangGraph pipeline as 5 tools that Claude Code can call
directly from the chat window — no terminal needed.

Stack used inside each tool:
  LangGraph  → orchestrates the 4-node pipeline graph
  LangChain  → ChatGroq chains in each agent node
  LangSmith  → automatic tracing of every LLM call
  Groq API   → the actual LLM (Llama 3.3-70B)

Available tools:
  run_pipeline(dry_run)    → process new POD emails end-to-end
  list_jds()               → see all parsed Job Descriptions
  list_drafts()            → see draft emails pending approval
  approve_draft(path)      → send one approved draft via SMTP
  get_status()             → pipeline stats + config info

SETUP (one-time):
  Add to ~/.claude/mcp_servers.json:
  {
    "mcpServers": {
      "pod-jd": {
        "command": "python",
        "args": ["/full/path/to/pod_jd_intelligence/mcp_server.py"],
        "env": { "PYTHONPATH": "/full/path/to/pod_jd_intelligence" }
      }
    }
  }

  Then restart Claude Code — the 5 tools appear automatically.

RUN:
  python mcp_server.py
"""

import json
import sys
from pathlib import Path

# Add project root to Python path so all imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

import config
from services import draft_service

# Create the MCP server (name shown in Claude Code)
mcp = FastMCP("pod-jd-intelligence")


# ── TOOL 1 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def run_pipeline(dry_run: bool = True) -> str:
    """
    Fetch new POD placement emails and run the full LangGraph pipeline.

    Pipeline steps (all traced in LangSmith):
      1. Fetch emails via IMAP (POD senders only)
      2. parse_jd     node → email_agent  (LangChain + Groq → structured JD)
      3. student_prep node → student_agent (LangChain + Groq → prep brief)
      4. teacher_reco node → teacher_agent (LangChain + Groq → curriculum reco)
      5. notify       node → save drafts to output/drafts/

    Args:
        dry_run: True  = save drafts only, do NOT send (default — safe)
                 False = process + send emails immediately

    Returns:
        JSON summary of all JDs processed and their draft file paths.
    """
    from services.email_service import get_pod_emails
    from workflow.graph import run_for_email
    from main import _save_jd

    # Fetch unread POD emails
    emails = get_pod_emails()
    if not emails:
        return json.dumps({"status": "no_new_emails", "processed": []})

    summary = []
    for email in emails:
        # Run through the LangGraph pipeline (all 4 nodes)
        result  = run_for_email(email, dry_run=dry_run)
        jd      = result.get("jd") or {}

        # Save JD JSON to output/jds/
        if jd:
            _save_jd(jd)

        summary.append({
            "company":  jd.get("company", "?"),
            "role":     jd.get("role", "?"),
            "error":    result.get("error"),
            "drafts": [
                {
                    "draft_id":  d["draft_id"],
                    "recipient": d["recipient"],
                    "path":      d["path"],
                    "sent":      d.get("sent", False),
                }
                for d in result.get("drafts", [])
            ],
        })

    return json.dumps({
        "status":    "ok",
        "dry_run":   dry_run,
        "processed": summary,
    }, indent=2)


# ── TOOL 2 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_jds() -> str:
    """
    List all Job Descriptions that have been parsed and saved so far.

    Returns:
        JSON array — each item has: file, company, role, deadline, required_skills.
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


# ── TOOL 3 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_drafts(pending_only: bool = True) -> str:
    """
    List saved draft emails.

    Args:
        pending_only: True  = only show unsent drafts (default)
                      False = show all drafts including already sent ones

    Returns:
        JSON array — each item has: draft_id, recipient, subject, sent, path.
    """
    drafts = draft_service.list_all_drafts(pending_only=pending_only)
    return json.dumps(drafts, indent=2)


# ── TOOL 4 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def approve_draft(draft_path: str) -> str:
    """
    Send an approved draft email via SMTP.

    This is the only way emails are sent — nothing sends automatically.

    Args:
        draft_path: Full file path to the draft JSON file.
                    Get paths from list_drafts() → "path" field.

    Returns:
        JSON with success=true/false and a message.
    """
    ok = draft_service.send_approved_draft(draft_path)
    return json.dumps({
        "success":    ok,
        "draft_path": draft_path,
        "message":    "Email sent." if ok else "Send failed — check SMTP config in .env",
    })


# ── TOOL 5 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_status() -> str:
    """
    Return a summary of the pipeline's current state and configuration.

    Returns:
        JSON with JD counts, draft counts, LangSmith status, model info.
    """
    all_drafts  = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count  = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )

    return json.dumps({
        "jds_parsed":          len(list(config.OUTPUT_JDS.glob("*.json"))),
        "drafts_total":        len(all_drafts),
        "drafts_sent":         sent_count,
        "drafts_pending":      len(all_drafts) - sent_count,

        # LLM + monitoring config
        "llm_model":           config.LLM_MODEL,
        "langsmith_tracing":   config.LANGCHAIN_TRACING_V2,
        "langsmith_project":   config.LANGCHAIN_PROJECT,
        "langsmith_dashboard": "https://smith.langchain.com",

        # Pipeline config
        "dry_run_default":     config.DRY_RUN,
        "pod_senders":         list(config.POD_ALLOWED_SENDERS),
        "student_email":       config.STUDENT_LIST_EMAIL,
        "faculty_email":       config.FACULTY_LIST_EMAIL,
    }, indent=2)


# ── START SERVER ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("POD-JD MCP Server starting...")
    print(f"  LLM       : {config.LLM_MODEL} via LangChain + Groq")
    print(f"  LangSmith : {'ON — ' + config.LANGCHAIN_PROJECT if config.LANGCHAIN_TRACING_V2 == 'true' else 'off'}")
    print("Waiting for Claude Code tool calls (stdio)...\n")
    mcp.run()
