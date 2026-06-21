"""
MCP SERVER — POD-JD Intelligence
==================================
This file turns the pipeline into a set of TOOLS that Claude Code can call
directly from the chat, without needing a terminal.

What is MCP?
  MCP (Model Context Protocol) lets AI assistants like Claude Code call
  external tools and functions. Think of it as a plugin system.

Tools exposed by this server:
  1. run_pipeline(dry_run)     → run the full pipeline on new POD emails
  2. list_jds()                → see all Job Descriptions that were parsed
  3. list_drafts()             → see draft emails waiting for approval
  4. approve_draft(path)       → send one approved draft via SMTP
  5. get_status()              → summary of pipeline stats

HOW TO SET UP (one-time):
  1. Make sure the server runs: python mcp_server.py
  2. Add to ~/.claude/mcp_servers.json:

     {
       "mcpServers": {
         "pod-jd": {
           "command": "python",
           "args": ["/full/path/to/pod_jd_intelligence/mcp_server.py"],
           "env": { "PYTHONPATH": "/full/path/to/pod_jd_intelligence" }
         }
       }
     }

  3. Restart Claude Code — the tools will appear automatically.

RUN:
  python mcp_server.py
"""

import json
import sys
from pathlib import Path

# Add project root to Python path so all imports work
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP

import config
from services import draft_service

# Create the MCP server with a name Claude Code will display
mcp = FastMCP("pod-jd-intelligence")


# ------------------------------------------------------------------
# TOOL 1 — Run the pipeline
# ------------------------------------------------------------------

@mcp.tool()
def run_pipeline(dry_run: bool = True) -> str:
    """
    Fetch new POD placement emails, run all three AI agents, and save
    email drafts for students and faculty.

    Args:
        dry_run: True  = save drafts locally, do NOT send (default — safe)
                 False = process + send emails immediately

    Returns:
        JSON summary of all processed JDs and their draft file paths.
    """
    # Import here to avoid circular imports at module load time
    from agents import email_agent, student_agent, teacher_agent
    from services.draft_service import save_drafts
    from main import _save_jd

    # Step 1: Email Agent reads the inbox
    jd_list = email_agent.run()

    if not jd_list:
        return json.dumps({"status": "no_new_emails", "processed": []})

    summary = []

    for jd in jd_list:
        # Step 2: Student Agent
        student_brief = student_agent.run(jd)

        # Step 3: Teacher Agent
        teacher_reco = teacher_agent.run(jd)

        # Step 4: Save JD + drafts
        jd_path = _save_jd(jd)
        drafts  = save_drafts(jd, student_brief, teacher_reco)

        # Step 5: Send immediately if not dry run
        if not dry_run:
            for draft in drafts:
                draft_service.send_approved_draft(draft["path"])

        summary.append({
            "company":  jd.get("company"),
            "role":     jd.get("role"),
            "jd_path":  jd_path,
            "drafts":   drafts,
        })

    return json.dumps({
        "status":    "ok",
        "dry_run":   dry_run,
        "processed": summary
    }, indent=2)


# ------------------------------------------------------------------
# TOOL 2 — List all parsed JDs
# ------------------------------------------------------------------

@mcp.tool()
def list_jds() -> str:
    """
    Return a list of all Job Descriptions that have been parsed so far.

    Returns:
        JSON array with company, role, skills, deadline for each JD.
    """
    files = sorted(config.OUTPUT_JDS.glob("*.json"))
    results = []

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({
                "file":             f.name,
                "company":          data.get("company"),
                "role":             data.get("role"),
                "deadline":         data.get("deadline"),
                "required_skills":  data.get("required_skills", []),
            })
        except Exception:
            pass

    return json.dumps(results, indent=2)


# ------------------------------------------------------------------
# TOOL 3 — List pending drafts
# ------------------------------------------------------------------

@mcp.tool()
def list_drafts(pending_only: bool = True) -> str:
    """
    Return a list of saved draft emails.

    Args:
        pending_only: True  = show only drafts NOT yet sent (default)
                      False = show all drafts including already sent ones

    Returns:
        JSON array with draft_id, recipient, subject, sent status, file path.
    """
    drafts = draft_service.list_all_drafts(pending_only=pending_only)
    return json.dumps(drafts, indent=2)


# ------------------------------------------------------------------
# TOOL 4 — Approve and send a draft
# ------------------------------------------------------------------

@mcp.tool()
def approve_draft(draft_path: str) -> str:
    """
    Send an approved draft email via SMTP.
    This is the only way emails get sent — nothing is auto-sent.

    Args:
        draft_path: Full file path to the draft JSON file.
                    Get this from list_drafts() → "path" field.

    Returns:
        JSON with success status and the file path.
    """
    ok = draft_service.send_approved_draft(draft_path)
    return json.dumps({
        "success":    ok,
        "draft_path": draft_path,
        "message":    "Email sent successfully." if ok else "Send failed — check SMTP config in .env"
    })


# ------------------------------------------------------------------
# TOOL 5 — Pipeline status
# ------------------------------------------------------------------

@mcp.tool()
def get_status() -> str:
    """
    Return a summary of the pipeline: how many JDs processed,
    how many drafts are pending, and current config settings.

    Returns:
        JSON object with all stats and config info.
    """
    all_drafts = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )

    return json.dumps({
        "jds_processed":    len(list(config.OUTPUT_JDS.glob("*.json"))),
        "drafts_total":     len(all_drafts),
        "drafts_sent":      sent_count,
        "drafts_pending":   len(all_drafts) - sent_count,
        "llm_model":        config.LLM_MODEL,
        "dry_run_default":  config.DRY_RUN,
        "pod_senders":      list(config.POD_ALLOWED_SENDERS),
        "student_email":    config.STUDENT_LIST_EMAIL,
        "faculty_email":    config.FACULTY_LIST_EMAIL,
    }, indent=2)


# ------------------------------------------------------------------
# START SERVER
# ------------------------------------------------------------------

if __name__ == "__main__":
    # mcp.run() starts the stdio MCP server.
    # It will wait silently for tool calls from Claude Code.
    # This is normal — it is not frozen or hanging.
    print("POD-JD MCP Server running. Waiting for Claude Code tool calls...")
    mcp.run()
