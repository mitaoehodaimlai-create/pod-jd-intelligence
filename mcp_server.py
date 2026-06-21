"""
FastMCP server — exposes the POD-JD Intelligence pipeline as Claude Code tools.

Run:
  python mcp_server.py

Then add to .claude/mcp_servers.json:
  {
    "pod-jd": {
      "command": "python",
      "args": ["/path/to/pod_jd_intelligence/mcp_server.py"],
      "env": { "PYTHONPATH": "/path/to/pod_jd_intelligence" }
    }
  }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
import config
from tools.email_reader import fetch_pod_emails
from workflow.graph import process_email
from workflow.nodes.notify import approve_and_send

mcp = FastMCP("pod-jd-intelligence")


@mcp.tool()
def run_pipeline(dry_run: bool = True) -> str:
    """
    Fetch unread POD emails, parse JDs, generate student prep + faculty
    curriculum briefs, and save email drafts locally.

    Args:
        dry_run: When True (default) drafts are saved but NOT sent.
                 Set False to dispatch immediately via SMTP.

    Returns:
        JSON summary of processed JDs and draft file paths.
    """
    emails = fetch_pod_emails(mark_seen=False)
    if not emails:
        return json.dumps({"status": "no_new_emails", "processed": []})

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
                    {"draft_id": d["draft_id"], "recipient": d["recipient"], "path": d["path"]}
                    for d in result.get("drafts", [])
                ],
            }
        )

    return json.dumps({"status": "ok", "dry_run": dry_run, "processed": summary}, indent=2)


@mcp.tool()
def list_processed_jds() -> str:
    """
    List all previously parsed JD JSON files in output/jds/.

    Returns:
        JSON array of {filename, company, role, deadline, path}.
    """
    jd_files = sorted(config.OUTPUT_JDS.glob("*.json"))
    results = []
    for f in jd_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(
                {
                    "filename": f.name,
                    "company": data.get("company", "?"),
                    "role": data.get("role", "?"),
                    "deadline": data.get("deadline", "?"),
                    "path": str(f),
                }
            )
        except Exception:
            pass
    return json.dumps(results, indent=2)


@mcp.tool()
def list_pending_drafts() -> str:
    """
    List draft emails that have not yet been sent (sent=false).

    Returns:
        JSON array of {draft_id, recipient, subject, sent, path}.
    """
    draft_files = sorted(config.OUTPUT_DRAFTS.glob("*.json"))
    pending = []
    for f in draft_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not data.get("sent", False):
                pending.append(
                    {
                        "draft_id": data["draft_id"],
                        "recipient": data["to"],
                        "subject": data["subject"],
                        "sent": False,
                        "path": str(f),
                    }
                )
        except Exception:
            pass
    return json.dumps(pending, indent=2)


@mcp.tool()
def approve_draft(draft_path: str) -> str:
    """
    Send an approved draft email via SMTP.

    Args:
        draft_path: Full path to the draft JSON file (from list_pending_drafts).

    Returns:
        JSON with {success, draft_path}.
    """
    ok = approve_and_send(draft_path)
    return json.dumps({"success": ok, "draft_path": draft_path})


@mcp.tool()
def get_pipeline_status() -> str:
    """
    Return a quick status summary: counts of JDs processed and drafts pending.

    Returns:
        JSON with counts and config info.
    """
    jd_count = len(list(config.OUTPUT_JDS.glob("*.json")))
    all_drafts = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )
    return json.dumps(
        {
            "jds_processed": jd_count,
            "drafts_total": len(all_drafts),
            "drafts_sent": sent,
            "drafts_pending": len(all_drafts) - sent,
            "dry_run_default": config.DRY_RUN,
            "pod_senders": list(config.POD_ALLOWED_SENDERS),
            "llm_model": config.LLM_MODEL,
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run()
