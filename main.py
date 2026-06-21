"""
MAIN APPLICATION — POD-JD Intelligence
=======================================
Entry point. Orchestrates the full pipeline using LangGraph.

HOW IT WORKS:
  1. EmailService  → fetch unread POD emails from IMAP inbox
  2. LangGraph     → run each email through a 4-node graph:
       Node 1: parse_jd      (email_agent   — LangChain + ChatGroq)
       Node 2: student_prep  (student_agent — LangChain + ChatGroq)
       Node 3: teacher_reco  (teacher_agent — LangChain + ChatGroq)
       Node 4: notify        (draft_service — save to output/drafts/)
  3. LangSmith     → every node is traced automatically in the dashboard

MONITORING:
  Open https://smith.langchain.com and select project "pod-jd-intelligence"
  to see all traces, prompts, responses, and timings.

CLI COMMANDS:
  python main.py run              → process new POD emails (safe, drafts only)
  python main.py run --send       → process + send emails immediately via SMTP
  python main.py list-jds         → list all parsed Job Descriptions
  python main.py list-drafts      → list all saved draft emails
  python main.py approve <path>   → send one specific draft
  python main.py status           → pipeline stats + config
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# LangGraph pipeline (workflow/graph.py)
from workflow.graph import run_for_email

# Email fetching (services/email_service.py)
from services.email_service import get_pod_emails

# Draft management (services/draft_service.py)
from services import draft_service

import config

console = Console()


# ── PIPELINE ─────────────────────────────────────────────────────────────────

def run_pipeline(dry_run: bool = True):
    """
    Fetch all new POD emails and process each through the LangGraph pipeline.

    For each email:
      email → [LangGraph: parse_jd → student_prep → teacher_reco → notify]
      → output/jds/{company}.json  +  output/drafts/{student|teacher}_*.json

    Args:
        dry_run: True  = save drafts only (default, safe)
                 False = save drafts + send via SMTP immediately
    """
    mode_label = "[yellow]Dry Run — drafts only[/yellow]" if dry_run \
                 else "[green]Live — will send emails[/green]"

    console.print(Panel(
        f"[bold cyan]POD-JD Intelligence Pipeline[/bold cyan]\n"
        f"LangGraph + LangChain + Groq + LangSmith\n"
        f"Mode: {mode_label}",
        border_style="blue"
    ))

    # Step 1: Fetch emails from IMAP inbox
    emails = get_pod_emails()

    if not emails:
        console.print("[yellow]No new POD emails found. Nothing to process.[/yellow]")
        return

    results = []

    for i, email in enumerate(emails, start=1):
        console.rule(f"Email {i}/{len(emails)} — {email['subject'][:60]}")

        # Step 2: Run through LangGraph (4 nodes)
        # LangSmith traces each node call automatically.
        with console.status("Running LangGraph pipeline..."):
            result = run_for_email(email, dry_run=dry_run)

        jd = result.get("jd")

        if not jd:
            console.print(f"[red]Could not parse JD from this email.[/red]")
            continue

        # Step 3: Save JD to output/jds/
        jd_path = _save_jd(jd)

        # Step 4: Show what was saved
        console.print(Panel(
            f"[bold]{jd.get('company')}[/bold] — {jd.get('role')}\n"
            f"Location : {jd.get('location')}   CTC : {jd.get('ctc')}\n"
            f"Deadline : {jd.get('deadline')}\n"
            f"Skills   : {', '.join(jd.get('required_skills', [])[:4])}",
            title="Parsed JD",
            border_style="green",
        ))

        for draft in result.get("drafts", []):
            status = "[green]SENT[/green]" if draft.get("sent") else "[yellow]Saved (pending)[/yellow]"
            console.print(f"  Draft → {draft['recipient']} | {status}")
            console.print(f"  File  : {Path(draft['path']).name}")

        results.append(result)

    # Summary
    console.print(f"\n[bold green]Done.[/bold green] Processed {len(results)} JD(s).")
    if dry_run and results:
        console.print("[dim]Run [bold]python main.py list-drafts[/bold] to review drafts.[/dim]")
        console.print("[dim]Run [bold]python main.py approve <path>[/bold] to send one.[/dim]")


# ── CLI COMMANDS ──────────────────────────────────────────────────────────────

def cmd_list_jds():
    """Print a table of all parsed JD JSON files."""
    files = sorted(config.OUTPUT_JDS.glob("*.json"))
    if not files:
        console.print("[yellow]No JDs found in output/jds/[/yellow]")
        return

    table = Table(title="Parsed Job Descriptions", show_lines=True)
    table.add_column("#",         style="dim",       width=4)
    table.add_column("Company",   style="bold cyan")
    table.add_column("Role")
    table.add_column("CTC")
    table.add_column("Deadline")
    table.add_column("Top Skills", style="dim")

    for i, f in enumerate(files, 1):
        d = json.loads(f.read_text(encoding="utf-8"))
        table.add_row(
            str(i),
            d.get("company", "?"),
            d.get("role", "?"),
            d.get("ctc", "?"),
            d.get("deadline", "?"),
            ", ".join(d.get("required_skills", [])[:3]),
        )
    console.print(table)


def cmd_list_drafts():
    """Print a table of all saved draft emails."""
    drafts = draft_service.list_all_drafts()
    if not drafts:
        console.print("[yellow]No draft emails in output/drafts/[/yellow]")
        return

    table = Table(title="Draft Emails", show_lines=True)
    table.add_column("Draft ID",   style="dim")
    table.add_column("Recipient",  style="cyan")
    table.add_column("Subject")
    table.add_column("Sent?",      justify="center")
    table.add_column("Filename",   style="dim")

    for d in drafts:
        sent = "[green]Yes[/green]" if d["sent"] else "[yellow]Pending[/yellow]"
        subj = d["subject"]
        table.add_row(
            d["draft_id"],
            d["recipient"],
            subj[:55] + "…" if len(subj) > 55 else subj,
            sent,
            Path(d["path"]).name,
        )
    console.print(table)


def cmd_approve(draft_path: str):
    """Send one specific draft via SMTP."""
    console.print(f"Sending: [cyan]{draft_path}[/cyan]")
    ok = draft_service.send_approved_draft(draft_path)
    if ok:
        console.print("[green]✓ Email sent.[/green]")
    else:
        console.print("[red]✗ Send failed — check SMTP credentials in .env[/red]")


def cmd_status():
    """Print pipeline stats and current config."""
    all_drafts  = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count  = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )
    tracing_on = config.LANGCHAIN_TRACING_V2.lower() == "true"

    console.print(Panel(
        f"JDs Parsed             : [bold]{len(list(config.OUTPUT_JDS.glob('*.json')))}[/bold]\n"
        f"Drafts Total           : {len(all_drafts)}\n"
        f"Drafts Sent            : [green]{sent_count}[/green]\n"
        f"Drafts Pending         : [yellow]{len(all_drafts) - sent_count}[/yellow]\n\n"
        f"LLM                    : {config.LLM_MODEL}  (Groq via LangChain)\n"
        f"LangSmith Tracing      : {'[green]ON[/green]' if tracing_on else '[dim]off[/dim]'}\n"
        f"LangSmith Project      : {config.LANGCHAIN_PROJECT}\n"
        f"Dry Run Default        : {config.DRY_RUN}\n"
        f"POD Allowed Senders    : {', '.join(config.POD_ALLOWED_SENDERS)}",
        title="Pipeline Status",
        border_style="blue",
    ))


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _save_jd(jd: dict) -> str:
    """Save a parsed JD dict as a JSON file in output/jds/."""
    company   = jd.get("company", "unknown").replace(" ", "_").lower()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path      = config.OUTPUT_JDS / f"{company}_{timestamp}.json"
    path.write_text(json.dumps(jd, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ JD saved → {path.name}")
    return str(path)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] == "run":
        run_pipeline(dry_run=("--send" not in args))

    elif args[0] == "list-jds":
        cmd_list_jds()

    elif args[0] == "list-drafts":
        cmd_list_drafts()

    elif args[0] == "approve" and len(args) > 1:
        cmd_approve(args[1])

    elif args[0] == "status":
        cmd_status()

    else:
        console.print(__doc__)


if __name__ == "__main__":
    main()
