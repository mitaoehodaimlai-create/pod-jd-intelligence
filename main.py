"""
MAIN APPLICATION — POD-JD Intelligence
=======================================
This is the entry point of the entire project.

HOW THE PIPELINE WORKS (in simple terms):
  Step 1:  Email Agent  → reads POD emails, extracts PDF text, parses JD with AI
  Step 2:  Student Agent → reads the JD, generates an interview prep guide for students
  Step 3:  Teacher Agent → reads the JD, generates curriculum update tips for faculty
  Step 4:  Draft Service → saves both outputs as email drafts (NOT auto-sent)
  Step 5:  Human approves → run `python main.py approve <path>` to actually send

PIPELINE FLOW:
  POD Email (IMAP)
       ↓
  [Email Agent]  ← reads email + PDF, parses JD using Groq AI
       ↓ JD dict
    ┌──┴──┐
    ↓     ↓
[Student  [Teacher       ← both agents run one after the other
 Agent]    Agent]
    ↓     ↓
    └──┬──┘
       ↓
  [Draft Service]  ← saves two draft email files locally
       ↓
  output/drafts/   ← student_*.json  and  teacher_*.json
       ↓
  (human approval) ← `python main.py approve <path>`
       ↓
  SMTP Send        ← email goes out only after explicit approval

CLI COMMANDS:
  python main.py run              → process new POD emails (drafts only)
  python main.py run --send       → process + auto-send (use only after testing)
  python main.py list-jds         → show all parsed Job Descriptions
  python main.py list-drafts      → show all saved draft emails
  python main.py approve <path>   → send one specific draft
  python main.py status           → show counts and config summary
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Our three AI agents
from agents import email_agent, student_agent, teacher_agent

# Services handle email + file operations
from services import draft_service

import config

console = Console()


# ------------------------------------------------------------------
# PIPELINE  (the core logic — runs all three agents in sequence)
# ------------------------------------------------------------------

def run_pipeline(dry_run=True):
    """
    Run the full pipeline for all new POD emails.

    Args:
        dry_run: True  = save drafts only, do NOT send (default, safe)
                 False = save drafts AND send immediately via SMTP
    """
    console.print(Panel(
        f"[bold cyan]POD-JD Intelligence Pipeline[/bold cyan]\n"
        f"Mode: {'[yellow]Dry Run (drafts only)[/yellow]' if dry_run else '[green]Live (will send emails)[/green]'}",
        border_style="blue"
    ))

    # ── STEP 1: Email Agent ──────────────────────────────────────
    # Reads POD inbox, extracts PDFs, parses JDs using Groq AI
    jd_list = email_agent.run()

    if not jd_list:
        console.print("[yellow]No new JDs to process. Exiting.[/yellow]")
        return

    results = []

    # Process each JD one by one
    for jd in jd_list:
        console.rule(f"[bold]{jd.get('company', '?')} — {jd.get('role', '?')}[/bold]")

        # ── STEP 2: Student Agent ────────────────────────────────
        # Generates interview prep brief for students
        student_brief = student_agent.run(jd)

        # ── STEP 3: Teacher Agent ────────────────────────────────
        # Generates curriculum update recommendations for faculty
        teacher_reco = teacher_agent.run(jd)

        # ── STEP 4: Save JD to output/jds/ ──────────────────────
        jd_path = _save_jd(jd)

        # ── STEP 5: Save drafts ──────────────────────────────────
        # Both drafts are saved as JSON files — never sent automatically
        console.print("\n[bold]Saving email drafts...[/bold]")
        drafts = draft_service.save_drafts(jd, student_brief, teacher_reco)

        # ── STEP 6: Send (only if not dry run) ───────────────────
        if not dry_run:
            for draft in drafts:
                console.print(f"Sending to {draft['recipient']}...")
                draft_service.send_approved_draft(draft["path"])

        results.append({"jd": jd, "drafts": drafts, "jd_path": jd_path})

    # Print summary table
    _print_summary(results, dry_run)


# ------------------------------------------------------------------
# CLI COMMANDS
# ------------------------------------------------------------------

def cmd_list_jds():
    """Show all previously parsed JDs."""
    files = sorted(config.OUTPUT_JDS.glob("*.json"))
    if not files:
        console.print("[yellow]No JDs found in output/jds/[/yellow]")
        return

    table = Table(title="Parsed Job Descriptions", show_lines=True)
    table.add_column("#",          style="dim",   width=4)
    table.add_column("Company",    style="bold cyan")
    table.add_column("Role")
    table.add_column("CTC")
    table.add_column("Deadline")
    table.add_column("Top Skills")

    for i, f in enumerate(files, 1):
        data = json.loads(f.read_text(encoding="utf-8"))
        skills_preview = ", ".join(data.get("required_skills", [])[:3])
        table.add_row(
            str(i),
            data.get("company", "?"),
            data.get("role", "?"),
            data.get("ctc", "?"),
            data.get("deadline", "?"),
            skills_preview,
        )

    console.print(table)


def cmd_list_drafts():
    """Show all saved draft emails with their send status."""
    drafts = draft_service.list_all_drafts()
    if not drafts:
        console.print("[yellow]No draft emails found in output/drafts/[/yellow]")
        return

    table = Table(title="Draft Emails", show_lines=True)
    table.add_column("Draft ID",  style="dim")
    table.add_column("Recipient", style="cyan")
    table.add_column("Subject")
    table.add_column("Sent?",     justify="center")
    table.add_column("File Path", style="dim")

    for d in drafts:
        sent_label = "[green]Yes[/green]" if d["sent"] else "[yellow]Pending[/yellow]"
        table.add_row(
            d["draft_id"],
            d["recipient"],
            d["subject"][:55] + "..." if len(d["subject"]) > 55 else d["subject"],
            sent_label,
            Path(d["path"]).name,
        )

    console.print(table)


def cmd_approve(draft_path):
    """Send one specific draft email via SMTP."""
    console.print(f"Sending draft: [cyan]{draft_path}[/cyan]")
    ok = draft_service.send_approved_draft(draft_path)
    if ok:
        console.print("[green]✓ Email sent successfully.[/green]")
    else:
        console.print("[red]✗ Send failed. Check SMTP credentials in .env[/red]")


def cmd_status():
    """Show pipeline stats and current configuration."""
    jd_count    = len(list(config.OUTPUT_JDS.glob("*.json")))
    all_drafts  = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count  = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )

    console.print(Panel(
        f"JDs Parsed           : [bold]{jd_count}[/bold]\n"
        f"Draft Emails Total   : {len(all_drafts)}\n"
        f"Drafts Sent          : [green]{sent_count}[/green]\n"
        f"Drafts Pending       : [yellow]{len(all_drafts) - sent_count}[/yellow]\n\n"
        f"LLM Model            : {config.LLM_MODEL}\n"
        f"Dry Run Default      : {config.DRY_RUN}\n"
        f"POD Allowed Senders  : {', '.join(config.POD_ALLOWED_SENDERS)}\n"
        f"Student Email List   : {config.STUDENT_LIST_EMAIL}\n"
        f"Faculty Email List   : {config.FACULTY_LIST_EMAIL}",
        title="Pipeline Status",
        border_style="blue",
    ))


# ------------------------------------------------------------------
# HELPERS (private)
# ------------------------------------------------------------------

def _save_jd(jd):
    """Save a parsed JD dict to output/jds/ as a JSON file."""
    company = jd.get("company", "unknown").replace(" ", "_").lower()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = config.OUTPUT_JDS / f"{company}_{timestamp}.json"
    path.write_text(json.dumps(jd, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ JD saved → {path.name}")
    return str(path)


def _print_summary(results, dry_run):
    """Print a final summary table after all emails are processed."""
    console.print("\n")
    table = Table(title="Pipeline Summary", show_lines=True)
    table.add_column("Company",       style="bold cyan")
    table.add_column("Role")
    table.add_column("Drafts Saved",  justify="center")
    table.add_column("Status")

    for r in results:
        status = "[yellow]Drafts saved (pending approval)[/yellow]" if dry_run \
                 else "[green]Emails sent[/green]"
        table.add_row(
            r["jd"].get("company", "?"),
            r["jd"].get("role", "?"),
            str(len(r["drafts"])),
            status,
        )

    console.print(table)
    if dry_run:
        console.print("\n[dim]Run [bold]python main.py list-drafts[/bold] to see drafts."
                      "  Run [bold]python main.py approve <path>[/bold] to send one.[/dim]")


# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args or args[0] == "run":
        # `run` command: use --send flag to override dry_run
        send_now = "--send" in args
        dry_run  = not send_now
        run_pipeline(dry_run=dry_run)

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
