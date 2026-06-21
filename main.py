"""
MAIN APPLICATION — POD-JD Intelligence
=======================================

HOW THE PIPELINE WORKS (step by step):

  Step 1  → email_service  reads unread emails from POD inbox (IMAP)
  Step 2  → pdf_service    extracts text from PDF attachments
  Step 3  → email_agent    sends email + PDF text to Groq AI → gets structured JD
  Step 4  → student_agent  sends JD to Groq AI → gets student interview prep brief
  Step 5  → teacher_agent  sends JD to Groq AI → gets faculty curriculum recommendations
  Step 6  → draft_service  saves both outputs as local draft files (NOT auto-sent)
  Step 7  → human approval → run  python main.py approve <path>  to send

PIPELINE FLOW (no LangGraph needed — simple sequence of function calls):

  POD Inbox (IMAP)
       ↓
  email_service.get_pod_emails()
       ↓   list of emails
  FOR EACH EMAIL:
       ↓
  pdf_service.extract_text_from_pdf()   ← reads PDF attachments
       ↓   combined text
  email_agent.parse_jd()                ← LangChain + Groq → structured JD dict
       ↓   jd dict
  student_agent.run(jd)                 ← LangChain + Groq → Markdown brief
  teacher_agent.run(jd)                 ← LangChain + Groq → Markdown reco
       ↓   two text outputs
  draft_service.save_drafts()           ← saves to output/drafts/*.json
       ↓
  output/jds/   &   output/drafts/      ← local files

ALL LLM CALLS ARE TRACED IN LANGSMITH:
  Open https://smith.langchain.com → project "pod-jd-intelligence"

CLI COMMANDS:
  python main.py run              → process new emails (safe, drafts only)
  python main.py run --send       → process + send emails immediately
  python main.py list-jds         → show all parsed JDs
  python main.py list-drafts      → show all saved drafts
  python main.py approve <path>   → send one specific draft
  python main.py status           → pipeline stats
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Agents (do the AI work using LangChain + Groq)
from agents import email_agent, student_agent, teacher_agent

# Services (handle email, PDF, and file operations)
from services.email_service import get_pod_emails
from services.pdf_service import extract_text_from_pdf
from services import draft_service

import config

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE  — core logic, called by `python main.py run`
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(dry_run: bool = True):
    """
    Run the full pipeline for all new POD emails.

    No LangGraph needed — just a simple for-loop calling each agent in order.

    Args:
        dry_run: True  = save drafts, do NOT send email (default — safe)
                 False = save drafts AND send via SMTP immediately
    """
    mode = "[yellow]Dry Run (drafts only)[/yellow]" if dry_run \
           else "[green]Live (emails will be sent)[/green]"

    console.print(Panel(
        f"[bold cyan]POD-JD Intelligence[/bold cyan]\n"
        f"LangChain + Groq + LangSmith\n"
        f"Mode: {mode}",
        border_style="blue"
    ))

    # ── STEP 1: Fetch emails ──────────────────────────────────────────────────
    # Reads only unread emails from trusted POD senders with JD keywords
    emails = get_pod_emails()

    if not emails:
        console.print("[yellow]No new POD emails found.[/yellow]")
        return

    for i, email in enumerate(emails, start=1):
        console.rule(f"Email {i}/{len(emails)} — {email['subject'][:65]}")

        # ── STEP 2: Extract PDF text ──────────────────────────────────────────
        pdf_texts = []
        for filename, pdf_bytes in email["attachments"].items():
            text = extract_text_from_pdf(pdf_bytes)
            if text:
                pdf_texts.append(f"--- PDF: {filename} ---\n{text}")
                console.print(f"  PDF: extracted text from [cyan]{filename}[/cyan]")

        # Build one combined text block (email body + all PDF text)
        combined = f"Subject: {email['subject']}\n\nEmail Body:\n{email['body']}"
        if pdf_texts:
            combined += "\n\n" + "\n\n".join(pdf_texts)

        # ── STEP 3: Agent 1 — Parse JD ───────────────────────────────────────
        # Sends combined text to Groq AI via LangChain → returns structured dict
        jd = email_agent.parse_jd(combined, email["message_id"])

        if not jd:
            console.print("[red]Could not parse JD from this email. Skipping.[/red]")
            continue

        # Save parsed JD as JSON file
        jd_path = _save_jd(jd)

        # Show what was parsed
        console.print(Panel(
            f"[bold]{jd.get('company')}[/bold] — {jd.get('role')}\n"
            f"Location : {jd.get('location')}     CTC : {jd.get('ctc')}\n"
            f"Deadline : {jd.get('deadline')}\n"
            f"Skills   : {', '.join(jd.get('required_skills', [])[:4])}",
            title="Parsed JD",
            border_style="green",
        ))

        # ── STEP 4: Agent 2 — Student Prep Brief ─────────────────────────────
        # Sends JD to Groq AI → returns Markdown interview prep guide
        with console.status("Student Agent generating prep brief..."):
            student_brief = student_agent.run(jd)

        # ── STEP 5: Agent 3 — Teacher Curriculum Reco ────────────────────────
        # Sends JD to Groq AI → returns Markdown curriculum recommendations
        with console.status("Teacher Agent generating curriculum recommendations..."):
            teacher_reco = teacher_agent.run(jd)

        # ── STEP 6: Save draft emails ─────────────────────────────────────────
        # Saves to output/drafts/ — nothing is sent yet
        drafts = draft_service.save_drafts(jd, student_brief, teacher_reco)

        # ── STEP 7: Send (only if not dry run) ───────────────────────────────
        if not dry_run:
            for draft in drafts:
                draft_service.send_approved_draft(draft["path"])

        # Show draft locations
        for draft in drafts:
            status = "[green]SENT[/green]" if draft.get("sent") \
                     else "[yellow]Saved — pending approval[/yellow]"
            console.print(f"  Draft → {draft['recipient']} | {status}")
            console.print(f"  File  : [dim]{Path(draft['path']).name}[/dim]")

    if dry_run:
        console.print(
            "\n[dim]Use [bold]python main.py list-drafts[/bold] to see drafts."
            "  Use [bold]python main.py approve <path>[/bold] to send one.[/dim]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

def cmd_list_jds():
    """Show a table of all saved JD JSON files."""
    files = sorted(config.OUTPUT_JDS.glob("*.json"))
    if not files:
        console.print("[yellow]No JDs found in output/jds/[/yellow]")
        return

    table = Table(title="Parsed Job Descriptions", show_lines=True)
    table.add_column("#",        style="dim", width=4)
    table.add_column("Company",  style="bold cyan")
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
    """Show a table of all saved draft emails."""
    drafts = draft_service.list_all_drafts()
    if not drafts:
        console.print("[yellow]No drafts found in output/drafts/[/yellow]")
        return

    table = Table(title="Draft Emails", show_lines=True)
    table.add_column("Draft ID",  style="dim")
    table.add_column("Recipient", style="cyan")
    table.add_column("Subject")
    table.add_column("Sent?",     justify="center")
    table.add_column("Filename",  style="dim")

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
        console.print("[red]✗ Failed — check SMTP credentials in .env[/red]")


def cmd_status():
    """Show pipeline stats and current config."""
    all_drafts = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent_count = sum(
        1 for f in all_drafts
        if json.loads(f.read_text(encoding="utf-8")).get("sent", False)
    )
    tracing = config.LANGCHAIN_TRACING_V2.lower() == "true"

    console.print(Panel(
        f"JDs Parsed          : [bold]{len(list(config.OUTPUT_JDS.glob('*.json')))}[/bold]\n"
        f"Drafts Total        : {len(all_drafts)}\n"
        f"Drafts Sent         : [green]{sent_count}[/green]\n"
        f"Drafts Pending      : [yellow]{len(all_drafts) - sent_count}[/yellow]\n\n"
        f"LLM Model           : {config.LLM_MODEL}  (via LangChain + Groq)\n"
        f"LangSmith Tracing   : {'[green]ON[/green] — ' + config.LANGCHAIN_PROJECT if tracing else '[dim]off[/dim]'}\n"
        f"Dry Run Default     : {config.DRY_RUN}\n"
        f"POD Senders         : {', '.join(config.POD_ALLOWED_SENDERS)}",
        title="Pipeline Status",
        border_style="blue",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save_jd(jd: dict) -> str:
    """Save a parsed JD dict as a JSON file in output/jds/."""
    company   = jd.get("company", "unknown").replace(" ", "_").lower()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path      = config.OUTPUT_JDS / f"{company}_{timestamp}.json"
    path.write_text(json.dumps(jd, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"  ✓ JD saved → [dim]{path.name}[/dim]")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

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
