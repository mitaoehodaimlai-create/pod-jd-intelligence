"""
POD-JD Intelligence — CLI entry point.

Usage:
  python main.py run              # process new POD emails (dry_run from .env)
  python main.py run --send       # process + send drafts immediately
  python main.py list-jds         # list all parsed JDs
  python main.py list-drafts      # list pending draft emails
  python main.py approve <path>   # send a specific draft
  python main.py status           # show pipeline status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def cmd_run(send: bool = False) -> None:
    import config
    from tools.email_reader import fetch_pod_emails
    from workflow.graph import process_email

    dry_run = not send
    console.print(f"\n[bold cyan]POD-JD Intelligence[/bold cyan] — fetching emails (dry_run={dry_run})")

    emails = fetch_pod_emails(mark_seen=False)
    if not emails:
        console.print("[yellow]No new unread POD emails found.[/yellow]")
        return

    console.print(f"[green]Found {len(emails)} email(s) to process.[/green]\n")

    for idx, email in enumerate(emails, 1):
        console.rule(f"Email {idx}/{len(emails)} · {email.subject[:60]}")
        with console.status("Parsing JD..."):
            result = process_email(email, dry_run=dry_run)

        jd = result.get("jd") or {}
        if result.get("error"):
            console.print(f"[red]Error:[/red] {result['error']}")
            continue

        console.print(Panel(
            f"[bold]{jd.get('company')}[/bold] — {jd.get('role')}\n"
            f"Location: {jd.get('location')} | CTC: {jd.get('ctc')} | Deadline: {jd.get('deadline')}",
            title="Parsed JD",
            border_style="green",
        ))

        for draft in result.get("drafts", []):
            status = "[green]SENT[/green]" if draft["sent"] else "[yellow]SAVED (pending)[/yellow]"
            console.print(f"  Draft [{draft['draft_id']}] → {draft['recipient']} | {status}")
            console.print(f"  File: {draft['path']}")


def cmd_list_jds() -> None:
    import config

    files = sorted(config.OUTPUT_JDS.glob("*.json"))
    if not files:
        console.print("[yellow]No JDs found in output/jds/[/yellow]")
        return

    table = Table(title="Processed JDs", show_lines=True)
    table.add_column("File", style="dim")
    table.add_column("Company", style="bold cyan")
    table.add_column("Role")
    table.add_column("Deadline")
    table.add_column("Required Skills")

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        skills = ", ".join(data.get("required_skills", [])[:4])
        table.add_row(f.name, data.get("company","?"), data.get("role","?"), data.get("deadline","?"), skills)

    console.print(table)


def cmd_list_drafts() -> None:
    import config

    files = sorted(config.OUTPUT_DRAFTS.glob("*.json"))
    if not files:
        console.print("[yellow]No drafts found in output/drafts/[/yellow]")
        return

    table = Table(title="Draft Emails", show_lines=True)
    table.add_column("Draft ID", style="dim")
    table.add_column("Recipient", style="cyan")
    table.add_column("Subject")
    table.add_column("Sent", justify="center")
    table.add_column("Path", style="dim")

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        sent_str = "[green]Yes[/green]" if data.get("sent") else "[yellow]No[/yellow]"
        table.add_row(data["draft_id"], data["to"], data["subject"][:50], sent_str, str(f))

    console.print(table)


def cmd_approve(path: str) -> None:
    from workflow.nodes.notify import approve_and_send

    console.print(f"Sending draft: [cyan]{path}[/cyan]")
    ok = approve_and_send(path)
    if ok:
        console.print("[green]Draft sent successfully.[/green]")
    else:
        console.print("[red]Failed to send. Check SMTP credentials and logs.[/red]")


def cmd_status() -> None:
    import config

    jd_count = len(list(config.OUTPUT_JDS.glob("*.json")))
    all_drafts = list(config.OUTPUT_DRAFTS.glob("*.json"))
    sent = sum(1 for f in all_drafts if json.loads(f.read_text(encoding="utf-8")).get("sent", False))

    console.print(Panel(
        f"JDs processed    : [bold]{jd_count}[/bold]\n"
        f"Drafts total     : {len(all_drafts)}\n"
        f"Drafts sent      : [green]{sent}[/green]\n"
        f"Drafts pending   : [yellow]{len(all_drafts) - sent}[/yellow]\n"
        f"Model            : {config.LLM_MODEL}\n"
        f"Dry-run default  : {config.DRY_RUN}\n"
        f"POD senders      : {', '.join(config.POD_ALLOWED_SENDERS)}",
        title="POD-JD Intelligence · Status",
        border_style="blue",
    ))


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "run":
        cmd_run(send="--send" in args)
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
