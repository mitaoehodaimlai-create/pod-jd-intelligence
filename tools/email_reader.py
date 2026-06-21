"""IMAP email reader — fetches unread POD emails and their attachments."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from imap_tools import MailBox, AND, MailMessage

import config


@dataclass
class RawEmail:
    uid: str
    message_id: str
    sender: str
    subject: str
    body: str
    attachment_bytes: dict[str, bytes] = field(default_factory=dict)  # filename → bytes


def _subject_has_keyword(subject: str) -> bool:
    lower = subject.lower()
    return any(kw in lower for kw in config.JD_SUBJECT_KEYWORDS)


def fetch_pod_emails(mark_seen: bool = False) -> list[RawEmail]:
    """Return unread emails from allowed POD senders that match JD keywords."""
    results: list[RawEmail] = []

    with MailBox(config.IMAP_HOST, port=config.IMAP_PORT).login(
        config.IMAP_USER, config.IMAP_PASSWORD, initial_folder=config.IMAP_FOLDER
    ) as mb:
        for msg in mb.fetch(AND(seen=False), mark_seen=mark_seen):
            sender_addr = _extract_addr(msg.from_)
            if sender_addr not in config.POD_ALLOWED_SENDERS:
                continue
            if not _subject_has_keyword(msg.subject):
                continue

            pdf_attachments: dict[str, bytes] = {
                att.filename: att.payload
                for att in msg.attachments
                if att.filename.lower().endswith(".pdf")
            }

            results.append(
                RawEmail(
                    uid=msg.uid,
                    message_id=msg.headers.get("message-id", [msg.uid])[0],
                    sender=sender_addr,
                    subject=msg.subject,
                    body=msg.text or msg.html or "",
                    attachment_bytes=pdf_attachments,
                )
            )

    return results


def _extract_addr(raw: str) -> str:
    match = re.search(r"<([^>]+)>", raw)
    return (match.group(1) if match else raw).strip().lower()
