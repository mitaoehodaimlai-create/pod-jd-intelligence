"""
EMAIL SERVICE
=============
Handles low-level email operations:
  - Connecting to the inbox via IMAP (read emails)
  - Sending approved drafts via SMTP
  - Filtering emails so only trusted POD senders are processed

This file does NOT do any AI work — it is purely about email I/O.
"""

import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from imap_tools import MailBox, AND

import config


# ------------------------------------------------------------------
# READ  (IMAP)
# ------------------------------------------------------------------

def get_pod_emails():
    """
    Connect to the inbox and return unread emails that came from
    a trusted POD sender AND have a placement-related subject.

    Returns a list of dicts, each looking like:
    {
        "uid":         "123",
        "message_id":  "<abc@mail.com>",
        "sender":      "placement@mitaoe.ac.in",
        "subject":     "Campus Drive – TCS 2025",
        "body":        "Dear students...",
        "attachments": { "jd.pdf": <bytes>, ... }   ← only PDF files
    }
    """
    emails = []

    print(f"Connecting to {config.IMAP_HOST} as {config.IMAP_USER} ...")

    with MailBox(config.IMAP_HOST, port=config.IMAP_PORT).login(
        config.IMAP_USER,
        config.IMAP_PASSWORD,
        initial_folder=config.IMAP_FOLDER
    ) as mailbox:

        # Fetch all unread messages (we filter them below)
        for msg in mailbox.fetch(AND(seen=False), mark_seen=False):

            sender_addr = _extract_email_address(msg.from_)

            # GUARD: skip if sender is not in the trusted POD list
            if sender_addr not in config.POD_ALLOWED_SENDERS:
                continue

            # GUARD: skip if subject does not contain placement keywords
            if not _has_jd_keyword(msg.subject):
                continue

            # Collect only PDF attachments (ignore images, docs, etc.)
            pdf_attachments = {
                att.filename: att.payload
                for att in msg.attachments
                if att.filename.lower().endswith(".pdf")
            }

            emails.append({
                "uid":         msg.uid,
                "message_id":  msg.headers.get("message-id", [msg.uid])[0],
                "sender":      sender_addr,
                "subject":     msg.subject,
                "body":        msg.text or msg.html or "",
                "attachments": pdf_attachments,
            })

    print(f"Found {len(emails)} unread POD email(s).")
    return emails


# ------------------------------------------------------------------
# SEND  (SMTP)
# ------------------------------------------------------------------

def send_email(to_address, subject, body_text):
    """
    Send a plain-text email via SMTP.
    Called only when a draft is explicitly approved by the user.

    Args:
        to_address: recipient email address
        subject:    email subject line
        body_text:  plain text body
    """
    # Build the email message
    msg = MIMEMultipart("alternative")
    msg["From"]    = config.SMTP_USER
    msg["To"]      = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    # Connect and send
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.ehlo()
        server.starttls()          # Encrypt the connection
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.sendmail(config.SMTP_USER, [to_address], msg.as_string())

    print(f"  ✓ Email sent to {to_address}")


# ------------------------------------------------------------------
# HELPERS  (private, used only inside this file)
# ------------------------------------------------------------------

def _extract_email_address(raw_from):
    """
    Pull the bare email address from a 'From' header string.
    Example: "POD Office <placement@mitaoe.ac.in>" → "placement@mitaoe.ac.in"
    """
    match = re.search(r"<([^>]+)>", raw_from)
    return (match.group(1) if match else raw_from).strip().lower()


def _has_jd_keyword(subject):
    """
    Return True if the email subject contains at least one
    placement-related keyword (defined in config).
    """
    subject_lower = subject.lower()
    return any(keyword in subject_lower for keyword in config.JD_SUBJECT_KEYWORDS)
