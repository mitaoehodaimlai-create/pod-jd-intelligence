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
import ssl
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from imap_tools import MailBox, AND

# Use certifi CA bundle if available (fixes macOS SSL cert verification)
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl.create_default_context()

import config


# ------------------------------------------------------------------
# READ  (IMAP)
# ------------------------------------------------------------------

def get_pod_emails():
    """
    Connect to the inbox and return unread emails that came from
    a trusted POD sender AND have a placement-related subject.

    Uses server-side FROM filter per allowed sender — only matching
    emails are downloaded, so the inbox size does not matter.

    Returns a list of dicts:
    {
        "uid":         "123",
        "message_id":  "<abc@mail.com>",
        "sender":      "placement@mitaoe.ac.in",
        "subject":     "Campus Drive – TCS 2025",
        "body":        "Dear students...",
        "attachments": { "jd.pdf": <bytes>, ... }   ← PDF files only
    }
    Returns empty list on any error — pipeline continues safely.
    """
    emails = []
    seen_uids = set()   # deduplicate if a sender appears multiple times in config

    print(f"Connecting to {config.IMAP_HOST} as {config.IMAP_USER} ...")

    # ── CONNECT ───────────────────────────────────────────────────────────────
    try:
        mailbox_conn = MailBox(
            config.IMAP_HOST,
            port=config.IMAP_PORT,
            ssl_context=_SSL_CONTEXT,
        ).login(
            config.IMAP_USER,
            config.IMAP_PASSWORD,
            initial_folder=config.IMAP_FOLDER,
        )
    except Exception as e:
        print(f"  ✗ Could not connect to mailbox: {e}")
        print(f"  Tip: check IMAP_HOST, IMAP_USER, IMAP_PASSWORD in your .env file")
        return []

    # ── FETCH — one server-side query per trusted sender ─────────────────────
    # This avoids downloading the entire inbox; only emails FROM each allowed
    # sender are fetched, regardless of how large the inbox is.
    try:
        with mailbox_conn as mailbox:
            for trusted_sender in config.POD_ALLOWED_SENDERS:
                try:
                    criteria = AND(seen=False, from_=trusted_sender)
                    for msg in mailbox.fetch(criteria, mark_seen=False):
                        if msg.uid in seen_uids:
                            continue
                        seen_uids.add(msg.uid)

                        try:
                            # Extra guard: verify sender exactly (server FROM is prefix-match)
                            sender_addr = _extract_email_address(msg.from_)
                            if sender_addr not in config.POD_ALLOWED_SENDERS:
                                continue

                            # Skip if subject has no placement keywords
                            if not _has_jd_keyword(msg.subject):
                                continue

                            # Collect only PDF attachments
                            pdf_attachments = {
                                att.filename: att.payload
                                for att in msg.attachments
                                if att.filename and att.filename.lower().endswith(".pdf")
                            }

                            emails.append({
                                "uid":         msg.uid,
                                "message_id":  msg.headers.get("message-id", [msg.uid])[0],
                                "sender":      sender_addr,
                                "subject":     msg.subject,
                                "body":        msg.text or msg.html or "",
                                "attachments": pdf_attachments,
                            })

                        except Exception as e:
                            print(f"  ✗ Error reading one email, skipping: {e}")
                            continue

                except Exception as e:
                    print(f"  ✗ Error querying for {trusted_sender}: {e}")
                    continue

    except Exception as e:
        print(f"  ✗ Error fetching emails: {e}")
        return []

    # ── RESULT ────────────────────────────────────────────────────────────────
    if not emails:
        senders = ", ".join(config.POD_ALLOWED_SENDERS)
        keywords = ", ".join(config.JD_SUBJECT_KEYWORDS)
        print(f"  No matching emails found.")
        print(f"  Waiting for unread email from: {senders}")
        print(f"  Subject must contain one of: {keywords}")
    else:
        print(f"Found {len(emails)} unread POD email(s).")

    return emails


# ------------------------------------------------------------------
# SEND  (SMTP)
# ------------------------------------------------------------------

def send_email(to_address, subject, body_text):
    """
    Send a plain-text email via SMTP (STARTTLS on port 587).
    Called only when a draft is explicitly approved by the user.
    """
    msg = MIMEMultipart("alternative")
    msg["From"]    = config.SMTP_USER
    msg["To"]      = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=_SSL_CONTEXT)
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.SMTP_USER, [to_address], msg.as_string())
        print(f"  ✓ Email sent to {to_address}")
        return True

    except smtplib.SMTPAuthenticationError:
        print(f"  ✗ SMTP login failed — check SMTP_USER and SMTP_PASSWORD in .env")
        return False
    except smtplib.SMTPException as e:
        print(f"  ✗ SMTP error sending to {to_address}: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error sending email: {e}")
        return False


# ------------------------------------------------------------------
# HELPERS  (private)
# ------------------------------------------------------------------

def _extract_email_address(raw_from):
    """'POD Office <placement@mitaoe.ac.in>' → 'placement@mitaoe.ac.in'"""
    match = re.search(r"<([^>]+)>", raw_from)
    return (match.group(1) if match else raw_from).strip().lower()


def _has_jd_keyword(subject):
    """Return True if subject contains at least one keyword from config."""
    subject_lower = subject.lower()
    return any(keyword.lower() in subject_lower for keyword in config.JD_SUBJECT_KEYWORDS)
