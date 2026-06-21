"""
AGENT 1 — EMAIL AGENT
======================
This agent is the FIRST step in the pipeline.

What it does:
  1. Reads unread emails from the POD (Placement Office) inbox
  2. Skips emails that are NOT from trusted POD senders
  3. Downloads PDF attachments from each valid email
  4. Extracts text from those PDFs
  5. Asks Groq AI (Llama 3.3) to read the email + PDF and
     pull out a clean, structured Job Description

Output:
  A list of JD dicts, one per email, e.g.:
  {
    "company":         "TCS",
    "role":            "Software Engineer",
    "location":        "Pune",
    "eligibility":     "CGPA >= 6.5, CSE/AIML, 2025 batch",
    "required_skills": ["Python", "SQL", "Data Structures"],
    "tools_tech":      ["Git", "Linux"],
    "responsibilities":["Write backend code", "Code reviews"],
    "nice_to_have":    ["Docker", "AWS"],
    "ctc":             "3.5 LPA",
    "deadline":        "30 June 2025",
    "source_email_id": "<msg001@mail.com>"
  }

Usage:
    from agents.email_agent import run
    jd_list = run()
"""

import json
import re

from groq import Groq

import config
from services.email_service import get_pod_emails
from services.pdf_service import extract_text_from_pdf

# ------------------------------------------------------------------
# PUBLIC FUNCTION  (called from main.py and mcp_server.py)
# ------------------------------------------------------------------

def run():
    """
    Fetch all new POD emails and return a list of parsed JD dicts.

    Steps inside this function:
      1. Call email_service to get unread POD emails
      2. For each email, extract PDF text
      3. Send combined content to Groq AI for structured JD extraction
      4. Collect and return all results

    Returns:
        List of JD dicts (empty list if no new emails)
    """
    print("\n=== EMAIL AGENT: Checking POD inbox ===")

    # Step 1 — Get emails (only from trusted POD senders, with JD keywords)
    emails = get_pod_emails()

    if not emails:
        print("No new POD emails to process.")
        return []

    parsed_jds = []

    for i, email in enumerate(emails, start=1):
        print(f"\n[{i}/{len(emails)}] Subject: {email['subject']}")
        print(f"       From:    {email['sender']}")

        # Step 2 — Extract text from each PDF attachment
        pdf_texts = _extract_all_pdfs(email["attachments"])

        # Step 3 — Build one combined text block (email + PDFs) for the AI
        combined_text = _combine_content(email, pdf_texts)

        # Step 4 — Ask Groq AI to parse the JD
        jd = _parse_jd_with_groq(combined_text, email["message_id"])

        if jd:
            parsed_jds.append(jd)
            print(f"       ✓ Parsed: {jd.get('company')} — {jd.get('role')}")
        else:
            print(f"       ✗ Could not parse JD from this email")

    print(f"\n=== EMAIL AGENT: Done. Parsed {len(parsed_jds)} JD(s). ===")
    return parsed_jds


# ------------------------------------------------------------------
# PRIVATE HELPERS  (used only inside this file)
# ------------------------------------------------------------------

def _extract_all_pdfs(attachments):
    """
    Extract text from every PDF attachment in the email.

    Args:
        attachments: dict of { "filename.pdf": <bytes>, ... }

    Returns:
        List of strings, one per PDF that had readable text
    """
    pdf_texts = []

    for filename, pdf_bytes in attachments.items():
        print(f"       Reading PDF: {filename}")
        text = extract_text_from_pdf(pdf_bytes)

        if text:
            # Tag the text so the AI knows where it came from
            pdf_texts.append(f"--- Content from PDF: {filename} ---\n{text}")
            print(f"       ✓ Extracted {len(text)} characters from {filename}")
        else:
            print(f"       ✗ Could not extract text from {filename}")

    return pdf_texts


def _combine_content(email, pdf_texts):
    """
    Join the email subject, body, and PDF text into one big string.
    This combined text is what we send to the AI.

    The 12000 character limit prevents hitting Groq token limits.
    """
    parts = [
        f"EMAIL SUBJECT: {email['subject']}",
        f"\nEMAIL BODY:\n{email['body']}",
    ]

    if pdf_texts:
        parts.append("\n\nPDF ATTACHMENTS:\n" + "\n\n".join(pdf_texts))

    combined = "\n".join(parts)

    # Trim to safe length
    return combined[:12000]


def _parse_jd_with_groq(combined_text, message_id):
    """
    Send the combined email + PDF text to Groq AI and ask it to
    extract the Job Description fields as structured JSON.

    Args:
        combined_text: email body + PDF text merged together
        message_id:    the email's Message-ID header (for traceability)

    Returns:
        Dict with JD fields, or None if AI parsing failed
    """

    # This is the instruction we give to the AI (called "system prompt")
    system_prompt = """You are a Job Description parser for an academic placement office in India.

Read the email and PDF content and extract ONLY the job information.
Return a valid JSON object with EXACTLY these keys (no markdown, no explanation):

{
  "company":         "name of the company",
  "role":            "job title / position",
  "location":        "city, or Remote",
  "eligibility":     "CGPA cutoff, allowed branches, year of passing",
  "required_skills": ["list", "of", "required", "skills"],
  "tools_tech":      ["tools", "and", "technologies", "mentioned"],
  "responsibilities":["what", "the", "employee", "will", "do"],
  "nice_to_have":    ["optional", "preferred", "skills"],
  "ctc":             "salary or stipend package",
  "deadline":        "last date to apply"
}

Rules:
- Use "Not mentioned" for any string field that is not in the text
- Use [] for any list field that is not in the text
- Do NOT add any extra fields beyond the ones listed above
"""

    try:
        client = Groq(api_key=config.GROQ_API_KEY)

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": combined_text},
            ],
            max_tokens=1000,
            temperature=0.1,                           # Low = more consistent
            response_format={"type": "json_object"},   # Force JSON output
        )

        # Parse the AI's JSON response
        raw_json = response.choices[0].message.content
        # Remove markdown fences if the AI added them accidentally
        raw_json = re.sub(r"```json|```", "", raw_json).strip()

        jd = json.loads(raw_json)

        # Store which email this JD came from (for reference)
        jd["source_email_id"] = message_id

        return jd

    except Exception as e:
        print(f"       Groq API error: {e}")
        return None
