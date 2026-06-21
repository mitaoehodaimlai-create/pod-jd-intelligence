"""
AGENT 1 — EMAIL AGENT
======================
Reads POD emails, extracts PDF text, and parses the Job Description
using a LangChain + ChatGroq chain.

Libraries used:
  langchain-groq   → ChatGroq LLM (calls Groq API through LangChain)
  langchain-core   → ChatPromptTemplate (builds the prompt), JsonOutputParser
  langsmith        → @traceable (records this step in LangSmith dashboard)

LangSmith tracing:
  Every call to parse_jd() appears as a traced run in LangSmith under
  the project name from LANGCHAIN_PROJECT in your .env file.
  View traces at: https://smith.langchain.com

Exposed functions:
  run()            → fetch all new POD emails + return list of parsed JDs
  parse_jd()       → parse ONE combined text block into a structured JD dict
                     (also called from workflow/graph.py as a node)
"""

import json
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable

import config
from services.email_service import get_pod_emails
from services.pdf_service import extract_text_from_pdf


# ── LLM SETUP ────────────────────────────────────────────────────────────────

def _get_llm():
    """
    Create and return a ChatGroq LLM instance configured for JD parsing.
    temperature=0.1 → very deterministic, consistent JSON output.
    """
    return ChatGroq(
        model    = config.LLM_MODEL,
        api_key  = config.GROQ_API_KEY,
        temperature = 0.1,
        model_kwargs = {"response_format": {"type": "json_object"}},  # Force JSON
    )


# ── JD PARSING PROMPT ────────────────────────────────────────────────────────

# This template is sent to the LLM.
# {content} is replaced with the actual email+PDF text at runtime.
JD_PARSER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Job Description (JD) parser for an academic placement office.

Read the email subject, body, and any PDF content provided.
Extract the job information and return ONLY a valid JSON object with these exact keys:

{{
  "company":          "name of the hiring company",
  "role":             "job title / position name",
  "location":         "city or Remote",
  "eligibility":      "CGPA cutoff, allowed branches, year of passing",
  "required_skills":  ["skill1", "skill2", "skill3"],
  "tools_tech":       ["tool1", "technology2"],
  "responsibilities": ["what the employee will do"],
  "nice_to_have":     ["optional preferred skills"],
  "ctc":              "salary or stipend package",
  "deadline":         "last date to apply"
}}

Rules:
- Use "Not mentioned" for any string field not found in the text
- Use [] for any list field not found in the text
- Do NOT add any extra keys beyond the ones listed above
- Return only the JSON — no explanation, no markdown fences""",
    ),
    ("human", "{content}"),
])


# ── PUBLIC FUNCTIONS ─────────────────────────────────────────────────────────

@traceable(name="email-agent-run", run_type="chain")
def run():
    """
    Main function: fetch all new POD emails and return parsed JDs.

    Steps:
      1. Connect to inbox via IMAP and get unread POD emails
      2. Extract text from PDF attachments
      3. Call parse_jd() for each email to get structured JD
      4. Return list of all successfully parsed JDs

    Returns:
        List of JD dicts (empty list if no new emails or all failed)
    """
    print("\n=== EMAIL AGENT: Checking POD inbox ===")

    # Step 1: Get emails from trusted POD senders only
    emails = get_pod_emails()

    if not emails:
        print("No new POD emails found.")
        return []

    parsed_jds = []

    for i, email in enumerate(emails, start=1):
        print(f"\n[{i}/{len(emails)}] Subject : {email['subject']}")
        print(f"             From    : {email['sender']}")

        # Step 2: Extract PDF attachment text
        pdf_texts = []
        for filename, pdf_bytes in email["attachments"].items():
            text = extract_text_from_pdf(pdf_bytes)
            if text:
                pdf_texts.append(f"--- PDF: {filename} ---\n{text}")
                print(f"             PDF     : extracted {len(text)} chars from {filename}")

        # Step 3: Combine everything into one text block for the AI
        combined = (
            f"Subject: {email['subject']}\n\n"
            f"Email Body:\n{email['body']}"
        )
        if pdf_texts:
            combined += "\n\n" + "\n\n".join(pdf_texts)

        # Step 4: Parse JD with LangChain + Groq
        jd = parse_jd(combined, email["message_id"])

        if jd:
            parsed_jds.append(jd)
            print(f"             ✓ Parsed : {jd.get('company')} — {jd.get('role')}")
        else:
            print(f"             ✗ Failed to parse JD from this email")

    print(f"\n=== EMAIL AGENT: Done. {len(parsed_jds)} JD(s) parsed. ===")
    return parsed_jds


@traceable(name="parse-jd", run_type="llm")
def parse_jd(combined_text: str, message_id: str) -> dict | None:
    """
    Parse a combined email+PDF text block into a structured JD dict
    using a LangChain chain (prompt → ChatGroq → JsonOutputParser).

    This function is also called directly from workflow/graph.py as a
    LangGraph node step.

    Args:
        combined_text: email subject + body + PDF text merged together
        message_id:    the email's Message-ID (stored for reference)

    Returns:
        Dict with JD fields, or None if parsing failed
    """
    try:
        # Build the LangChain chain:
        #   Prompt template → ChatGroq LLM → JSON output parser
        # LangSmith traces every step of this chain automatically.
        chain = JD_PARSER_PROMPT | _get_llm() | JsonOutputParser()

        # Trim to safe length (avoids hitting Groq token limits)
        jd = chain.invoke({"content": combined_text[:12000]})

        # Attach the source email ID for traceability
        jd["source_email_id"] = message_id

        return jd

    except Exception as e:
        print(f"  Groq/LangChain error in parse_jd: {e}")
        return None
