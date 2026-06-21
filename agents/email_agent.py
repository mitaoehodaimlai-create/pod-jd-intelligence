"""
AGENT 1 — EMAIL AGENT
======================
Reads POD emails, extracts PDF text, and uses an LLM to parse the
Job Description into a structured Python dict.

────────────────────────────────────────────────────────────────────────────
LLM (LARGE LANGUAGE MODEL) USAGE — WHY AND HOW
────────────────────────────────────────────────────────────────────────────

  WHY LLM?
    Every company writes JDs differently — different headings, different
    order, some in PDF, some in email body, some mixing Hindi/English.
    A rule-based parser (regex, keyword matching) would fail on edge cases.
    An LLM understands intent and extracts the right data regardless of format.

  WHICH LLM?
    Meta's Llama 3.3-70B via Groq API (configured in .env as LLM_MODEL).
    Groq runs the model on custom inference hardware — very fast (~1-2 sec).

  HOW (LangChain chain):
    Input text
         ↓
    ChatPromptTemplate   ← fills {content} with the actual email+PDF text,
                            formats it as a chat message list for the LLM
         ↓
    ChatGroq             ← sends the messages to Groq API over HTTPS,
                            returns the LLM's response as an AIMessage object
         ↓
    JsonOutputParser     ← reads AIMessage.content (a JSON string),
                            returns a Python dict  {company, role, skills, ...}

    The "|" operator is LangChain Expression Language (LCEL) —
    it chains Runnable objects like Unix pipes, passing output left → right.

  WHY temperature=0.1?
    For JD parsing we want deterministic, consistent JSON output.
    Low temperature = the model picks the most probable token each step.
    (Student/teacher agents use 0.4 because they benefit from some creativity.)

  WHY model_kwargs={"response_format": {"type": "json_object"}}?
    Tells Groq's API to enforce valid JSON output — no markdown fences,
    no extra explanations, just the JSON object.
    Without this, the model sometimes wraps JSON in ```json...``` blocks.

────────────────────────────────────────────────────────────────────────────
RAG — THIS AGENT'S ROLE IN THE RAG PIPELINE
────────────────────────────────────────────────────────────────────────────

  This agent produces the "documents" that fill the RAG store.
  After parse_jd() returns, main.py calls rag_store.add_jd(jd) to embed
  and store the JD in ChromaDB.

  This agent itself does NOT use RAG context (it is parsing raw text, not
  generating recommendations). Agents 2 and 3 consume the RAG context.

────────────────────────────────────────────────────────────────────────────
LANGSMITH TRACING
────────────────────────────────────────────────────────────────────────────

  @traceable on run() and parse_jd() records every LLM call to LangSmith:
    - Full prompt sent to the model
    - Full response received
    - Latency (how long Groq took)
    - Token counts (input + output)
    - Errors (if any)

  View traces: https://smith.langchain.com → project "pod-jd-intelligence"
  This lets you debug bad parses by seeing exactly what the LLM received.
"""

import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable

import config
from services.email_service import get_pod_emails
from services.pdf_service import extract_text_from_pdf


# ── LLM CONFIGURATION ────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    """
    Create a ChatGroq LLM instance tuned for structured JSON extraction.

    ChatGroq is a LangChain wrapper that:
      1. Takes your prompt (as a list of messages)
      2. Sends it to Groq's API (api.groq.com)
      3. Returns the response as an AIMessage object

    temperature=0.1 → near-deterministic output (good for parsing JSON)
    response_format="json_object" → Groq enforces strict JSON, no extra text
    """
    return ChatGroq(
        model        = config.LLM_MODEL,
        api_key      = config.GROQ_API_KEY,
        temperature  = 0.1,
        model_kwargs = {"response_format": {"type": "json_object"}},
    )


# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────
# ChatPromptTemplate defines the conversation structure sent to the LLM.
# It has a "system" role (instructions to the model) and "human" (the input text).
# {content} is a placeholder filled at runtime via chain.invoke({"content": ...}).

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
    # {content} will be replaced with: email subject + body + PDF text
    ("human", "{content}"),
])


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────────────

@traceable(name="email-agent-run", run_type="chain")
def run() -> list[dict]:
    """
    Main function: fetch all new POD emails and return list of parsed JDs.

    Steps:
      1. IMAP: read unread emails from trusted POD senders
      2. PDF:  extract text from attachments (pdfplumber → pypdf fallback)
      3. LLM:  parse combined text into a structured JD dict (see parse_jd)
      4. Return all successfully parsed JDs

    @traceable records this entire function in LangSmith as one trace.
    """
    print("\n=== EMAIL AGENT: Checking POD inbox ===")

    emails = get_pod_emails()
    if not emails:
        print("No new POD emails found.")
        return []

    parsed_jds = []

    for i, email in enumerate(emails, start=1):
        print(f"\n[{i}/{len(emails)}] Subject : {email['subject']}")
        print(f"             From    : {email['sender']}")

        # Extract text from every PDF attachment
        pdf_texts = []
        for filename, pdf_bytes in email["attachments"].items():
            text = extract_text_from_pdf(pdf_bytes)
            if text:
                pdf_texts.append(f"--- PDF: {filename} ---\n{text}")
                print(f"             PDF     : {len(text)} chars from {filename}")

        # Merge email + PDFs into one text block for the LLM
        combined = (
            f"Subject: {email['subject']}\n\n"
            f"Email Body:\n{email['body']}"
        )
        if pdf_texts:
            combined += "\n\n" + "\n\n".join(pdf_texts)

        # Call LLM to extract structured JD
        jd = parse_jd(combined, email["message_id"])

        if jd:
            parsed_jds.append(jd)
            print(f"             ✓ Parsed : {jd.get('company')} — {jd.get('role')}")
        else:
            print("             ✗ Failed to parse JD from this email")

    print(f"\n=== EMAIL AGENT: Done. {len(parsed_jds)} JD(s) parsed. ===")
    return parsed_jds


@traceable(name="parse-jd", run_type="llm")
def parse_jd(combined_text: str, message_id: str) -> dict | None:
    """
    Use LangChain + Groq to parse raw email+PDF text into a structured JD dict.

    ── HOW THE LangChain CHAIN WORKS ──────────────────────────────────────────

      chain = JD_PARSER_PROMPT | _get_llm() | JsonOutputParser()

      The "|" is LCEL (LangChain Expression Language) — a pipe operator
      that passes the output of each step as input to the next:

        JD_PARSER_PROMPT
          → Fills {content} with combined_text
          → Returns: [SystemMessage("..."), HumanMessage(combined_text)]
               ↓
        ChatGroq (calls Groq API)
          → Sends the message list via HTTPS to api.groq.com
          → Returns: AIMessage(content='{"company": "TCS", "role": "SDE", ...}')
               ↓
        JsonOutputParser
          → Reads AIMessage.content (a JSON string)
          → Returns: {"company": "TCS", "role": "SDE", "required_skills": [...], ...}

    ── WHY TRIM TO 12000 CHARACTERS? ─────────────────────────────────────────
      Groq's Llama 3.3-70B supports a context window of ~128K tokens, but
      a 12000-char JD already exceeds what any real JD contains.
      Trimming prevents runaway cost on accidentally huge inputs (e.g., a
      200-page PDF attached by mistake).

    ── LANGSMITH TRACING ──────────────────────────────────────────────────────
      @traceable makes this appear as a nested span in the LangSmith trace tree.
      You can click on it to see exactly what prompt was sent and what JSON
      the LLM returned — useful for debugging bad parses.

    Args:
        combined_text: email subject + body + all PDF text merged together
        message_id:    the email's Message-ID header (stored for traceability)

    Returns:
        JD dict with fields: company, role, location, eligibility,
        required_skills, tools_tech, responsibilities, nice_to_have, ctc,
        deadline, source_email_id.
        Returns None if parsing fails (non-fatal — pipeline skips this email).
    """
    try:
        # Build and invoke the LangChain chain
        # LangSmith automatically traces each step (prompt, LLM call, parse).
        chain = JD_PARSER_PROMPT | _get_llm() | JsonOutputParser()

        # Trim to ~12K chars to stay well within token limits
        jd = chain.invoke({"content": combined_text[:12000]})

        # Tag with source email ID for audit trail
        jd["source_email_id"] = message_id

        return jd

    except Exception as e:
        print(f"  LLM/LangChain error in parse_jd: {e}")
        return None
