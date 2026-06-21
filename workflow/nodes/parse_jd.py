"""JD Parser Agent — structured extraction via Groq LLM (LangSmith traced)."""
from __future__ import annotations

import json
import re

from langsmith import traceable

from tools.llm import chat
from workflow.state import JDStructured, PipelineState

_SYSTEM = """\
You are a Job Description (JD) parser for an academic placement cell.
Extract the following fields from the combined email body and attached PDF text.
Return ONLY a valid JSON object with EXACTLY these keys — no markdown, no explanation:

{
  "company": "string",
  "role": "string",
  "location": "string",
  "eligibility": "string (CGPA cutoff, branches, year of passing, etc.)",
  "required_skills": ["list of required skills"],
  "tools_tech": ["tools and technologies explicitly mentioned"],
  "responsibilities": ["list of job responsibilities"],
  "nice_to_have": ["optional or preferred skills"],
  "ctc": "string (CTC / stipend or 'Not mentioned')",
  "deadline": "string (application deadline or 'Not mentioned')"
}

If a field is not in the text, use "Not mentioned" for strings and [] for arrays.
"""


@traceable(name="parse-jd-agent", run_type="chain")
def parse_jd_node(state: PipelineState) -> dict:
    combined = f"EMAIL SUBJECT: {state['subject']}\n\nEMAIL BODY:\n{state['email_body']}\n\n"
    for chunk in state.get("pdf_texts", []):
        combined += f"{chunk}\n\n"

    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": combined[:12000]},
        ],
        max_tokens=1500,
        json_mode=True,
        temperature=0.1,
    )

    # Strip accidental markdown fences
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed: JDStructured = json.loads(raw)
        parsed["source_email_id"] = state["message_id"]
        return {"jd": parsed, "error": None}
    except json.JSONDecodeError as exc:
        return {"jd": None, "error": f"JD parse failed: {exc} | raw={raw[:300]}"}
