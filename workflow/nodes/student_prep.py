"""Student-Prep Agent — interview prep brief (Groq + LangSmith traced)."""
from __future__ import annotations

import json

from langsmith import traceable

from tools.llm import chat
from workflow.state import PipelineState

_SYSTEM = """\
You are a career counsellor and technical interview coach at MITAOE for the
B.Tech CSE (Artificial Intelligence & Machine Learning) program (7th/8th semester).

Given a structured Job Description JSON, write a student-facing preparation brief
in clean Markdown. Be specific and actionable — no generic advice.

Sections to include:
1. **Role & Company Overview** (2-3 lines)
2. **Must-Know Topics** — one bullet per required skill with key sub-topics
3. **Skill Gaps vs CSE-AIML Curriculum** — what's in the JD but typically NOT yet
   taught in the B.Tech program (flag honestly)
4. **2-Week Prep Plan** — day-by-day schedule (label each day, specific task)
5. **Practice Resources** — platform name + what to practise there (LeetCode, Kaggle,
   HuggingFace, NPTEL, etc.)
6. **10 Likely Interview Questions** — mix of technical, ML theory, and HR
7. **Projects to Showcase** — one concrete project idea per major skill area

Tone: direct, motivating, practical. Targeted at engineering students in India.
"""


@traceable(name="student-prep-agent", run_type="chain")
def student_prep_node(state: PipelineState) -> dict:
    if not state.get("jd"):
        return {"student_brief": "", "error": state.get("error")}

    brief = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"JD:\n{json.dumps(state['jd'], indent=2)}"},
        ],
        max_tokens=3000,
        temperature=0.4,
    )
    return {"student_brief": brief}
