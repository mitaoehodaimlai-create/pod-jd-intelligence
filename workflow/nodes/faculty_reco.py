"""Faculty-Curriculum Agent — syllabus & pedagogy recommendations (Groq + LangSmith traced)."""
from __future__ import annotations

import json

from langsmith import traceable

from tools.llm import chat
from workflow.state import PipelineState

_SYSTEM = """\
You are an academic curriculum advisor for a B.Tech CSE (AI & ML) program at
MITAOE (Maharashtra Institute of Technology, Aurangabad), an autonomous college
affiliated to Savitribai Phule Pune University (SPPU), India.

Given a structured Job Description JSON, write a faculty-facing curriculum
recommendation brief in clean Markdown. Be specific and actionable.

Sections to include:
1. **Industry Demand Summary** — what this JD signals about current market needs
2. **Curriculum Gaps** — required skills in the JD not adequately covered in the
   standard AIML syllabus (name the exact course/unit that is missing or weak)
3. **Suggested Syllabus Updates** — for each gap: which existing unit to update,
   which topic to insert, and the relevant CO-PO/PSO mapping (use CO1–CO6, PO1–PO12
   numbering per NBA/NAAC norms)
4. **Practical / Lab Approaches** — specific hands-on exercises, tool-based
   assignments, or mini-projects (name the tool, dataset, and learning outcome)
5. **Study Material & References** — textbooks, NPTEL course links, Coursera/edX
   specialisations, GitHub repos worth assigning
6. **1-2 Aligned Semester Project Ideas** — full project title + 3-line scope
7. **Quick-Win Actions** — things a faculty member can add to class THIS semester
   without waiting for formal syllabus revision approval

Be specific to AIML engineering education in India. Avoid vague academic advice.
"""


@traceable(name="faculty-curriculum-agent", run_type="chain")
def faculty_reco_node(state: PipelineState) -> dict:
    if not state.get("jd"):
        return {"faculty_reco": "", "error": state.get("error")}

    reco = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"JD:\n{json.dumps(state['jd'], indent=2)}"},
        ],
        max_tokens=3000,
        temperature=0.4,
    )
    return {"faculty_reco": reco}
