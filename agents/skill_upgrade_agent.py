"""
AGENT 4 — SKILL UPGRADE AGENT
==============================
Analyses the gap between what a JD requires and what AIML B.Tech students
typically know. Produces a prioritised upgrade roadmap with resources.

RAG context makes this significantly richer:
  Without RAG: gap analysis for this JD alone
  With RAG:    "Python appears in 5/5 similar JDs → highest priority upgrade"
               "Docker missing from curriculum AND recurring in JDs → flag to HOD"
"""

import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

import config


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model       = config.LLM_MODEL,
        api_key     = config.GROQ_API_KEY,
        temperature = 0.3,
    )


SKILL_UPGRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a skills gap analyst and learning coach for B.Tech AIML students
at MIT Academy of Engineering, Pune (MITAOE).

Your task: Compare what the Job Description needs against what a typical final-year
B.Tech AIML student knows from their curriculum, and create a prioritised upgrade plan.

MITAOE AIML Curriculum typically covers:
  Core: Python, OOP, DSA (basics), DBMS, OS, CN
  AI/ML: Machine Learning, Deep Learning, NLP basics, Computer Vision basics
  Tools: Jupyter, NumPy, Pandas, Scikit-learn, basic TensorFlow/Keras
  Math: Linear Algebra, Calculus, Probability & Statistics
  Gaps usually found: Docker, cloud (AWS/GCP/Azure), MLOps, production deployment,
    advanced DSA (LeetCode level), system design, API development, Git workflows,
    Spark/big-data, advanced SQL, CI/CD

If RAG context from similar past JDs is provided, use it to:
  - Elevate skills that appear in MULTIPLE similar JDs (semester-wide pattern)
  - Mark skills as "Critical" if they appear in 3+ similar JDs
  - Flag recurring curriculum gaps to faculty

Write the output in Markdown with EXACTLY these sections:

## Skill Gap Summary
| Skill | JD Requires | Curriculum Coverage | Gap Level |
(Table: Gap Level = None / Partial / Significant / Critical)

## Priority Upgrade Plan
Rank skills from most → least important to learn before the interview.
For each skill:
  **[Rank]. [Skill Name]** — [why it matters for this role]
  - Current gap: [what students know vs what JD needs]
  - Time to learn: [realistic estimate, e.g., "3 days for basics / 1 week for proficiency"]
  - Where to learn: [specific resource — platform + exact course/topic]
  - Practice: [specific task to prove competency]

## Quick Wins (can learn in < 3 days)
List 3-5 skills the student can upgrade fast before the interview.

## Long-Term Recommendations
Skills that need 2+ weeks — add these to your semester plan even if this JD is over.

## Red Flags
Skills the JD heavily emphasises that are NOT in the B.Tech AIML curriculum at all.
These should be flagged to faculty for curriculum updates.

Rules:
- Be honest about gaps — do not soften criticism of curriculum coverage
- Give SPECIFIC resources (e.g., "LeetCode Blind 75 list" not just "LeetCode")
- Time estimates should be realistic for a focused student
- Keep recommendations actionable, not theoretical""",
    ),
    ("human", "{jd_content}"),
])


@traceable(name="skill-upgrade-agent", run_type="chain")
def run(jd: dict, rag_context: str = "") -> str:
    """
    Generate a skill gap analysis and upgrade roadmap for the given JD.

    Args:
        jd:          parsed JD dict from email_agent
        rag_context: similar past JDs from RAG store (empty = no history yet)

    Returns:
        Markdown string with gap table + upgrade plan.
        Returns "" on error (non-fatal — pipeline continues).
    """
    print("\n=== SKILL UPGRADE AGENT: Analysing skill gaps ===")
    print(f"    Company : {jd.get('company')}  |  Role : {jd.get('role')}")
    if rag_context:
        print(f"    RAG     : historical skill patterns included")

    try:
        jd_content = f"Job Description:\n{json.dumps(jd, indent=2)}"
        if rag_context:
            jd_content += f"\n\n{rag_context}"

        chain  = SKILL_UPGRADE_PROMPT | _get_llm() | StrOutputParser()
        result = chain.invoke({"jd_content": jd_content})

        print(f"    ✓ Skill upgrade plan ready ({len(result)} chars)")
        return result

    except Exception as e:
        print(f"    ✗ LLM error in skill_upgrade_agent.run: {e}")
        return ""
