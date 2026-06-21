"""
AGENT 3 — TEACHER AGENT
========================
Generates curriculum update recommendations for faculty using a
LangChain + ChatGroq chain.

Libraries used:
  langchain-groq  → ChatGroq LLM
  langchain-core  → ChatPromptTemplate, StrOutputParser
  langsmith       → @traceable (traces this agent in LangSmith dashboard)

LangSmith tracing:
  Every call to run() is recorded as a separate trace in LangSmith.
  Faculty recommendations can be compared and reviewed over time.

Output format:
  Returns a Markdown string with these sections:
    1. Industry Demand Summary
    2. Curriculum Gaps
    3. Suggested Syllabus Updates (with CO-PO mapping)
    4. Practical Lab Approaches
    5. Study Material & References
    6. Semester Project Ideas
    7. Quick-Win Actions This Semester
"""

import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

import config


# ── LLM SETUP ────────────────────────────────────────────────────────────────

def _get_llm():
    """
    Create and return a ChatGroq LLM instance for teacher recommendations.
    temperature=0.4 → some creativity for practical suggestions.
    """
    return ChatGroq(
        model       = config.LLM_MODEL,
        api_key     = config.GROQ_API_KEY,
        temperature = 0.4,
    )


# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────

TEACHER_RECO_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an academic curriculum advisor for MITAOE
(Maharashtra Institute of Technology, Aurangabad) — an autonomous engineering college
affiliated with Savitribai Phule Pune University (SPPU), India.
Department: B.Tech CSE with specialization in Artificial Intelligence & Machine Learning.

A new placement JD has arrived. Based on it, write specific curriculum update
recommendations for faculty members in Markdown format.

Include EXACTLY these sections:

## 1. Industry Demand Summary
(3-4 bullet points on what this JD signals about current market needs)

## 2. Curriculum Gaps
(Skills required in the JD that are NOT adequately covered in the B.Tech AIML syllabus.
Name the specific subject and unit that is missing or weak — be direct.)

## 3. Suggested Syllabus Updates
For each gap:
- Which existing subject and unit to update
- Exactly which topic to add or expand
- CO-PO/PSO mapping (use CO1–CO6, PO1–PO12 as per NBA/NAAC norms)

## 4. Practical Lab Approaches
(Specific hands-on exercises to add to existing lab sessions.
Format: Tool name + Dataset/task + Expected learning outcome)

## 5. Study Material & References
(Specific resources: NPTEL course links, Coursera/edX courses,
textbook name + author, GitHub repos worth assigning)

## 6. Semester Project Ideas
(1-2 complete project titles with a 2-line scope that demonstrates JD skills)

## 7. Quick-Win Actions This Semester
(Things faculty can do WITHOUT official syllabus revision:
 guest lecture, extra reading, tool demo in class, online assignment, etc.)

Rules:
- Write for faculty who understand academic constraints (NBA, NAAC, approval process)
- Be specific to AIML engineering education — no generic higher-ed advice
- Focus on practical, immediately actionable suggestions""",
    ),
    ("human", "Job Description:\n{jd_json}"),
])


# ── PUBLIC FUNCTION ───────────────────────────────────────────────────────────

@traceable(name="teacher-agent", run_type="chain")
def run(jd: dict) -> str:
    """
    Generate faculty curriculum recommendations for the given Job Description.

    Uses a LangChain chain:
      Prompt template → ChatGroq LLM → StrOutputParser (plain text output)

    LangSmith traces this entire run — you can review what was recommended
    for each JD in the LangSmith dashboard.

    Args:
        jd: parsed JD dict from email_agent (company, role, skills, etc.)

    Returns:
        Markdown string with full curriculum recommendations,
        or "" if the API call failed.
    """
    print("\n=== TEACHER AGENT: Generating curriculum recommendations ===")
    print(f"    Company : {jd.get('company')}  |  Role : {jd.get('role')}")

    try:
        # LangChain chain: prompt → LLM → plain text parser
        chain = TEACHER_RECO_PROMPT | _get_llm() | StrOutputParser()

        reco = chain.invoke({"jd_json": json.dumps(jd, indent=2)})

        print(f"    ✓ Teacher reco ready ({len(reco)} characters)")
        return reco

    except Exception as e:
        print(f"    ✗ Groq/LangChain error: {e}")
        return ""
