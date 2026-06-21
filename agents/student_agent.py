"""
AGENT 2 — STUDENT AGENT
========================
Generates an interview preparation guide for students using a
LangChain + ChatGroq chain.

Libraries used:
  langchain-groq  → ChatGroq LLM
  langchain-core  → ChatPromptTemplate (builds the prompt), StrOutputParser
  langsmith       → @traceable (traces this agent in LangSmith dashboard)

LangSmith tracing:
  Every call to run() is recorded as a separate trace in LangSmith.
  You can compare briefs across different JDs in the dashboard.

Output format:
  Returns a Markdown string with these sections:
    1. Company & Role Overview
    2. Must-Know Topics
    3. Skill Gaps vs B.Tech AIML Curriculum
    4. 2-Week Prep Plan
    5. Practice Resources
    6. 10 Likely Interview Questions
    7. Projects to Showcase
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
    Create and return a ChatGroq LLM instance for student brief generation.
    temperature=0.4 → slightly creative, produces natural-sounding guidance.
    """
    return ChatGroq(
        model       = config.LLM_MODEL,
        api_key     = config.GROQ_API_KEY,
        temperature = 0.4,
    )


# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────

STUDENT_PREP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a career coach and technical interview trainer at MITAOE
for B.Tech students in the Computer Science (AI & ML) program.

A campus placement opportunity has arrived. Based on the Job Description,
write a complete, actionable interview preparation guide in Markdown.

Include EXACTLY these sections:

## 1. Company & Role Overview
(2-3 lines about the company and what the role involves)

## 2. Must-Know Topics
(For each required skill, list the key sub-topics students should study)

## 3. Skill Gaps vs B.Tech AIML Curriculum
(What the JD expects that is NOT typically well-covered in the program.
Be specific and honest — name the exact topic that is missing.)

## 4. 2-Week Prep Plan
(Day-by-day schedule. Label each day: "Day 1 (Mon)", "Day 2 (Tue)", etc.
Give a specific task for each day — not vague like "study Python".)

## 5. Practice Resources
(Name the platform + exactly what to practise there.
E.g.: LeetCode — Arrays/Strings/DP medium problems, Kaggle — Titanic notebook)

## 6. 10 Likely Interview Questions
(3 DSA/coding + 3 ML/AI theory + 2 tool/domain-specific + 2 HR behavioural)

## 7. Projects to Showcase
(One project idea per major required skill — give a short title and 1-line description)

Rules:
- Write for a final-year B.Tech student in India
- Keep language simple, practical, motivating
- Everything must be specific to THIS JD — no generic advice""",
    ),
    ("human", "Job Description:\n{jd_json}"),
])


# ── PUBLIC FUNCTION ───────────────────────────────────────────────────────────

@traceable(name="student-agent", run_type="chain")
def run(jd: dict) -> str:
    """
    Generate a student interview prep brief for the given Job Description.

    Uses a LangChain chain:
      Prompt template → ChatGroq LLM → StrOutputParser (plain text output)

    LangSmith traces this entire run, including the prompt sent and
    the response received from Groq.

    Args:
        jd: parsed JD dict from email_agent (company, role, skills, etc.)

    Returns:
        Markdown string with the full prep guide,
        or "" if the API call failed.
    """
    print("\n=== STUDENT AGENT: Generating prep brief ===")
    print(f"    Company : {jd.get('company')}  |  Role : {jd.get('role')}")

    try:
        # LangChain chain: prompt → LLM → plain text parser
        # LangSmith traces each step automatically.
        chain = STUDENT_PREP_PROMPT | _get_llm() | StrOutputParser()

        # Pass the JD as formatted JSON so the LLM can read all fields
        brief = chain.invoke({"jd_json": json.dumps(jd, indent=2)})

        print(f"    ✓ Student brief ready ({len(brief)} characters)")
        return brief

    except Exception as e:
        print(f"    ✗ Groq/LangChain error: {e}")
        return ""
