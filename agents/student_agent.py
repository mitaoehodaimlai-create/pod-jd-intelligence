"""
AGENT 2 — STUDENT AGENT
========================
Generates an interview preparation guide for students using a
LangChain + ChatGroq chain with optional RAG context.

────────────────────────────────────────────────────────────────────────────
LLM USAGE — WHAT THIS AGENT DOES
────────────────────────────────────────────────────────────────────────────

  Input  : JD dict (from email_agent) + optional RAG context (from rag/store.py)
  Output : Markdown string — a 7-section interview prep guide

  LangChain chain:

    STUDENT_PREP_PROMPT  ← fills {jd_content} with JD JSON + RAG context,
                           formats as a chat message list
           ↓
    ChatGroq             ← sends to Groq API, gets the prep guide back
           ↓
    StrOutputParser      ← extracts the plain text from AIMessage.content

  temperature=0.4 → allows some creativity for natural-sounding guidance
  (slightly higher than the JD parser's 0.1 — recommendations benefit from
  varied, non-robotic language)

────────────────────────────────────────────────────────────────────────────
RAG USAGE — HOW HISTORICAL CONTEXT IMPROVES OUTPUT
────────────────────────────────────────────────────────────────────────────

  The rag_context parameter is a formatted string like:

    "[3 SIMILAR PAST JDs FROM THIS SEMESTER FOR CONTEXT]
     Past JD #1: TCS | Backend Dev | Python, Java, SQL
     Past JD #2: Infosys | SDE | Python, REST, DSA
     Past JD #3: Wipro | Data Engineer | Python, Spark, SQL"

  WHY IT MATTERS:
    Without RAG: AI gives a good generic prep guide for THIS JD alone.
    With RAG:    AI notices Python + SQL appear across all 3 similar JDs and
                 says "Python and SQL are the #1 priorities this semester —
                 even if you already know basics, go deeper."
                 It can also note: "DSA appeared in 2/3 similar roles — allocate
                 extra prep time even though this JD doesn't list it explicitly."

  The LLM prompt instructs the model to use this context for pattern analysis.
  If rag_context is empty (no similar JDs yet), the agent works exactly as before.

────────────────────────────────────────────────────────────────────────────
LANGSMITH TRACING
────────────────────────────────────────────────────────────────────────────

  @traceable records every run() call with:
    - Full prompt (including RAG context if present)
    - Full generated brief
    - Latency and token counts
  Compare student briefs for different JDs side-by-side in LangSmith.
"""

import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

import config


# ── LLM CONFIGURATION ────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    """
    ChatGroq for student brief generation.
    temperature=0.4 → slightly creative, produces natural-sounding guidance
    (not too random, not robotically repetitive).
    """
    return ChatGroq(
        model       = config.LLM_MODEL,
        api_key     = config.GROQ_API_KEY,
        temperature = 0.4,
    )


# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────
# {jd_content} is filled at runtime with: JD JSON + optional RAG context.
# The system message tells the LLM WHO it is and WHAT structure to follow.

STUDENT_PREP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a career coach and technical interview trainer at MITAOE
for B.Tech students in the Computer Science (AI & ML) program.

A campus placement opportunity has arrived. Based on the Job Description,
write a complete, actionable interview preparation guide in Markdown.

If historical context from similar past JDs is provided, use it to:
  - Emphasize skills that appear repeatedly across similar roles
  - Note any skill gaps that keep appearing (signal to prepare harder)
  - Tailor the prep plan based on what this type of company typically expects

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
    # {jd_content} = "Job Description: ...\n\n[historical context if available]"
    ("human", "{jd_content}"),
])


# ── PUBLIC FUNCTION ───────────────────────────────────────────────────────────

@traceable(name="student-agent", run_type="chain")
def run(jd: dict, rag_context: str = "") -> str:
    """
    Generate a student interview prep brief for the given JD.

    ── LangChain CHAIN STEPS ─────────────────────────────────────────────────
      1. STUDENT_PREP_PROMPT: fills {jd_content} → returns [SystemMsg, HumanMsg]
      2. ChatGroq:            calls Groq API → returns AIMessage with Markdown text
      3. StrOutputParser:     extracts plain text string from AIMessage.content

      chain = STUDENT_PREP_PROMPT | _get_llm() | StrOutputParser()
      brief = chain.invoke({"jd_content": jd_content})

    ── RAG AUGMENTATION ──────────────────────────────────────────────────────
      rag_context (passed from main.py) contains similar past JDs from ChromaDB.
      When present, it is appended to the user message so the LLM can:
        - Spot recurring skill requirements across similar roles
        - Mention patterns like "Python is in every similar JD — top priority"
        - Adjust the prep plan weight based on historical frequency

    Args:
        jd:          parsed JD dict from email_agent
        rag_context: formatted string from rag_store.format_rag_context()
                     (empty string = no RAG, agent still works fine)

    Returns:
        Markdown string with the full prep guide.
        Returns "" on LLM/API error (non-fatal).
    """
    print("\n=== STUDENT AGENT: Generating prep brief ===")
    print(f"    Company : {jd.get('company')}  |  Role : {jd.get('role')}")
    if rag_context:
        print(f"    RAG     : historical context added to prompt")

    try:
        # Build the user message.
        # Plain JD JSON is always included.
        # rag_context is appended when available (the "Augmented" part of RAG).
        jd_content = f"Job Description:\n{json.dumps(jd, indent=2)}"
        if rag_context:
            jd_content += f"\n\n{rag_context}"

        # LangChain chain: PROMPT | LLM | PARSER
        # LangSmith traces each step automatically — prompt, response, timing.
        chain = STUDENT_PREP_PROMPT | _get_llm() | StrOutputParser()
        brief = chain.invoke({"jd_content": jd_content})

        print(f"    ✓ Brief ready ({len(brief)} chars)")
        return brief

    except Exception as e:
        print(f"    ✗ LLM error in student_agent.run: {e}")
        return ""
