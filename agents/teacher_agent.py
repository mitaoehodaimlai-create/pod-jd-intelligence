"""
AGENT 3 — TEACHER AGENT
========================
Generates curriculum update recommendations for faculty using a
LangChain + ChatGroq chain with optional RAG context.

────────────────────────────────────────────────────────────────────────────
LLM USAGE — WHAT THIS AGENT DOES
────────────────────────────────────────────────────────────────────────────

  Input  : JD dict (from email_agent) + optional RAG context (from rag/store.py)
  Output : Markdown string — a 7-section curriculum recommendation report

  LangChain chain (same pattern as student_agent):

    TEACHER_RECO_PROMPT  ← fills {jd_content} with JD JSON + RAG context
           ↓
    ChatGroq             ← calls Groq API (Llama 3.3-70B), gets Markdown back
           ↓
    StrOutputParser      ← returns the plain Markdown string

  temperature=0.4 → some creativity for practical lab suggestions while
  keeping academic recommendations grounded and specific.

────────────────────────────────────────────────────────────────────────────
RAG USAGE — HOW HISTORICAL CONTEXT IMPROVES FACULTY RECOMMENDATIONS
────────────────────────────────────────────────────────────────────────────

  The rag_context parameter adds patterns from similar past JDs.

  WHY IT MATTERS FOR FACULTY:
    Without RAG: "This JD needs Docker — add a Docker lab session."
    With RAG:    "Docker appears in 5/7 similar JDs this semester.
                 This is a persistent curriculum gap, not a one-off.
                 Recommend escalating to a full Docker module, not a
                 one-class demo."

  The model can also detect RECURRING recommendations:
    "DSA gap mentioned 4 times this semester — suggest a standing weekly
    problem-solving session rather than ad-hoc mentions."

  This turns isolated JD-level suggestions into semester-level insights.

────────────────────────────────────────────────────────────────────────────
CO-PO MAPPING (Course Outcomes / Programme Outcomes)
────────────────────────────────────────────────────────────────────────────
  The prompt asks the LLM to map each suggestion to CO1–CO6 and PO1–PO12.
  These are NBA/NAAC accreditation codes used in Indian engineering colleges.
  The LLM knows these codes from its training data and applies them correctly.
  Faculty can use these mappings directly in OBE (Outcome-Based Education) docs.

────────────────────────────────────────────────────────────────────────────
LANGSMITH TRACING
────────────────────────────────────────────────────────────────────────────
  @traceable records every run() — full prompt (including RAG context),
  full recommendation text, latency, tokens.
  Review past recommendations in LangSmith to avoid repetition and track
  which suggestions have been made before.
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
    ChatGroq for curriculum recommendation generation.
    temperature=0.4 → balanced: specific enough for academic use, creative
    enough to suggest novel approaches (lab ideas, project titles, etc.)
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

If historical context from similar past JDs is provided, use it to:
  - Identify RECURRING skill gaps (seen in multiple JDs = urgent curriculum fix)
  - Distinguish one-off requests from sustained industry trends
  - Suggest proportional action: a skill in 1 JD → guest lecture;
    a skill in 5+ JDs → formal syllabus revision or new elective

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
    # {jd_content} = "Job Description: ...\n\n[historical context if available]"
    ("human", "{jd_content}"),
])


# ── PUBLIC FUNCTION ───────────────────────────────────────────────────────────

@traceable(name="teacher-agent", run_type="chain")
def run(jd: dict, rag_context: str = "") -> str:
    """
    Generate faculty curriculum recommendations for the given JD.

    ── LangChain CHAIN STEPS ─────────────────────────────────────────────────
      1. TEACHER_RECO_PROMPT: fills {jd_content} → [SystemMessage, HumanMessage]
      2. ChatGroq:            Groq API call → AIMessage with Markdown text
      3. StrOutputParser:     plain string from AIMessage.content

      chain = TEACHER_RECO_PROMPT | _get_llm() | StrOutputParser()
      reco  = chain.invoke({"jd_content": jd_content})

    ── RAG AUGMENTATION ──────────────────────────────────────────────────────
      rag_context from rag_store.format_rag_context() contains similar past JDs.
      Faculty get semester-level insights:
        - "Docker appeared in 5 JDs this semester → upgrade from demo to module"
        - "This is the 3rd JD needing system design → recommend a new elective"

    Args:
        jd:          parsed JD dict from email_agent
        rag_context: formatted string from rag_store.format_rag_context()
                     (empty string = no RAG, agent still works fine)

    Returns:
        Markdown string with full curriculum recommendations.
        Returns "" on LLM/API error (non-fatal).
    """
    print("\n=== TEACHER AGENT: Generating curriculum recommendations ===")
    print(f"    Company : {jd.get('company')}  |  Role : {jd.get('role')}")
    if rag_context:
        print(f"    RAG     : historical context added to prompt")

    try:
        # Build user message: JD JSON + RAG context (if any)
        jd_content = f"Job Description:\n{json.dumps(jd, indent=2)}"
        if rag_context:
            jd_content += f"\n\n{rag_context}"

        # LangChain chain: PROMPT → LLM → PARSER
        # LangSmith traces every step — see at https://smith.langchain.com
        chain = TEACHER_RECO_PROMPT | _get_llm() | StrOutputParser()
        reco  = chain.invoke({"jd_content": jd_content})

        print(f"    ✓ Reco ready ({len(reco)} chars)")
        return reco

    except Exception as e:
        print(f"    ✗ LLM error in teacher_agent.run: {e}")
        return ""
