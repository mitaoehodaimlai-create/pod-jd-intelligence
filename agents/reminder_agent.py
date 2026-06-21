"""
AGENT 5 — REMINDER & DEADLINE PLAN AGENT
==========================================
Generates two things:
  1. Application reminder — tells students WHEN to apply and what to submit
  2. Deadline-adaptive preparation plan — exactly N days of work based on
     how many days remain until the application deadline

The plan automatically scales:
  ≤ 7 days  → intensive sprint (every day matters)
  8–14 days → focused 2-week plan
  15–30 days → balanced 4-week plan
  > 30 days → full month+ roadmap
  No deadline → 14-day default plan
"""

import json
import re
from datetime import date, datetime, timedelta

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

import config


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model       = config.LLM_MODEL,
        api_key     = config.GROQ_API_KEY,
        temperature = 0.4,
    )


def _parse_deadline(deadline_str: str) -> date | None:
    """
    Try to parse a deadline string into a date object.
    Handles common formats: YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY,
    'June 30, 2025', '30 June 2025', etc.
    Returns None if parsing fails.
    """
    if not deadline_str or not deadline_str.strip():
        return None

    s = deadline_str.strip()

    # ISO format
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass

    # Common date patterns
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%d %B %Y",
                "%b %d, %Y", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    # Extract 4-digit year and try to guess
    year_match = re.search(r"\b(202[4-9]|203[0-9])\b", s)
    if year_match:
        # Try with just year — treat as Dec 31 of that year
        try:
            return date(int(year_match.group(1)), 12, 31)
        except ValueError:
            pass

    return None


def _days_remaining(deadline_str: str) -> tuple[int | None, str]:
    """
    Returns (days_remaining, human_readable_deadline).
    days_remaining is None if deadline cannot be parsed.
    """
    d = _parse_deadline(deadline_str)
    if d is None:
        return None, deadline_str or "not specified"

    today   = date.today()
    delta   = (d - today).days
    readable = d.strftime("%B %d, %Y")
    return delta, readable


def _plan_label(days: int | None) -> str:
    if days is None:
        return "14-day"
    if days <= 0:
        return "immediate"
    if days <= 7:
        return f"{days}-day sprint"
    if days <= 14:
        return "2-week"
    if days <= 30:
        return "4-week"
    return f"{min(days, 45)}-day"


REMINDER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a placement coordinator at MITAOE writing a preparation
reminder and application deadline notice for B.Tech AIML students.

You will be given a Job Description AND the number of days remaining until
the application deadline. Write a motivating, specific, actionable email.

The email must have EXACTLY these sections:

## Application Deadline Notice
**Company:** [name]
**Role:** [role]
**CTC:** [package]
**Application Deadline:** [date]
**Days Remaining:** [N days — or "DEADLINE PASSED" if 0 or negative]
**Apply Via:** POD Office / portal (whichever the JD mentions)

⚠️ Action Required: Submit your resume to the POD office by [date].
Documents needed: [list what typically needs to be submitted: resume, transcript, etc.]

## {plan_label} Preparation Plan
(Build a day-by-day schedule for EXACTLY {days_available} days.
Label each day: Day 1, Day 2, etc.
Each day must have:
  - Morning task (1-2 hrs): [specific topic to study]
  - Evening task (1-2 hrs): [specific practice / coding / project work]
  - Goal: [what the student should be able to do after this day]

If deadline is very close (≤3 days), replace the daily plan with an
"Emergency Prep Checklist" — the absolute minimum to prepare in time.)

## Key Topics to Focus On This Week
(3-5 highest-priority skills from the JD — focus here first)

## Don't Forget
- Update your resume to highlight: [2-3 specific skills from the JD]
- Practice these 3 questions out loud before the interview: [list 3 from the JD domain]
- Check POD portal / notice board daily for interview schedule updates

## Motivational Note
(2-3 lines of genuine, specific encouragement tied to this company/role — not generic)

Rules:
- Tone: friendly, urgent-but-calm, supportive
- Language: simple English a final-year student understands
- No jargon, no corporate speak
- The prep plan must match EXACTLY the number of days available""",
    ),
    ("human", "{jd_content}"),
])


@traceable(name="reminder-agent", run_type="chain")
def run(jd: dict) -> str:
    """
    Generate an application reminder + deadline-adaptive prep plan.

    Args:
        jd: parsed JD dict (must contain 'deadline' field for best results)

    Returns:
        Markdown string — deadline notice + day-by-day prep plan.
        Returns "" on error (non-fatal).
    """
    print("\n=== REMINDER AGENT: Building deadline prep plan ===")
    print(f"    Company  : {jd.get('company')}  |  Role : {jd.get('role')}")

    days, readable_deadline = _days_remaining(jd.get("deadline", ""))

    if days is None:
        print(f"    Deadline : not parseable ('{jd.get('deadline', '')}') — using 14-day default")
        days_available = 14
    elif days <= 0:
        print(f"    Deadline : PASSED ({readable_deadline})")
        days_available = 2
    else:
        print(f"    Deadline : {readable_deadline} ({days} days remaining)")
        days_available = min(days, 45)

    label = _plan_label(days)
    print(f"    Plan     : {label}")

    try:
        jd_content = (
            f"Job Description:\n{json.dumps(jd, indent=2)}\n\n"
            f"Deadline Info:\n"
            f"  Raw deadline string: {jd.get('deadline', 'not provided')}\n"
            f"  Parsed deadline    : {readable_deadline}\n"
            f"  Days remaining     : {days if days is not None else 'unknown (use 14)'}\n"
            f"  Days available     : {days_available}\n"
            f"  Plan type          : {label}"
        )

        chain  = REMINDER_PROMPT | _get_llm() | StrOutputParser()
        result = chain.invoke({
            "jd_content"   : jd_content,
            "plan_label"   : label.title(),
            "days_available": days_available,
        })

        print(f"    ✓ Reminder + plan ready ({len(result)} chars)")
        return result

    except Exception as e:
        print(f"    ✗ LLM error in reminder_agent.run: {e}")
        return ""
