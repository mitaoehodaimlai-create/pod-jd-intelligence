"""
AGENT 2 — STUDENT AGENT
========================
This agent is the SECOND step in the pipeline.

What it does:
  Takes a parsed Job Description (from the Email Agent) and generates
  a complete, actionable interview preparation guide for students.

The guide includes:
  • Must-know topics mapped to each required skill
  • Skill gaps — what the JD expects but B.Tech AIML may not cover
  • A 2-week day-by-day study plan
  • Practice resources (LeetCode, Kaggle, NPTEL, HuggingFace, etc.)
  • 10 likely interview questions (technical + HR)
  • Project ideas to build and showcase

Output:
  A detailed Markdown text ready to be sent to students via email.

Usage:
    from agents.student_agent import run
    brief = run(jd_dict)
"""

from groq import Groq
import json
import config

# ------------------------------------------------------------------
# PUBLIC FUNCTION  (called from main.py)
# ------------------------------------------------------------------

def run(jd):
    """
    Generate an interview prep brief for students based on the given JD.

    Args:
        jd: dict — the parsed Job Description from email_agent

    Returns:
        String containing the full prep guide in Markdown format
    """
    print("\n=== STUDENT AGENT: Generating prep brief ===")
    print(f"    Company: {jd.get('company')}  |  Role: {jd.get('role')}")

    brief = _generate_with_groq(jd)

    if brief:
        print(f"    ✓ Student brief generated ({len(brief)} characters)")
    else:
        print("    ✗ Failed to generate student brief")

    return brief or ""


# ------------------------------------------------------------------
# PRIVATE HELPER
# ------------------------------------------------------------------

def _generate_with_groq(jd):
    """
    Sends the JD to Groq AI with instructions to write a student-friendly
    interview prep guide.

    Args:
        jd: parsed JD dict

    Returns:
        Markdown text string, or None if API call fails
    """

    # Tell the AI exactly what kind of guide to write
    system_prompt = """You are a career coach and technical interview trainer at MITAOE
for B.Tech students in the Computer Science (AI & ML) stream.

A student has been shortlisted for a campus placement. Based on the Job Description
provided, write a complete interview preparation guide in Markdown format.

The guide must include these exact sections:

## 1. Company & Role Overview
(2-3 lines about the company and what the role involves)

## 2. Must-Know Topics
(For each required skill in the JD, list the key sub-topics to study)

## 3. Skill Gaps vs Your B.Tech AIML Curriculum
(What the JD asks for that is NOT typically covered well in the B.Tech program —
be honest and specific, not generic)

## 4. 2-Week Prep Plan
(Day-by-day schedule — label each day e.g. "Day 1 (Mon)", with specific tasks)

## 5. Practice Resources
(Platform name + what to practice there — LeetCode, Kaggle, HuggingFace, NPTEL, etc.)

## 6. 10 Likely Interview Questions
(Mix of: 3 DSA/coding, 3 ML/AI theory, 2 domain/tool-specific, 2 HR)

## 7. Projects to Showcase
(One concrete project idea per major required skill — give a short project title
and 1-line description)

Keep the language simple, practical, and motivating.
Write for a final-year engineering student in India.
Do NOT write generic advice — make it specific to this exact JD.
"""

    # Convert the JD dict to a clean JSON string for the AI to read
    jd_text = json.dumps(jd, indent=2, ensure_ascii=False)

    try:
        client = Groq(api_key=config.GROQ_API_KEY)

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Job Description:\n{jd_text}"},
            ],
            max_tokens=3000,
            temperature=0.4,  # Slightly higher = more natural writing style
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"    Groq API error: {e}")
        return None
