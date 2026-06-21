"""
AGENT 3 — TEACHER AGENT
========================
This agent is the THIRD step in the pipeline.

What it does:
  Takes the same parsed Job Description and generates specific,
  actionable curriculum update recommendations for faculty members.

The recommendations include:
  • Industry demand summary (what the market wants right now)
  • Curriculum gaps (skills in JD not covered in B.Tech AIML syllabus)
  • Suggested syllabus updates with CO-PO/PSO mapping (NBA/NAAC compliant)
  • Practical lab exercises and tool-based assignments
  • Study materials and references (NPTEL, Coursera, textbooks)
  • 1-2 semester project ideas aligned with the JD
  • Quick-win actions faculty can take THIS semester (no approval needed)

Output:
  A detailed Markdown text ready to be sent to faculty via email.

Usage:
    from agents.teacher_agent import run
    recommendations = run(jd_dict)
"""

from groq import Groq
import json
import config

# ------------------------------------------------------------------
# PUBLIC FUNCTION  (called from main.py)
# ------------------------------------------------------------------

def run(jd):
    """
    Generate curriculum update recommendations for faculty based on the JD.

    Args:
        jd: dict — the parsed Job Description from email_agent

    Returns:
        String containing full faculty recommendations in Markdown format
    """
    print("\n=== TEACHER AGENT: Generating curriculum recommendations ===")
    print(f"    Company: {jd.get('company')}  |  Role: {jd.get('role')}")

    reco = _generate_with_groq(jd)

    if reco:
        print(f"    ✓ Teacher recommendations generated ({len(reco)} characters)")
    else:
        print("    ✗ Failed to generate teacher recommendations")

    return reco or ""


# ------------------------------------------------------------------
# PRIVATE HELPER
# ------------------------------------------------------------------

def _generate_with_groq(jd):
    """
    Sends the JD to Groq AI with instructions to write faculty-specific
    curriculum improvement recommendations.

    Args:
        jd: parsed JD dict

    Returns:
        Markdown text string, or None if API call fails
    """

    # Tell the AI about the academic context and what to generate
    system_prompt = """You are an academic curriculum advisor for MITAOE (Maharashtra Institute
of Technology, Aurangabad) — an autonomous college affiliated with SPPU, India.
The department is B.Tech CSE with specialization in Artificial Intelligence & Machine Learning.

A new Job Description has arrived from the campus placement office. Based on this JD,
write specific curriculum update recommendations for faculty members in Markdown format.

The recommendations must include these exact sections:

## 1. Industry Demand Summary
(What this JD tells us about current industry requirements — 3-4 bullet points)

## 2. Curriculum Gaps
(Skills in the JD that are NOT adequately covered in the standard B.Tech AIML syllabus.
Name the specific subject/unit that is missing or weak — be direct, not vague)

## 3. Suggested Syllabus Updates
For each gap, specify:
- Which existing subject and unit to update
- Exactly which topic to add or expand
- CO-PO/PSO mapping (use CO1–CO6, PO1–PO12 numbering as per NBA norms)

## 4. Practical Lab Approaches
(Specific hands-on exercises the faculty can add to existing lab sessions.
Give: tool name + dataset name + learning outcome for each suggestion)

## 5. Study Material & References
(Specific resources: NPTEL course links, Coursera/edX specializations,
textbook names with authors, GitHub repositories worth assigning as reading)

## 6. Semester Project Ideas
(1-2 complete project titles with a 2-line scope — projects students can
do in one semester that demonstrate the skills the JD requires)

## 7. Quick-Win Actions This Semester
(Things a faculty member can do WITHOUT waiting for official syllabus revision:
guest lecture, add a tool tutorial, assign an online course as extra credit, etc.)

Keep recommendations specific to AIML engineering education in India.
Write for faculty who understand academic constraints (syllabus approval process,
NBA/NAAC compliance, limited lab time).
"""

    # Convert the JD dict to a readable JSON string
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
            temperature=0.4,  # Some creativity needed for good recommendations
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"    Groq API error: {e}")
        return None
