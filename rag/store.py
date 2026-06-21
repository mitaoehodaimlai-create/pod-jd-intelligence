"""
RAG (RETRIEVAL-AUGMENTED GENERATION) STORE
===========================================

RAG explained in one line: "give the LLM relevant past JDs so it can
make better recommendations for the current one."

────────────────────────────────────────────────────────────────────────────
WHAT IS RAG AND WHY DOES IT MATTER HERE?
────────────────────────────────────────────────────────────────────────────

  Without RAG:
    AI sees only THIS JD → generates generic recommendations

  With RAG:
    AI sees THIS JD + 3 similar past JDs → context-aware recommendations
    e.g., "Python appears in ALL 4 similar JDs — it is the #1 priority skill"
    e.g., "DSA gaps appeared in 3 similar roles — escalate to curriculum board"
    e.g., "TCS has posted 4 times — they consistently value system design"

────────────────────────────────────────────────────────────────────────────
HOW RAG WORKS (step by step):
────────────────────────────────────────────────────────────────────────────

  Step R1 — STORE (done once per new JD):
    JD text → Embedding model → 384-number vector → ChromaDB (disk)

  Step R2 — RETRIEVE (done before each agent call):
    Current JD text → Embedding model → 384-number vector
    → ChromaDB finds top-3 closest vectors → returns those JD dicts

  Step A  — AUGMENT (done in agent run()):
    Format retrieved JDs as text → inject into LLM prompt

  Step G  — GENERATE (done in ChatGroq call):
    LLM reads: current JD + historical context → richer output

────────────────────────────────────────────────────────────────────────────
WHAT IS A VECTOR / EMBEDDING?
────────────────────────────────────────────────────────────────────────────

  Text:   "Python machine learning data pipelines SQL"
  Vector: [-0.12, 0.84, 0.33, 0.01, ... 384 numbers total]

  Two JDs with similar skills produce vectors that are "close" in 384-D space.
  ChromaDB finds the k closest vectors in milliseconds — this is "semantic search".
  It finds meaning-level similarity, not just keyword overlap.

  "Python developer with ML experience" ≈ "AI engineer with Python background"
  even if zero words are shared — because their vectors are close.

────────────────────────────────────────────────────────────────────────────
EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2
────────────────────────────────────────────────────────────────────────────
  - Free and open-source (HuggingFace)
  - Downloads ~90MB once to ~/.cache/huggingface/
  - Runs entirely on CPU — no GPU needed, works on any laptop
  - Produces 384-dimensional vectors
  - Fast: ~5ms per JD on laptop CPU

LangChain components used:
  HuggingFaceEmbeddings   text → 384-dim vector (wraps sentence-transformers)
  Chroma                  stores/searches vectors on disk (wraps ChromaDB)
  Document                standard document type: page_content + metadata
"""

from __future__ import annotations

import json
from typing import Optional

# LangChain's standard document wrapper: text (to embed) + metadata (stored as-is)
from langchain.schema import Document

# HuggingFaceEmbeddings: loads a sentence-transformer model locally.
# Converts any text string to a 384-number vector.
# First import triggers model download (~90MB, one time only).
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma: LangChain wrapper around ChromaDB.
# Persists vectors to disk so they survive across runs.
from langchain_community.vectorstores import Chroma

import config


# ── EMBEDDING MODEL ───────────────────────────────────────────────────────────
# "all-MiniLM-L6-v2" maps text to 384 semantic dimensions.
# One model instance shared across all calls to avoid re-loading.
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def _get_vectorstore() -> Chroma:
    """
    Initialize (or return cached) ChromaDB + embedding model.

    ChromaDB persists all vectors to RAG_DB_PATH on disk.
    JDs accumulate across runs — the more JDs processed, the better the context.
    """
    global _embeddings, _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    print("RAG: Loading embedding model (one-time ~30s download on first run)...")

    # HuggingFaceEmbeddings wraps sentence-transformers.
    # normalize_embeddings=True ensures cosine similarity works correctly.
    _embeddings = HuggingFaceEmbeddings(
        model_name     = _EMBED_MODEL,
        model_kwargs   = {"device": "cpu"},                    # CPU — works on any machine
        encode_kwargs  = {"normalize_embeddings": True},       # required for cosine similarity
    )

    # Chroma reads/writes to RAG_DB_PATH (a directory with SQLite + vector index files).
    # get_or_create: if no database exists yet, creates an empty one.
    _vectorstore = Chroma(
        collection_name   = "jds",
        persist_directory = str(config.RAG_DB_PATH),
        embedding_function = _embeddings,
    )

    count = _vectorstore._collection.count()
    print(f"RAG: Ready — {count} JD(s) in store at {config.RAG_DB_PATH.name}/")

    return _vectorstore


def _jd_to_text(jd: dict) -> str:
    """
    Convert a JD dict to a single searchable text string.

    This text is what gets embedded into a vector.
    We include the most semantically important fields — role, skills, tech.
    Company/location/CTC go into metadata (not the vector) for filtering.
    """
    parts = [
        f"Role: {jd.get('role', '')}",
        f"Skills: {', '.join(jd.get('required_skills', []))}",
        f"Technologies: {', '.join(jd.get('tools_tech', []))}",
        f"Responsibilities: {'; '.join(jd.get('responsibilities', [])[:3])}",
        f"Nice to have: {', '.join(jd.get('nice_to_have', []))}",
        f"Eligibility: {jd.get('eligibility', '')}",
    ]
    # Only include lines that have actual content after the ":"
    return "\n".join(p for p in parts if p.split(": ", 1)[-1].strip())


# ── PUBLIC FUNCTIONS ──────────────────────────────────────────────────────────

def add_jd(jd: dict) -> bool:
    """
    STORE STEP — embed a parsed JD and save to ChromaDB.

    Called from main.py right after email_agent.parse_jd() succeeds.

    What happens internally:
      1. _jd_to_text(jd)        → builds the text to embed
      2. _embeddings.embed(text) → 384-number vector
      3. ChromaDB.add(vector + metadata) → saved to disk

    The full JD dict is stored as JSON in metadata so it can be
    reconstructed exactly when retrieved later.

    Returns True on success, False on failure (non-fatal — pipeline continues).
    """
    try:
        vs       = _get_vectorstore()
        jd_text  = _jd_to_text(jd)
        if not jd_text.strip():
            return False

        # Document wraps:
        #   page_content → text that gets embedded into a vector
        #   metadata     → additional fields stored as-is for retrieval
        doc = Document(
            page_content = jd_text,
            metadata = {
                "company":   jd.get("company", "Unknown"),
                "role":      jd.get("role", "Unknown"),
                "location":  jd.get("location", ""),
                "deadline":  jd.get("deadline", ""),
                "ctc":       jd.get("ctc", ""),
                # Full JD stored as JSON string — reconstructed in get_similar_jds()
                "jd_json":   json.dumps(jd),
            },
        )

        vs.add_documents([doc])
        print(f"RAG: Stored → {jd.get('company')} / {jd.get('role')}")
        return True

    except Exception as e:
        print(f"RAG: add_jd failed — {e}")
        return False


def get_similar_jds(jd: dict, k: int = 3) -> list[dict]:
    """
    RETRIEVE STEP — find the k most semantically similar past JDs.

    How similarity works:
      1. _jd_to_text(current_jd) → text
      2. Embedding model converts text → 384-number vector
      3. ChromaDB finds top-k stored vectors with highest cosine similarity
      4. Returns the original JD dicts from metadata

    "Similar" = same role type, overlapping skills, comparable responsibilities.
    A Python ML engineer role surfaces other ML/data roles, not Java backend roles.

    Returns empty list if:
      - The store has no JDs yet (first run)
      - All past JDs are too different
      - Embedding or ChromaDB errors (non-fatal)
    """
    try:
        vs = _get_vectorstore()

        if vs._collection.count() == 0:
            return []      # Nothing stored yet

        query_text = _jd_to_text(jd)
        if not query_text.strip():
            return []

        # similarity_search: embed query_text, find k nearest vectors in ChromaDB.
        # Returns Document objects with page_content and metadata.
        results = vs.similarity_search(query_text, k=k)

        similar = []
        for doc in results:
            jd_json_str = doc.metadata.get("jd_json", "")
            if jd_json_str:
                try:
                    similar.append(json.loads(jd_json_str))
                except json.JSONDecodeError:
                    pass

        return similar

    except Exception as e:
        print(f"RAG: get_similar_jds failed — {e}")
        return []


def format_rag_context(similar_jds: list[dict]) -> str:
    """
    AUGMENT STEP — format retrieved JDs as plain text for the LLM prompt.

    This text is appended to the agent's user message so the LLM can see
    historical patterns when generating recommendations.

    LLM reasoning enabled by this context:
      "Python appears in all 3 past JDs → #1 must-study skill"
      "DSA is in 2/3 past JDs → include in prep plan even if not in current JD"
      "Infosys consistently wants system design → add that project idea"

    Returns empty string if no similar JDs found.
    (Agents still work fine with empty context — RAG is additive, not required.)
    """
    if not similar_jds:
        return ""

    lines = [f"[{len(similar_jds)} SIMILAR PAST JD(s) FROM THIS SEMESTER FOR CONTEXT]"]

    for i, past_jd in enumerate(similar_jds, 1):
        lines.append(f"\nPast JD #{i}:")
        lines.append(f"  Company  : {past_jd.get('company', '?')}")
        lines.append(f"  Role     : {past_jd.get('role', '?')}")
        lines.append(f"  Skills   : {', '.join(past_jd.get('required_skills', []))}")
        lines.append(f"  Tech     : {', '.join(past_jd.get('tools_tech', []))}")
        if past_jd.get("ctc"):
            lines.append(f"  CTC      : {past_jd.get('ctc')}")

    return "\n".join(lines)


def get_skill_trends(top_n: int = 10) -> dict[str, int]:
    """
    Analyze all stored JDs and return the most frequently requested skills.

    Useful for curriculum planning and status display:
      {"python": 18, "sql": 15, "machine learning": 12, "docker": 7, ...}

    Returns empty dict if no JDs in store or on error.
    """
    try:
        vs    = _get_vectorstore()
        total = vs._collection.count()
        if total == 0:
            return {}

        all_docs = vs._collection.get(include=["metadatas"])

        skill_counts: dict[str, int] = {}
        for meta in all_docs.get("metadatas", []):
            jd_json_str = meta.get("jd_json", "")
            if not jd_json_str:
                continue
            try:
                past_jd = json.loads(jd_json_str)
                all_skills = (
                    past_jd.get("required_skills", []) +
                    past_jd.get("tools_tech", [])
                )
                for skill in all_skills:
                    key = skill.strip().lower()
                    if key:
                        skill_counts[key] = skill_counts.get(key, 0) + 1
            except Exception:
                pass

        # Return top-N sorted by frequency
        return dict(
            sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

    except Exception as e:
        print(f"RAG: get_skill_trends failed — {e}")
        return {}


def get_store_stats() -> dict:
    """Return count, unique companies, and top skills — shown in `python main.py status`."""
    try:
        vs    = _get_vectorstore()
        total = vs._collection.count()
        if total == 0:
            return {"total_jds": 0, "companies": [], "top_skills": {}}

        all_docs  = vs._collection.get(include=["metadatas"])
        companies = sorted({
            m.get("company", "")
            for m in all_docs.get("metadatas", [])
            if m.get("company")
        })
        return {
            "total_jds":  total,
            "companies":  companies,
            "top_skills": get_skill_trends(top_n=5),
        }

    except Exception as e:
        return {"total_jds": 0, "companies": [], "top_skills": {}, "error": str(e)}
