"""Ingest node — PDF extraction is done here; email fetching is pre-step."""
from __future__ import annotations

from tools.pdf_tool import extract_text_from_pdf
from workflow.state import PipelineState


def ingest_node(state: PipelineState, attachment_bytes: dict[str, bytes]) -> dict:
    """Extract text from all PDF attachments and attach to state."""
    pdf_texts: list[str] = []
    for filename, raw in attachment_bytes.items():
        text = extract_text_from_pdf(raw)
        if text:
            pdf_texts.append(f"[PDF: {filename}]\n{text}")

    return {"pdf_texts": pdf_texts, "error": None}
