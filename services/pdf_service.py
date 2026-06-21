"""
PDF SERVICE
===========
Extracts plain text from a PDF file given as raw bytes.

Why two libraries?
  pdfplumber  — works well for text-based PDFs (most JDs)
  pypdf       — fallback for PDFs that pdfplumber cannot read

Usage:
    from services.pdf_service import extract_text_from_pdf
    text = extract_text_from_pdf(pdf_bytes)
"""

import io


def extract_text_from_pdf(pdf_bytes):
    """
    Try to extract all readable text from a PDF file.

    Args:
        pdf_bytes: raw bytes of the PDF file (from email attachment)

    Returns:
        A single string with all the text, or "" if nothing could be extracted.
    """

    # --- Attempt 1: pdfplumber (better accuracy for most JD PDFs) ---
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Extract text page by page and join with newlines
            pages_text = [page.extract_text() or "" for page in pdf.pages]

        full_text = "\n".join(pages_text).strip()
        if full_text:
            return full_text

    except Exception:
        pass   # pdfplumber failed — try the next library

    # --- Attempt 2: pypdf (handles different PDF encodings) ---
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]

        full_text = "\n".join(pages_text).strip()
        if full_text:
            return full_text

    except Exception:
        pass   # pypdf also failed

    # Both libraries failed — return empty string
    return ""
