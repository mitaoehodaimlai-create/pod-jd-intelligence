"""Extract plain text from PDF bytes. pdfplumber first, pypdf as fallback."""
from __future__ import annotations

import io


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages).strip()
    except Exception:
        pass

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception:
        return ""
