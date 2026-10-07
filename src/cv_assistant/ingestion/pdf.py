"""Extraction du texte d'un PDF.

Seul le texte présent dans le fichier est lu : un PDF scanné (images de pages) ne
donne aucun texte, il n'y a pas d'OCR.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def read_pdf_sections(pdf_path: Path) -> list[str]:
    """Renvoie le texte de chaque page du PDF, dans l'ordre des pages."""
    return [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
