"""Lecture du CV, envoyé en entier au modèle à chaque question.

Le CV est assez court pour tenir dans le prompt : il n'est donc ni découpé en chunks ni
indexé dans ChromaDB, contrairement aux ouvrages.
"""

from __future__ import annotations

from pathlib import Path

from pypdf.errors import PdfReadError

from cv_assistant.ingestion.pdf import read_pdf_sections


class CvNotFoundError(RuntimeError):
    """Le fichier du CV est absent, illisible ou sans texte extractible."""


def load_cv(cv_path: Path) -> str:
    """Renvoie le texte complet du CV (un PDF), pages mises bout à bout."""
    try:
        pages = read_pdf_sections(cv_path)
    except (OSError, PdfReadError) as error:
        raise CvNotFoundError(f"Impossible de lire le CV {cv_path} : {error}") from error

    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text:
        raise CvNotFoundError(f"Aucun texte extractible dans le CV {cv_path}.")
    return text
