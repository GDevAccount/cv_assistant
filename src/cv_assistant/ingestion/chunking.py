"""Découpage d'un texte en chunks qui se chevauchent, sans couper les mots."""

from __future__ import annotations


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être supérieur à zéro.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap doit être compris entre zéro et chunk_size exclus.")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Découpe le texte en chunks d'au plus chunk_size caractères.

    Chaque chunk reprend environ les chunk_overlap derniers caractères du précédent.
    """
    validate_chunking(chunk_size, chunk_overlap)

    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = _chunk_end(text, start, chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = _next_chunk_start(text, start, end, chunk_overlap)

    return chunks


def _chunk_end(text: str, start: int, chunk_size: int) -> int:
    """Fin du chunk commençant à start, ramenée au dernier espace pour ne pas couper un mot."""
    end = min(start + chunk_size, len(text))
    if end == len(text):
        return end

    last_space = text.rfind(" ", start, end)
    # Sans espace dans le chunk (mot plus long que chunk_size), on coupe dans le mot.
    return last_space if last_space > start else end


def _next_chunk_start(text: str, start: int, end: int, chunk_overlap: int) -> int:
    """Début du chunk suivant : chunk_overlap caractères avant end, calé sur un début de mot."""
    if not chunk_overlap:
        return end

    # max(start + 1, ...) garantit qu'on avance toujours d'au moins un caractère.
    overlap_start = max(start + 1, end - chunk_overlap)

    # On préfère reculer jusqu'au début du mot en cours...
    space_before = text.rfind(" ", start, overlap_start)
    if space_before >= start:
        return space_before + 1

    # ...sinon on avance jusqu'au début du mot suivant...
    space_after = text.find(" ", overlap_start, end)
    if space_after >= 0:
        return space_after + 1

    # ...et s'il n'y a aucun espace, on renonce au chevauchement.
    return end
