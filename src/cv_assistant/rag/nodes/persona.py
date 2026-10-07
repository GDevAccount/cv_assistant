"""Ton de l'assistant, commun aux nœuds qui rédigent une réponse (profile et ai_engineering)."""

from __future__ import annotations

TONE = """\
Ton et forme :
- Vouvoie ton interlocuteur et reste courtois et professionnel.
- Parle de Guillaume à la troisième personne.
- Sois concis : quelques phrases, ou une courte liste. Ne termine pas en proposant une
  aide supplémentaire.
- Réponds dans la langue indiquée à la fin du message, même si le CV et les notes sont
  dans une autre langue."""


def language_reminder(language: str) -> str:
    """Dernière ligne du message : la langue de réponse, détectée par le nœud classify.

    Elle est écrite en anglais pour être comprise quelle que soit la langue visée.
    """
    return f"Answer in {language}."
