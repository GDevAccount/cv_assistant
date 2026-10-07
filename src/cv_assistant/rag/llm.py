"""Accès au modèle de langage : le seul fichier du RAG qui parle à l'API d'OpenAI."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from cv_assistant.rag.state import Exchange

# Petit modèle d'OpenAI : suffisant pour classer une question et reformuler des extraits.
CHAT_MODEL = "gpt-5.4-nano"
# Nombre de tours précédents envoyés au modèle : les plus anciens sont laissés de côté.
HISTORY_LIMIT = 6


class Llm:
    """Envoie au modèle des consignes, les tours précédents et un message, et renvoie sa réponse."""

    def __init__(self, model: str = CHAT_MODEL) -> None:
        self._model = model
        # La clé d'API est lue dans la variable d'environnement OPENAI_API_KEY.
        self._client = OpenAI()

    def complete(self, instructions: str, message: str, history: list[Exchange]) -> str:
        """Renvoie le texte rédigé par le modèle."""
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=build_input(message, history),
            reasoning={"effort": "low"},
        )
        return response.output_text

    def complete_json(
        self, instructions: str, message: str, history: list[Exchange], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Renvoie un objet JSON que le modèle est contraint de rendre conforme au schéma."""
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=build_input(message, history),
            reasoning={"effort": "low"},
            text={
                "format": {"type": "json_schema", "name": "reponse", "strict": True, "schema": schema}
            },
        )
        return json.loads(response.output_text)


def build_input(message: str, history: list[Exchange]) -> list[dict[str, str]]:
    """Messages envoyés au modèle : les derniers tours de la conversation, puis le message."""
    messages: list[dict[str, str]] = []
    for exchange in history[-HISTORY_LIMIT:]:
        messages.append({"role": "user", "content": exchange["question"]})
        messages.append({"role": "assistant", "content": exchange["answer"]})
    messages.append({"role": "user", "content": message})
    return messages
