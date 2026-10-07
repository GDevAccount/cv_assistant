"""Modèle d'embedding partagé par l'ingest et la recherche.

Les questions (en français) et les documents (en anglais) doivent être vectorisés par le
même modèle multilingue pour être comparables. Changer de modèle impose de supprimer la
collection et de relancer l'ingest.
"""

from __future__ import annotations

import os

from chromadb.api.types import Documents, EmbeddingFunction
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Multilingue, vecteurs de 1 536 dimensions, entrées jusqu'à 8 191 tokens.
EMBEDDING_MODEL = "text-embedding-3-small"

# Une requête d'embedding accepte au plus 2 048 textes et 300 000 tokens au total. Un token
# couvre en général plusieurs caractères : borner les caractères laisse donc de la marge.
MAX_BATCH_TEXTS = 2048
MAX_BATCH_CHARACTERS = 150_000


class MissingApiKeyError(RuntimeError):
    """La clé d'API OpenAI n'est pas définie dans l'environnement."""


def build_embedding_function(model_name: str = EMBEDDING_MODEL) -> EmbeddingFunction[Documents]:
    """Prépare les appels à l'API d'embedding d'OpenAI (la clé est lue dans l'environnement)."""
    if not os.getenv("OPENAI_API_KEY"):
        raise MissingApiKeyError(
            "La clé d'API OpenAI est absente : renseignez OPENAI_API_KEY dans le fichier "
            ".env ou dans l'environnement."
        )
    return OpenAIEmbeddingFunction(model_name=model_name)
