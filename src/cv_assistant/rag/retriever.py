"""Recherche des chunks d'ouvrages proches d'une question : le seul fichier du RAG qui interroge ChromaDB."""

from __future__ import annotations

from chromadb.api.models.Collection import Collection


class Retriever:
    """Cherche dans la collection des ouvrages les chunks proches d'une question."""

    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    def search(self, question: str, top_k: int = 5) -> list[str]:
        """Renvoie le texte des top_k chunks les plus proches de la question, du plus proche au moins proche."""
        # ChromaDB vectorise la question, puis la compare aux vecteurs des chunks.
        results = self._collection.query(
            query_texts=[question], n_results=top_k, include=["documents"]
        )
        # Une seule question est envoyée : on lit donc la première (et unique) liste de résultats.
        return results["documents"][0]
