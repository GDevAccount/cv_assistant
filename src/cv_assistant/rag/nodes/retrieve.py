"""Nœud retrieve : cherche dans les ouvrages les extraits proches de la question."""

from __future__ import annotations

from cv_assistant.rag.retriever import Retriever
from cv_assistant.rag.state import RagState


class RetrieveNode:
    """Pose dans l'état les top_k extraits d'ouvrages les plus proches de la question.

    La recherche porte sur la question reformulée par classify : une question de suivi
    comme « Et en production ? » ne désigne aucun sujet à elle seule.
    """

    def __init__(self, retriever: Retriever, top_k: int = 5) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def __call__(self, state: RagState) -> RagState:
        return {"passages": self._retriever.search(state["standalone_question"], self._top_k)}
