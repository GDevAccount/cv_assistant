"""Nœud off_topic : décline poliment une question hors sujet, sans appeler le modèle."""

from __future__ import annotations

from cv_assistant.rag.state import RagState

OFF_TOPIC_ANSWER_FRENCH = (
    "Je vous remercie pour votre question, mais elle sort du cadre de cet assistant. "
    "Je peux vous renseigner sur le parcours de Guillaume Legall et sur ses connaissances "
    "en IA engineering."
)
OFF_TOPIC_ANSWER_ENGLISH = (
    "Thank you for your question, but it falls outside the scope of this assistant. "
    "I can tell you about Guillaume Legall's background and his knowledge of AI engineering."
)


class OffTopicNode:
    """Pose dans l'état un message de refus fixe : en français, ou en anglais pour toute autre langue."""

    def __call__(self, state: RagState) -> RagState:
        answer = OFF_TOPIC_ANSWER_FRENCH if state["language"] == "French" else OFF_TOPIC_ANSWER_ENGLISH
        return {"answer": answer, "history": [{"question": state["question"], "answer": answer}]}
