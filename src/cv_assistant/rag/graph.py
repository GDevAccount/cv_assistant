"""Graphe LangGraph de l'assistant : le point d'entrée du RAG.

Une question entre par START, le nœud classify la classe, puis l'arête conditionnelle
l'envoie vers la branche de sa route, qui rédige la réponse.

    START → classify ─┬─ profile ────────→ profile ───────────────────┐
                      ├─ ai_engineering ─→ retrieve → ai_engineering ─┼→ END
                      └─ off_topic ──────→ off_topic ─────────────────┘

Le graphe garde en mémoire l'état de chaque conversation : les questions posées avec le
même thread_id se suivent, et chaque nœud reçoit les tours précédents dans history.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from cv_assistant.rag.llm import Llm
from cv_assistant.rag.nodes import (
    AiEngineeringNode,
    ClassifyNode,
    OffTopicNode,
    ProfileNode,
    RetrieveNode,
)
from cv_assistant.rag.retriever import Retriever
from cv_assistant.rag.state import RagState, Route

# Premier nœud de chaque route. Une question d'IA engineering passe d'abord par retrieve.
ROUTE_ENTRY_NODES = {
    Route.PROFILE.value: "profile",
    Route.AI_ENGINEERING.value: "retrieve",
    Route.OFF_TOPIC.value: "off_topic",
}


def build_graph(
    cv_text: str, retriever: Retriever, llm: Llm, top_k: int = 5
) -> CompiledStateGraph:
    """Assemble le graphe.

    On l'exécute avec invoke({"question": ...}, {"configurable": {"thread_id": ...}}) :
    le thread_id désigne la conversation à laquelle la question appartient.
    """
    builder = StateGraph(RagState)

    builder.add_node("classify", ClassifyNode(cv_text, llm))
    builder.add_node("profile", ProfileNode(cv_text, llm))
    builder.add_node("retrieve", RetrieveNode(retriever, top_k))
    builder.add_node("ai_engineering", AiEngineeringNode(cv_text, llm))
    builder.add_node("off_topic", OffTopicNode())

    builder.add_edge(START, "classify")
    # Prompt routing : la route posée par classify désigne la branche suivie.
    builder.add_conditional_edges("classify", _read_route, ROUTE_ENTRY_NODES)
    builder.add_edge("profile", END)
    builder.add_edge("retrieve", "ai_engineering")
    builder.add_edge("ai_engineering", END)
    builder.add_edge("off_topic", END)

    # Les conversations vivent dans la mémoire du processus : elles sont perdues à son arrêt.
    # LangGraph ne relit de l'état sauvegardé que les types du projet déclarés ici.
    serializer = JsonPlusSerializer(allowed_msgpack_modules=[(Route.__module__, "Route")])
    return builder.compile(checkpointer=InMemorySaver(serde=serializer))


def save_graph_image(graph: CompiledStateGraph, image_path: Path) -> None:
    """Dessine le graphe dans un fichier PNG (rendu par le service en ligne mermaid.ink)."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    # Le service répond parfois lentement : quelques nouvelles tentatives suffisent.
    image = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    image_path.write_bytes(image)


def _read_route(state: RagState) -> str:
    return state["route"].value
