"""État qui circule dans le graphe, et routes entre lesquelles une question est classée."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, TypedDict


class Exchange(TypedDict):
    """Un tour de la conversation : la question du recruteur et la réponse de l'assistant."""

    question: str
    answer: str


class Route(Enum):
    """Traitement appliqué à une question."""

    # Question sur Guillaume ou sur son CV : le CV suffit pour répondre.
    PROFILE = "profile"
    # Question sur une notion d'IA engineering : le CV et des extraits d'ouvrages.
    AI_ENGINEERING = "ai_engineering"
    # Tout le reste : refus poli, sans appeler le modèle de réponse.
    OFF_TOPIC = "off_topic"


class RagState(TypedDict, total=False):
    """Chaque nœud lit l'état et renvoie les clés qu'il y ajoute.

    L'état est conservé d'une question à l'autre d'une même conversation : chaque clé est
    remplacée par sa nouvelle valeur, sauf history, qui s'allonge.
    """

    # Fournie à l'entrée du graphe.
    question: str
    # Posées par le nœud classify. La langue est nommée en anglais : « French », « English »…
    route: Route
    language: str
    # La question reformulée pour se comprendre sans la conversation : « Et en production ? »
    # devient « Comment évaluer un RAG en production ? ».
    standalone_question: str
    # Posés par le nœud retrieve : le texte des extraits d'ouvrages trouvés.
    passages: list[str]
    # Posée par le nœud qui répond : profile, ai_engineering ou off_topic.
    answer: str
    # Les tours précédents. Le nœud qui répond y ajoute le sien (operator.add concatène).
    history: Annotated[list[Exchange], operator.add]
