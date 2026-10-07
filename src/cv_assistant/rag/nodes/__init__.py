"""Nœuds du graphe, un fichier par nœud.

Tous ont la même forme : le constructeur reçoit ce dont le nœud a besoin, et l'appel
reçoit l'état courant puis renvoie les clés qu'il y ajoute. Chaque nœud porte ses propres
consignes ; persona.py contient le ton commun aux nœuds qui rédigent une réponse.
"""

from cv_assistant.rag.nodes.ai_engineering import AiEngineeringNode
from cv_assistant.rag.nodes.classify import ClassifyNode
from cv_assistant.rag.nodes.off_topic import OffTopicNode
from cv_assistant.rag.nodes.profile import ProfileNode
from cv_assistant.rag.nodes.retrieve import RetrieveNode

__all__ = ["AiEngineeringNode", "ClassifyNode", "OffTopicNode", "ProfileNode", "RetrieveNode"]
