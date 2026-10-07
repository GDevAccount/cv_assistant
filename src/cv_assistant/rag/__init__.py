"""Assistant qui répond aux recruteurs sur le profil de Guillaume Legall (RAG).

Le CV est envoyé en entier à chaque question ; les ouvrages d'IA engineering indexés par
l'ingest apportent les connaissances techniques.

Trois couches, qui ne dépendent que de celle du dessous :

- graph.py assemble le graphe LangGraph : les nœuds et les arêtes qui les relient.
  state.py décrit l'état qui circule de nœud en nœud ;
- nodes/ contient un fichier par nœud, chacun avec ses propres consignes ;
- llm.py, retriever.py et cv.py sont les seuls fichiers qui parlent à l'extérieur :
  le modèle de langage d'OpenAI, la collection ChromaDB et le fichier du CV.
"""

from cv_assistant.rag.cv import CvNotFoundError, load_cv
from cv_assistant.rag.graph import build_graph, save_graph_image
from cv_assistant.rag.llm import Llm
from cv_assistant.rag.retriever import Retriever
from cv_assistant.rag.state import Exchange, RagState, Route

__all__ = [
    "CvNotFoundError",
    "Exchange",
    "Llm",
    "RagState",
    "Retriever",
    "Route",
    "build_graph",
    "load_cv",
    "save_graph_image",
]
