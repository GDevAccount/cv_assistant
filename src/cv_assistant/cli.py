"""Ligne de commande : ingest, ask, chat et graph."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from cv_assistant.embeddings import MissingApiKeyError
from cv_assistant.ingestion import ingest_documents
from cv_assistant.rag import (
    CvNotFoundError,
    Llm,
    Retriever,
    build_graph,
    load_cv,
    save_graph_image,
)
from cv_assistant.vector_store import (
    DEFAULT_COLLECTION,
    CollectionNotFoundError,
    open_collection,
)

# Cherché dans le dossier courant, comme les dossiers de données par défaut.
ENV_FILE = Path(".env")
# La ligne de commande ne mène qu'une conversation par lancement.
CONVERSATION = {"configurable": {"thread_id": "cli"}}


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("La valeur doit être positive ou nulle.")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("La valeur doit être supérieure à zéro.")
    return parsed


def _add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("data/chroma"),
        help="Dossier de persistance ChromaDB (par défaut : data/chroma).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Nom de la collection ChromaDB (par défaut : {DEFAULT_COLLECTION}).",
    )


def _add_graph_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments nécessaires pour assembler le graphe : le CV et la collection des ouvrages."""
    parser.add_argument(
        "--cv",
        type=Path,
        default=Path("data/cv/cv.pdf"),
        help="CV au format PDF, envoyé en entier au modèle (par défaut : data/cv/cv.pdf).",
    )
    _add_collection_arguments(parser)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assistant qui répond aux recruteurs sur le profil de Guillaume Legall, "
            "à partir de son CV et d'ouvrages d'IA engineering indexés."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser(
        "ingest", help="Découpe les PDF et les EPUB et les indexe dans ChromaDB."
    )
    ingest_parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
        help="Dossier des PDF (par défaut : data/pdfs).",
    )
    ingest_parser.add_argument(
        "--epub-dir",
        type=Path,
        default=Path("data/epubs"),
        help="Dossier des EPUB (par défaut : data/epubs).",
    )
    _add_collection_arguments(ingest_parser)
    ingest_parser.add_argument(
        "--chunk-size",
        type=_positive_int,
        default=1000,
        help="Taille maximale des chunks en caractères (par défaut : 1000).",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=_non_negative_int,
        default=150,
        help="Chevauchement des chunks en caractères (par défaut : 150).",
    )
    ask_parser = commands.add_parser(
        "ask", help="Répond à la question d'un recruteur sur le profil de Guillaume Legall."
    )
    ask_parser.add_argument("question", help="Question à poser, entre guillemets.")
    chat_parser = commands.add_parser(
        "chat", help="Ouvre une conversation : l'assistant se souvient des questions précédentes."
    )
    for answering_parser in (ask_parser, chat_parser):
        _add_graph_arguments(answering_parser)
        answering_parser.add_argument(
            "--top-k",
            type=_positive_int,
            default=5,
            help="Nombre d'extraits d'ouvrages transmis au modèle (par défaut : 5).",
        )
    graph_parser = commands.add_parser(
        "graph", help="Dessine le graphe de l'assistant dans une image PNG."
    )
    graph_parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/graph.png"),
        help="Image à écrire (par défaut : docs/graph.png).",
    )
    _add_graph_arguments(graph_parser)
    arguments = parser.parse_args(argv)

    if arguments.command == "ingest" and arguments.chunk_overlap >= arguments.chunk_size:
        parser.error("--chunk-overlap doit être inférieur à --chunk-size.")

    # Les variables déjà définies dans l'environnement priment sur celles du fichier .env.
    load_dotenv(ENV_FILE)

    run_command = {
        "ingest": _run_ingest,
        "ask": _run_ask,
        "chat": _run_chat,
        "graph": _run_graph,
    }
    try:
        return run_command[arguments.command](arguments)
    except (MissingApiKeyError, CollectionNotFoundError, CvNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1


def _run_ingest(arguments: argparse.Namespace) -> int:
    summary = ingest_documents(
        pdf_dir=arguments.pdf_dir,
        epub_dir=arguments.epub_dir,
        chroma_dir=arguments.chroma_dir,
        collection_name=arguments.collection,
        chunk_size=arguments.chunk_size,
        chunk_overlap=arguments.chunk_overlap,
    )
    print(
        f"Ingest terminé : {summary.pdf_count} PDF traité(s), "
        f"{summary.epub_count} EPUB traité(s), "
        f"{summary.chunk_count} chunk(s) indexé(s), "
        f"{len(summary.errors)} erreur(s)."
    )
    return 1 if summary.errors else 0


def _run_ask(arguments: argparse.Namespace) -> int:
    with _open_graph(arguments, top_k=arguments.top_k) as graph:
        state = graph.invoke({"question": arguments.question}, CONVERSATION)
    print(state["answer"])
    return 0


def _run_chat(arguments: argparse.Namespace) -> int:
    with _open_graph(arguments, top_k=arguments.top_k) as graph:
        print("Posez vos questions. Une ligne vide ou Ctrl+C termine la conversation.")
        while True:
            try:
                question = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            state = graph.invoke({"question": question}, CONVERSATION)
            print(state["answer"])
    return 0


def _run_graph(arguments: argparse.Namespace) -> int:
    with _open_graph(arguments) as graph:
        save_graph_image(graph, arguments.output)
    print(f"Graphe dessiné dans {arguments.output}.")
    return 0


@contextmanager
def _open_graph(arguments: argparse.Namespace, top_k: int = 5) -> Iterator[CompiledStateGraph]:
    """Assemble le graphe avec ses composants, puis referme la base ChromaDB à la sortie."""
    cv_text = load_cv(arguments.cv)
    with open_collection(arguments.chroma_dir, arguments.collection) as collection:
        yield build_graph(cv_text, Retriever(collection), Llm(), top_k)


if __name__ == "__main__":
    raise SystemExit(main())
