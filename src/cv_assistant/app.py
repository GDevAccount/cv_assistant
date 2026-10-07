"""Interface web (Streamlit) : les documents dans le volet de gauche, la conversation au centre.

Se lance depuis la racine du projet :

    uv run streamlit run src/cv_assistant/app.py
"""

from __future__ import annotations

import hmac
import os
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from cv_assistant.embeddings import MissingApiKeyError
from cv_assistant.ingestion import ingest_documents
from cv_assistant.rag import CvNotFoundError, Llm, Retriever, build_graph, load_cv
from cv_assistant.vector_store import CollectionNotFoundError, open_collection

# Cherchés dans le dossier courant, comme pour la ligne de commande.
CV_PATH = Path("data/cv/cv.pdf")
PDF_DIR = Path("data/pdfs")
EPUB_DIR = Path("data/epubs")
CHROMA_DIR = Path("data/chroma")


def _close_assistant(assistant: tuple[CompiledStateGraph, ExitStack]) -> None:
    assistant[1].close()


@st.cache_resource(show_spinner=False, on_release=_close_assistant)
def _open_assistant() -> tuple[CompiledStateGraph, ExitStack]:
    """Assemble le graphe une seule fois pour tout le processus, avec sa base ChromaDB ouverte.

    Le graphe porte la mémoire des conversations : il doit survivre aux relectures du
    script par Streamlit. La pile referme la base quand le graphe est abandonné.
    """
    cv_text = load_cv(CV_PATH)
    stack = ExitStack()
    collection = stack.enter_context(open_collection(CHROMA_DIR))
    return build_graph(cv_text, Retriever(collection), Llm()), stack


def _password_is_given(variable: str) -> bool:
    """Demande le mot de passe rangé dans `variable`, une fois par session du navigateur.

    APP_PASSWORD ouvre la page, ADMIN_PASSWORD le volet de dépôt des documents.
    """
    if st.session_state.get(variable):
        return True

    expected = os.environ.get(variable)
    if not expected:
        # Sans mot de passe défini, l'accès reste fermé plutôt qu'ouvert à tous.
        st.error(f"La variable {variable} est absente : ajoutez-la au fichier .env.")
        return False

    with st.form(f"form_{variable}"):
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Entrer")
    if submitted:
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state[variable] = True
            st.rerun()
        st.error("Mot de passe incorrect.")
    return False


def _start_conversation() -> None:
    """Chaque session du navigateur a son propre thread_id, donc sa propre mémoire."""
    st.session_state.thread_id = str(uuid4())
    st.session_state.messages = []


def _save_documents(cv_file, book_files) -> None:
    """Enregistre les fichiers déposés, puis indexe les ouvrages s'il y en a de nouveaux."""
    if cv_file is None and not book_files:
        st.warning("Aucun fichier à enregistrer.")
        return

    # Le graphe a lu l'ancien CV et tient la base ouverte : il sera réassemblé ensuite,
    # avec une mémoire vide. La conversation affichée repart donc de zéro elle aussi.
    _open_assistant.clear()
    _start_conversation()

    if cv_file is not None:
        CV_PATH.parent.mkdir(parents=True, exist_ok=True)
        CV_PATH.write_bytes(cv_file.getvalue())
        st.success("CV enregistré.")

    if book_files:
        for book_file in book_files:
            # Path(...).name écarte tout dossier glissé dans le nom du fichier.
            file_name = Path(book_file.name).name
            directory = EPUB_DIR if file_name.casefold().endswith(".epub") else PDF_DIR
            directory.mkdir(parents=True, exist_ok=True)
            (directory / file_name).write_bytes(book_file.getvalue())

        with st.spinner("Indexation des ouvrages…"):
            summary = ingest_documents(PDF_DIR, EPUB_DIR, CHROMA_DIR)
        st.success(
            f"{summary.pdf_count} PDF et {summary.epub_count} EPUB indexés "
            f"({summary.chunk_count} chunks)."
        )
        for error_path in summary.errors:
            st.error(f"Lecture impossible : {error_path.name}")


def _show_sidebar() -> None:
    with st.sidebar:
        if st.button("Nouvelle conversation"):
            _start_conversation()

        st.header("Documents")
        st.caption("Réservé à Guillaume.")
        if not _password_is_given("ADMIN_PASSWORD"):
            return

        with st.form("documents", clear_on_submit=True):
            cv_file = st.file_uploader("CV (PDF)", type="pdf")
            book_files = st.file_uploader(
                "Ouvrages (PDF ou EPUB)", type=["pdf", "epub"], accept_multiple_files=True
            )
            submitted = st.form_submit_button("Enregistrer")
        st.caption(
            "Ajouter un ouvrage réindexe tous les ouvrages : leur texte est envoyé à "
            "OpenAI, et l'appel est facturé. Enregistrer un document recommence la conversation."
        )
        if submitted:
            try:
                _save_documents(cv_file, book_files)
            except MissingApiKeyError as error:
                st.error(str(error))

        st.subheader("Déjà en place")
        st.write("CV : " + ("✅ " + CV_PATH.name if CV_PATH.is_file() else "aucun"))
        books = sorted(
            path.name
            for directory in (PDF_DIR, EPUB_DIR)
            for path in directory.rglob("*")
            if path.suffix.casefold() in (".pdf", ".epub")
        )
        st.write(f"Ouvrages : {len(books)}")
        for book in books:
            st.caption(book)


def main() -> None:
    st.set_page_config(page_title="Assistant d'entretien — Guillaume Legall", page_icon="💬")
    # Les variables déjà définies dans l'environnement priment sur celles du fichier .env.
    load_dotenv(Path(".env"))
    if not _password_is_given("APP_PASSWORD"):
        return
    if "thread_id" not in st.session_state:
        _start_conversation()

    _show_sidebar()

    st.title("Assistant d'entretien de Guillaume Legall")
    st.caption("Posez vos questions sur son parcours ou sur l'IA engineering.")

    try:
        graph, _ = _open_assistant()
    except (MissingApiKeyError, CollectionNotFoundError, CvNotFoundError) as error:
        st.info(f"{error}\n\nAjoutez le CV et les ouvrages dans le volet de gauche.")
        return

    for role, text in st.session_state.messages:
        st.chat_message(role).markdown(text)

    question = st.chat_input("Votre question")
    if question:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Réflexion…"):
                state = graph.invoke(
                    {"question": question},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
            st.markdown(state["answer"])
        st.session_state.messages += [("user", question), ("assistant", state["answer"])]


if __name__ == "__main__":
    main()
