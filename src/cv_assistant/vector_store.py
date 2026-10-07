"""Ouverture de la collection ChromaDB des ouvrages, pour l'ingest comme pour la recherche.

C'est le seul endroit qui crée le client ChromaDB : l'ingest et la recherche utilisent
ainsi forcément la même collection et le même modèle d'embedding.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from cv_assistant.embeddings import build_embedding_function

DEFAULT_COLLECTION = "pdf_documents"


class CollectionNotFoundError(RuntimeError):
    """La collection n'existe pas encore : l'ingest n'a pas été lancé."""


@contextmanager
def open_collection(
    chroma_dir: Path, collection_name: str = DEFAULT_COLLECTION, create: bool = False
) -> Iterator[Collection]:
    """Ouvre la collection, puis referme la base à la sortie du « with ».

    Avec create=True (l'ingest), la base et la collection sont créées si elles manquent.
    Sinon (la recherche), leur absence lève CollectionNotFoundError.
    """
    missing_collection = CollectionNotFoundError(
        f"La collection « {collection_name} » est introuvable dans {chroma_dir} : "
        "lancez d'abord l'ingest."
    )
    if create:
        chroma_dir.mkdir(parents=True, exist_ok=True)
    elif not chroma_dir.is_dir():
        raise missing_collection

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        # Les questions doivent être vectorisées par le même modèle que les documents.
        embedding_function = build_embedding_function()
        if create:
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_function,
            )
        else:
            try:
                collection = client.get_collection(
                    name=collection_name, embedding_function=embedding_function
                )
            except NotFoundError as error:
                raise missing_collection from error
        yield collection
    finally:
        client.close()
