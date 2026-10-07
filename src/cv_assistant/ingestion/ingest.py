"""Ingest des PDF et des EPUB dans ChromaDB.

Pour chaque fichier : lecture du texte par section (page ou chapitre), découpage en
chunks, puis remplacement des chunks du fichier dans la collection. ChromaDB vectorise
les chunks à l'ajout, en appelant la fonction d'embedding de la collection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from chromadb.api.models.Collection import Collection
from pypdf.errors import PdfReadError

from cv_assistant.embeddings import MAX_BATCH_CHARACTERS, MAX_BATCH_TEXTS
from cv_assistant.ingestion.chunking import split_text, validate_chunking
from cv_assistant.ingestion.epub import EpubReadError, read_epub_sections
from cv_assistant.ingestion.pdf import read_pdf_sections
from cv_assistant.vector_store import DEFAULT_COLLECTION, open_collection


@dataclass(frozen=True)
class IngestSummary:
    pdf_count: int
    epub_count: int
    chunk_count: int
    errors: tuple[Path, ...]


@dataclass(frozen=True)
class DocumentFormat:
    """Décrit comment lire un type de fichier et nommer ses sections."""

    suffix: str
    section_key: str
    read_sections: Callable[[Path], list[str]]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    section_number: int


@dataclass(frozen=True)
class _DirectoryResult:
    file_count: int
    chunk_count: int
    errors: list[Path]


PDF_FORMAT = DocumentFormat(
    suffix=".pdf", section_key="page", read_sections=read_pdf_sections
)
EPUB_FORMAT = DocumentFormat(
    suffix=".epub", section_key="chapter", read_sections=read_epub_sections
)


def ingest_documents(
    pdf_dir: Path,
    epub_dir: Path,
    chroma_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> IngestSummary:
    validate_chunking(chunk_size, chunk_overlap)

    for directory in (pdf_dir, epub_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with open_collection(chroma_dir, collection_name, create=True) as collection:
        pdfs = _index_directory(collection, pdf_dir, PDF_FORMAT, chunk_size, chunk_overlap)
        epubs = _index_directory(collection, epub_dir, EPUB_FORMAT, chunk_size, chunk_overlap)

    return IngestSummary(
        pdf_count=pdfs.file_count,
        epub_count=epubs.file_count,
        chunk_count=pdfs.chunk_count + epubs.chunk_count,
        errors=tuple(pdfs.errors + epubs.errors),
    )


def _index_directory(
    collection: Collection,
    directory: Path,
    document_format: DocumentFormat,
    chunk_size: int,
    chunk_overlap: int,
) -> _DirectoryResult:
    """Indexe tous les fichiers du dossier qui sont au format donné."""
    file_count = 0
    chunk_count = 0
    errors: list[Path] = []

    for file_path in _find_files(directory, document_format.suffix):
        try:
            sections = document_format.read_sections(file_path)
        except (OSError, PdfReadError, EpubReadError) as error:
            print(f"Erreur de lecture de {file_path}: {error}")
            errors.append(file_path)
            continue

        source = str(file_path.resolve())
        chunks = _build_chunks(source, sections, chunk_size, chunk_overlap)
        if not chunks:
            print(f"Aucun texte extractible dans {file_path}.")
        _replace_chunks(collection, source, document_format.section_key, chunks)

        file_count += 1
        chunk_count += len(chunks)

    return _DirectoryResult(file_count, chunk_count, errors)


def _find_files(directory: Path, suffix: str) -> list[Path]:
    """Fichiers du dossier (sous-dossiers compris) portant cette extension, triés par chemin."""
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() == suffix
    ]
    return sorted(files, key=lambda path: str(path).casefold())


def _build_chunks(
    source: str, sections: list[str], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Découpe chaque section en chunks.

    L'identifiant dépend du fichier, de la section et de la position du chunk : il reste
    donc le même d'un ingest à l'autre.
    """
    chunks: list[Chunk] = []
    for section_number, section_text in enumerate(sections, start=1):
        texts = split_text(section_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_index, text in enumerate(texts):
            chunk_id = str(uuid5(NAMESPACE_URL, f"{source}:{section_number}:{chunk_index}"))
            chunks.append(Chunk(id=chunk_id, text=text, section_number=section_number))
    return chunks


def _replace_chunks(
    collection: Collection, source: str, section_key: str, chunks: list[Chunk]
) -> None:
    """Supprime les anciens chunks du fichier source, puis ajoute les nouveaux."""
    collection.delete(where={"source": source})
    for batch in _batch_chunks(chunks):
        collection.add(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[{"source": source, section_key: chunk.section_number} for chunk in batch],
        )


def _batch_chunks(chunks: list[Chunk]) -> Iterator[list[Chunk]]:
    """Regroupe les chunks en lots qui tiennent chacun dans une requête d'embedding."""
    batch: list[Chunk] = []
    characters = 0
    for chunk in chunks:
        is_full = (
            len(batch) >= MAX_BATCH_TEXTS
            or characters + len(chunk.text) > MAX_BATCH_CHARACTERS
        )
        if batch and is_full:
            yield batch
            batch = []
            characters = 0
        batch.append(chunk)
        characters += len(chunk.text)
    if batch:
        yield batch
