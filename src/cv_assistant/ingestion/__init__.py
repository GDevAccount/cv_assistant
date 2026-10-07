"""Ingestion des documents (PDF, EPUB) dans ChromaDB."""

from cv_assistant.ingestion.ingest import (
    IngestSummary,
    ingest_documents,
)

__all__ = ["IngestSummary", "ingest_documents"]
