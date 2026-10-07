from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from pypdf.errors import PdfReadError

from cv_assistant.embeddings import (
    MissingApiKeyError,
    build_embedding_function,
)
from cv_assistant.ingestion.chunking import split_text
from cv_assistant.ingestion.epub import read_epub_sections
from cv_assistant.ingestion.ingest import (
    Chunk,
    _batch_chunks,
    ingest_documents,
)

CONTAINER_XML = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

CONTENT_OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="second" href="text/chapitre%202.xhtml" media-type="application/xhtml+xml"/>
    <item id="first" href="text/chapitre1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="first"/>
    <itemref idref="second"/>
  </spine>
</package>"""


def write_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", CONTAINER_XML)
        archive.writestr("OEBPS/content.opf", CONTENT_OPF)
        archive.writestr(
            "OEBPS/text/chapitre1.xhtml",
            "<html><head><title>Titre ignoré</title><style>p { color: red; }</style></head>"
            "<body><h1>Chapitre un</h1><p>Premier <em>para</em>graphe.</p></body></html>",
        )
        archive.writestr(
            "OEBPS/text/chapitre 2.xhtml",
            "<html><body><p>Deuxième chapitre.</p></body></html>",
        )


class FakePage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakeCollection:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, dict[str, str | int]]] = {}

    def delete(self, where: dict[str, str]) -> None:
        source = where["source"]
        self.documents = {
            identifier: document
            for identifier, document in self.documents.items()
            if document[1]["source"] != source
        }

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str | int]],
    ) -> None:
        self.documents.update(
            {
                identifier: (document, metadata)
                for identifier, document, metadata in zip(
                    ids, documents, metadatas, strict=True
                )
            }
        )


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.closed = False

    def get_or_create_collection(
        self, name: str, metadata: dict[str, str], embedding_function: object
    ) -> FakeCollection:
        return self.collection

    def close(self) -> None:
        self.closed = True


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        # Évite d'exiger une clé d'API et d'appeler OpenAI pendant les tests.
        patcher = patch(
            "cv_assistant.vector_store.build_embedding_function",
            return_value=None,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_split_text_respects_chunk_size_and_keeps_overlap(self) -> None:
        chunks = split_text("one two three four five six", chunk_size=12, chunk_overlap=4)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks))
        self.assertIn("three", chunks[1])

    def test_ingest_is_recursive_and_reruns_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_dir = root / "pdfs"
            nested_dir = pdf_dir / "nested"
            nested_dir.mkdir(parents=True)
            (pdf_dir / "first.pdf").touch()
            (nested_dir / "second.PDF").touch()
            collection = FakeCollection()
            client = FakeClient(collection)

            def read_pdf(path: Path) -> object:
                if path.name == "first.pdf":
                    return type("Reader", (), {"pages": [FakePage("un texte de test")]})()
                return type(
                    "Reader",
                    (),
                    {"pages": [FakePage("un autre document de test")]},
                )()

            with (
                patch(
                    "cv_assistant.vector_store.chromadb.PersistentClient",
                    return_value=client,
                ),
                patch(
                    "cv_assistant.ingestion.pdf.PdfReader",
                    side_effect=read_pdf,
                ),
            ):
                first = ingest_documents(
                    pdf_dir, root / "epubs", root / "chroma", chunk_size=10, chunk_overlap=2
                )
                original_ids = set(collection.documents)
                second = ingest_documents(
                    pdf_dir, root / "epubs", root / "chroma", chunk_size=10, chunk_overlap=2
                )

            self.assertEqual(first.pdf_count, 2)
            self.assertEqual(second.chunk_count, first.chunk_count)
            self.assertEqual(set(collection.documents), original_ids)
            self.assertTrue(client.closed)
            self.assertEqual(
                {metadata["page"] for _, metadata in collection.documents.values()},
                {1},
            )

    def test_chunks_are_added_in_batches_that_fit_one_embedding_request(self) -> None:
        chunks = [Chunk(id=str(index), text="abcd", section_number=1) for index in range(5)]

        with patch(
            "cv_assistant.ingestion.ingest.MAX_BATCH_CHARACTERS", 10
        ):
            by_characters = list(_batch_chunks(chunks))
        with patch("cv_assistant.ingestion.ingest.MAX_BATCH_TEXTS", 3):
            by_count = list(_batch_chunks(chunks))

        self.assertEqual([len(batch) for batch in by_characters], [2, 2, 1])
        self.assertEqual([len(batch) for batch in by_count], [3, 2])
        self.assertEqual([chunk for batch in by_characters for chunk in batch], chunks)

    def test_missing_api_key_is_reported_clearly(self) -> None:
        with patch.dict("os.environ", clear=True), self.assertRaises(MissingApiKeyError):
            build_embedding_function()

    def test_unreadable_pdf_is_reported_without_stopping_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            broken_pdf = pdf_dir / "broken.pdf"
            valid_pdf = pdf_dir / "valid.pdf"
            broken_pdf.touch()
            valid_pdf.touch()
            collection = FakeCollection()
            client = FakeClient(collection)

            def read_pdf(path: Path) -> object:
                if path.name == "broken.pdf":
                    raise PdfReadError("PDF invalide")
                return type("Reader", (), {"pages": [FakePage("texte lisible")]})()

            with (
                patch(
                    "cv_assistant.vector_store.chromadb.PersistentClient",
                    return_value=client,
                ),
                patch(
                    "cv_assistant.ingestion.pdf.PdfReader",
                    side_effect=read_pdf,
                ),
            ):
                summary = ingest_documents(pdf_dir, root / "epubs", root / "chroma")

            self.assertEqual(summary.pdf_count, 1)
            self.assertEqual(summary.errors, (broken_pdf,))
            self.assertGreater(summary.chunk_count, 0)
            self.assertTrue(client.closed)

    def test_read_epub_sections_follows_spine_order_and_strips_markup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            epub_path = Path(temporary_directory) / "livre.epub"
            write_epub(epub_path)

            sections = [" ".join(section.split()) for section in read_epub_sections(epub_path)]

        self.assertEqual(
            sections,
            ["Chapitre un Premier paragraphe.", "Deuxième chapitre."],
        )

    def test_epubs_are_ingested_and_broken_ones_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            epub_dir = root / "epubs"
            nested_dir = epub_dir / "nested"
            nested_dir.mkdir(parents=True)
            write_epub(nested_dir / "livre.EPUB")
            broken_epub = epub_dir / "broken.epub"
            broken_epub.write_text("pas un zip", encoding="utf-8")
            collection = FakeCollection()
            client = FakeClient(collection)

            with patch(
                "cv_assistant.vector_store.chromadb.PersistentClient",
                return_value=client,
            ):
                first = ingest_documents(root / "pdfs", epub_dir, root / "chroma")
                original_ids = set(collection.documents)
                second = ingest_documents(root / "pdfs", epub_dir, root / "chroma")

            self.assertEqual(first.pdf_count, 0)
            self.assertEqual(first.epub_count, 1)
            self.assertEqual(first.chunk_count, 2)
            self.assertEqual(first.errors, (broken_epub,))
            self.assertEqual(second.chunk_count, first.chunk_count)
            self.assertEqual(set(collection.documents), original_ids)
            self.assertEqual(
                {metadata["chapter"] for _, metadata in collection.documents.values()},
                {1, 2},
            )


if __name__ == "__main__":
    unittest.main()
