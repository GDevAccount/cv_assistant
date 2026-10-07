from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cv_assistant.vector_store import CollectionNotFoundError, open_collection


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def get_collection(self, name: str, embedding_function: object) -> str:
        self.calls.append(("get", name))
        return "la collection"

    def get_or_create_collection(
        self, name: str, metadata: dict[str, str], embedding_function: object
    ) -> str:
        self.calls.append(("get_or_create", name))
        return "la collection"

    def close(self) -> None:
        self.closed = True


class OpenCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Évite d'exiger une clé d'API et d'appeler OpenAI pendant les tests.
        patcher = patch("cv_assistant.vector_store.build_embedding_function", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_search_opens_the_existing_collection_and_closes_the_client(self) -> None:
        client = FakeClient()

        with (
            tempfile.TemporaryDirectory() as chroma_dir,
            patch("cv_assistant.vector_store.chromadb.PersistentClient", return_value=client),
            open_collection(Path(chroma_dir), "ouvrages") as collection,
        ):
            self.assertEqual(collection, "la collection")
            self.assertFalse(client.closed)

        self.assertEqual(client.calls, [("get", "ouvrages")])
        self.assertTrue(client.closed)

    def test_ingest_creates_the_directory_and_the_collection(self) -> None:
        client = FakeClient()

        with tempfile.TemporaryDirectory() as temporary_directory:
            chroma_dir = Path(temporary_directory) / "absent"
            with (
                patch("cv_assistant.vector_store.chromadb.PersistentClient", return_value=client),
                open_collection(chroma_dir, "ouvrages", create=True),
            ):
                self.assertTrue(chroma_dir.is_dir())

        self.assertEqual(client.calls, [("get_or_create", "ouvrages")])
        self.assertTrue(client.closed)

    def test_missing_collection_asks_to_run_the_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "vide").mkdir()

            for chroma_dir in (root / "absent", root / "vide"):
                with self.assertRaises(CollectionNotFoundError):
                    with open_collection(chroma_dir):
                        pass
            self.assertFalse((root / "absent").exists())


if __name__ == "__main__":
    unittest.main()
