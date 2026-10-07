from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from cv_assistant.rag import (
    CvNotFoundError,
    Exchange,
    Retriever,
    Route,
    build_graph,
    load_cv,
)
from cv_assistant.rag.nodes import (
    AiEngineeringNode,
    ClassifyNode,
    OffTopicNode,
    ProfileNode,
    RetrieveNode,
    ai_engineering,
    classify,
    profile,
)
from cv_assistant.rag.llm import HISTORY_LIMIT, build_input
from cv_assistant.rag.nodes.off_topic import (
    OFF_TOPIC_ANSWER_ENGLISH,
    OFF_TOPIC_ANSWER_FRENCH,
)

CV_TEXT = "Guillaume Legall, développeur IA"
PREVIOUS_EXCHANGE: Exchange = {"question": "Connaît-il le RAG ?", "answer": "Oui."}
CONVERSATION = {"configurable": {"thread_id": "test"}}
GUIDE_PASSAGE = "premier"
BOOK_PASSAGE = "deuxième"


class FakeCollection:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.queries: list[tuple[list[str], int]] = []

    def query(
        self, query_texts: list[str], n_results: int, include: list[str]
    ) -> dict[str, list[list[str]]]:
        self.queries.append((query_texts, n_results))
        return {"documents": [self.documents[:n_results]]}


class FakeRetriever:
    def __init__(self, passages: list[str]) -> None:
        self.passages = passages
        self.searches: list[tuple[str, int]] = []

    def search(self, question: str, top_k: int = 5) -> list[str]:
        self.searches.append((question, top_k))
        return self.passages[:top_k]


class FakeLlm:
    """Faux modèle : classe toujours la question de la même façon et rédige « la réponse »."""

    def __init__(self, route: Route = Route.PROFILE, language: str = "French") -> None:
        self.classification = {
            "route": route.value,
            "language": language,
            "standalone_question": "la question reformulée",
        }
        self.json_calls: list[tuple[str, str, list[Exchange], dict[str, Any]]] = []
        self.calls: list[tuple[str, str, list[Exchange]]] = []

    def complete(self, instructions: str, message: str, history: list[Exchange]) -> str:
        self.calls.append((instructions, message, history))
        return "la réponse"

    def complete_json(
        self, instructions: str, message: str, history: list[Exchange], schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.json_calls.append((instructions, message, history, schema))
        return self.classification


class LoadCvTests(unittest.TestCase):
    def test_pages_are_joined_and_blank_ones_are_dropped(self) -> None:
        with patch(
            "cv_assistant.rag.cv.read_pdf_sections",
            return_value=["  page un\n", "", "page deux"],
        ):
            self.assertEqual(load_cv(Path("cv.pdf")), "page un\n\npage deux")

    def test_missing_or_textless_cv_is_reported_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(CvNotFoundError):
                load_cv(Path(temporary_directory) / "absent.pdf")
        with (
            patch(
                "cv_assistant.rag.cv.read_pdf_sections",
                return_value=["", "  "],
            ),
            self.assertRaises(CvNotFoundError),
        ):
            load_cv(Path("cv.pdf"))


class RetrieverTests(unittest.TestCase):
    def test_search_returns_the_closest_chunks_in_order(self) -> None:
        collection = FakeCollection(documents=["premier", "deuxième", "troisième"])

        passages = Retriever(collection).search("une question", top_k=2)

        self.assertEqual(collection.queries, [(["une question"], 2)])
        self.assertEqual(passages, [GUIDE_PASSAGE, BOOK_PASSAGE])


class BuildInputTests(unittest.TestCase):
    def test_previous_exchanges_come_before_the_message(self) -> None:
        self.assertEqual(
            build_input("le message", [PREVIOUS_EXCHANGE]),
            [
                {"role": "user", "content": "Connaît-il le RAG ?"},
                {"role": "assistant", "content": "Oui."},
                {"role": "user", "content": "le message"},
            ],
        )

    def test_only_the_last_exchanges_are_sent(self) -> None:
        history: list[Exchange] = [
            {"question": f"question {number}", "answer": f"réponse {number}"}
            for number in range(HISTORY_LIMIT + 2)
        ]

        messages = build_input("le message", history)

        self.assertEqual(len(messages), 2 * HISTORY_LIMIT + 1)
        self.assertEqual(messages[0], {"role": "user", "content": "question 2"})


class NodeTests(unittest.TestCase):
    def test_classify_node_sets_the_route_the_language_and_the_standalone_question(self) -> None:
        llm = FakeLlm(route=Route.AI_ENGINEERING, language="English")

        update = ClassifyNode(CV_TEXT, llm)(
            {"question": "What is RAG?", "history": [PREVIOUS_EXCHANGE]}
        )

        self.assertEqual(
            update,
            {
                "route": Route.AI_ENGINEERING,
                "language": "English",
                "standalone_question": "la question reformulée",
            },
        )
        self.assertEqual(
            llm.json_calls,
            [
                (
                    classify.INSTRUCTIONS,
                    f"<cv>\n{CV_TEXT}\n</cv>\n\nQuestion à analyser : What is RAG?",
                    [PREVIOUS_EXCHANGE],
                    classify.ANSWER_SCHEMA,
                )
            ],
        )

    def test_profile_node_sends_the_cv_the_question_and_the_language(self) -> None:
        llm = FakeLlm()

        update = ProfileNode(CV_TEXT, llm)(
            {
                "question": "Quel est son parcours ?",
                "language": "French",
                "history": [PREVIOUS_EXCHANGE],
            }
        )

        self.assertEqual(
            update,
            {
                "answer": "la réponse",
                "history": [{"question": "Quel est son parcours ?", "answer": "la réponse"}],
            },
        )
        self.assertEqual(
            llm.calls,
            [
                (
                    profile.INSTRUCTIONS,
                    f"<cv>\n{CV_TEXT}\n</cv>\n\n"
                    "Question : Quel est son parcours ?\n\n"
                    "Answer in French.",
                    [PREVIOUS_EXCHANGE],
                )
            ],
        )

    def test_retrieve_node_searches_with_the_standalone_question(self) -> None:
        retriever = FakeRetriever([GUIDE_PASSAGE, BOOK_PASSAGE])

        update = RetrieveNode(retriever, top_k=1)(
            {"question": "Et en production ?", "standalone_question": "Le RAG en production ?"}
        )

        self.assertEqual(update, {"passages": [GUIDE_PASSAGE]})
        self.assertEqual(retriever.searches, [("Le RAG en production ?", 1)])

    def test_ai_engineering_node_adds_the_retrieved_passages_to_the_cv(self) -> None:
        llm = FakeLlm()

        update = AiEngineeringNode(CV_TEXT, llm)(
            {
                "question": "What does he know about RAG?",
                "language": "English",
                "passages": [GUIDE_PASSAGE, BOOK_PASSAGE],
            }
        )

        self.assertEqual(
            update,
            {
                "answer": "la réponse",
                "history": [
                    {"question": "What does he know about RAG?", "answer": "la réponse"}
                ],
            },
        )
        self.assertEqual(
            llm.calls,
            [
                (
                    ai_engineering.INSTRUCTIONS,
                    f"<cv>\n{CV_TEXT}\n</cv>\n\n"
                    "<notes_de_lecture>\n"
                    "<extrait>\npremier\n</extrait>\n\n"
                    "<extrait>\ndeuxième\n</extrait>\n"
                    "</notes_de_lecture>\n\n"
                    "Question : What does he know about RAG?\n\n"
                    "Answer in English.",
                    [],
                )
            ],
        )

    def test_off_topic_node_declines_in_french_or_in_english(self) -> None:
        node = OffTopicNode()

        for language, answer in [
            ("French", OFF_TOPIC_ANSWER_FRENCH),
            ("English", OFF_TOPIC_ANSWER_ENGLISH),
            ("Spanish", OFF_TOPIC_ANSWER_ENGLISH),
        ]:
            self.assertEqual(
                node({"question": "la météo ?", "language": language}),
                {"answer": answer, "history": [{"question": "la météo ?", "answer": answer}]},
            )


class GraphTests(unittest.TestCase):
    def run_graph(self, route: Route) -> tuple[list[str], dict[str, Any], FakeRetriever, FakeLlm]:
        """Exécute le graphe et renvoie les nœuds traversés, l'état final et les faux services."""
        retriever = FakeRetriever([GUIDE_PASSAGE])
        llm = FakeLlm(route=route)
        graph = build_graph(CV_TEXT, retriever, llm, top_k=3)

        visited: list[str] = []
        state: dict[str, Any] = {}
        for step in graph.stream({"question": "une question"}, CONVERSATION, stream_mode="updates"):
            for node, update in step.items():
                visited.append(node)
                state.update(update)
        return visited, state, retriever, llm

    def test_graph_has_one_branch_per_route(self) -> None:
        graph = build_graph(CV_TEXT, FakeRetriever([]), FakeLlm())

        edges = {(edge.source, edge.target, edge.data) for edge in graph.get_graph().edges}

        self.assertEqual(
            edges,
            {
                ("__start__", "classify", None),
                # LangGraph n'étiquette une arête que si la route et le nœud ont des noms différents.
                ("classify", "profile", None),
                ("classify", "retrieve", "ai_engineering"),
                ("classify", "off_topic", None),
                ("profile", "__end__", None),
                ("retrieve", "ai_engineering", None),
                ("ai_engineering", "__end__", None),
                ("off_topic", "__end__", None),
            },
        )

    def test_profile_question_is_answered_without_searching_the_books(self) -> None:
        visited, state, retriever, llm = self.run_graph(Route.PROFILE)

        self.assertEqual(visited, ["classify", "profile"])
        self.assertEqual(state["answer"], "la réponse")
        self.assertEqual(retriever.searches, [])
        self.assertEqual(llm.calls[0][0], profile.INSTRUCTIONS)

    def test_ai_engineering_question_is_answered_after_searching_the_books(self) -> None:
        visited, state, retriever, llm = self.run_graph(Route.AI_ENGINEERING)

        self.assertEqual(visited, ["classify", "retrieve", "ai_engineering"])
        self.assertEqual(state["passages"], [GUIDE_PASSAGE])
        self.assertEqual(state["answer"], "la réponse")
        self.assertEqual(retriever.searches, [("la question reformulée", 3)])
        self.assertEqual(llm.calls[0][0], ai_engineering.INSTRUCTIONS)

    def test_off_topic_question_is_declined_without_writing_an_answer(self) -> None:
        visited, state, retriever, llm = self.run_graph(Route.OFF_TOPIC)

        self.assertEqual(visited, ["classify", "off_topic"])
        self.assertEqual(state["answer"], OFF_TOPIC_ANSWER_FRENCH)
        self.assertEqual(retriever.searches, [])
        self.assertEqual(llm.calls, [])

    def test_second_question_of_a_conversation_receives_the_first_exchange(self) -> None:
        llm = FakeLlm(route=Route.AI_ENGINEERING)
        graph = build_graph(CV_TEXT, FakeRetriever([GUIDE_PASSAGE]), llm)
        first_exchange = {"question": "Connaît-il le RAG ?", "answer": "la réponse"}

        graph.invoke({"question": "Connaît-il le RAG ?"}, CONVERSATION)
        state = graph.invoke({"question": "Et en production ?"}, CONVERSATION)

        self.assertEqual(llm.json_calls[1][2], [first_exchange])
        self.assertEqual(llm.calls[1][2], [first_exchange])
        self.assertEqual(
            state["history"],
            [first_exchange, {"question": "Et en production ?", "answer": "la réponse"}],
        )

    def test_conversations_do_not_share_their_history(self) -> None:
        llm = FakeLlm()
        graph = build_graph(CV_TEXT, FakeRetriever([]), llm)

        graph.invoke({"question": "Bonjour"}, {"configurable": {"thread_id": "a"}})
        graph.invoke({"question": "Hello"}, {"configurable": {"thread_id": "b"}})

        self.assertEqual(llm.calls[1][2], [])


if __name__ == "__main__":
    unittest.main()
