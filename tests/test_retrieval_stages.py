"""Offline tests for retrieval stages and shared cross-encoder reranking."""

import unittest
from unittest.mock import Mock, patch

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.rag import retriever


class RetrievalStagesTests(unittest.TestCase):
    def setUp(self):
        self.a = Document(page_content="same text", metadata={"doc_id": "a"})
        self.b = Document(page_content="sparse b", metadata={"doc_id": "b"})
        self.dense_b = Document(page_content="dense b", metadata={"doc_id": "b"})
        self.c = Document(page_content="c", metadata={"doc_id": "c"})
        self.d = Document(page_content="same text", metadata={"doc_id": "d"})
        self.sparse_docs = [self.a, self.b, self.c]
        self.dense_docs = [self.dense_b, self.c, self.d]
        self.sparse = Mock(return_value=self.sparse_docs)
        self.dense = Mock(return_value=self.dense_docs)
        self.ensemble = EnsembleRetriever(
            retrievers=[RunnableLambda(self.sparse), RunnableLambda(self.dense)],
            weights=[0.25, 0.75],
            c=0,
            id_key="doc_id",
        )
        self.hybrid = retriever.HybridRetriever(
            {},
            {},
            {},
            reranker_model="mock-model",
            rerank_k=3,
            top_n=2,
            reranker_batch_size=4,
        )
        self.hybrid._retriever = self.ensemble
        self.encoder = self.enterContext(patch.object(retriever, "CrossEncoder"))
        self.encoder.return_value.predict.return_value = [0.2, 0.9, 0.9]
        self.enterContext(patch.object(retriever, "prepare_indexes"))

    def test_stages_invoke_each_child_once_and_use_real_rrf(self):
        self.hybrid.reranker_model = None
        with patch.object(self.hybrid, "_load", wraps=self.hybrid._load) as load:
            with patch.object(
                EnsembleRetriever,
                "weighted_reciprocal_rank",
                autospec=True,
                side_effect=EnsembleRetriever.weighted_reciprocal_rank,
            ) as fuse:
                with patch.object(EnsembleRetriever, "invoke") as invoke:
                    stages = self.hybrid.retrieve_stages("query")
        load.assert_called_once_with()
        self.sparse.assert_called_once_with("query")
        self.dense.assert_called_once_with("query")
        invoke.assert_not_called()
        fuse.assert_called_once_with(self.ensemble, [self.sparse_docs, self.dense_docs])
        self.assertEqual(set(stages), {"bm25", "faiss", "fusion"})
        self.assertIs(stages["bm25"], self.sparse_docs)
        self.assertIs(stages["faiss"], self.dense_docs)
        self.assertEqual(stages["fusion"], [self.b, self.c, self.a, self.d])
        self.assertIs(stages["fusion"][0], self.b)
        self.encoder.assert_not_called()

    def test_stages_expose_pool_and_rerank_the_same_fusion(self):
        with patch.object(self.hybrid, "rerank", wraps=self.hybrid.rerank) as rerank:
            stages = self.hybrid.retrieve_stages("query")
        self.assertEqual(
            set(stages), {"bm25", "faiss", "fusion", "reranker_candidates", "reranker"}
        )
        rerank.assert_called_once_with("query", stages["fusion"])
        self.assertIs(rerank.call_args.args[1], stages["fusion"])
        self.assertEqual(stages["fusion"], [self.b, self.c, self.a, self.d])
        self.assertEqual(stages["reranker_candidates"], [self.b, self.c, self.a])
        self.assertEqual(stages["reranker"], [self.c, self.a])
        self.encoder.assert_called_once_with(
            "mock-model", device="cpu", max_length=8192
        )
        self.encoder.return_value.predict.assert_called_once_with(
            [("query", "sparse b"), ("query", "c"), ("query", "same text")],
            batch_size=4,
            show_progress_bar=False,
        )
        self.sparse.assert_called_once_with("query")
        self.dense.assert_called_once_with("query")

    def test_retrieve_keeps_ensemble_invoke_then_rerank(self):
        documents = [self.d, self.a]
        expected = [self.a]
        calls = Mock()
        with patch.object(self.hybrid, "_load") as load:
            with patch.object(
                EnsembleRetriever, "invoke", return_value=documents
            ) as invoke:
                with patch.object(
                    self.hybrid, "rerank", return_value=expected
                ) as rerank:
                    calls.attach_mock(load, "load")
                    calls.attach_mock(invoke, "invoke")
                    calls.attach_mock(rerank, "rerank")
                    self.assertIs(self.hybrid.retrieve("query"), expected)
        self.assertEqual(
            [call[0] for call in calls.mock_calls], ["load", "invoke", "rerank"]
        )
        load.assert_called_once_with()
        invoke.assert_called_once_with("query")
        rerank.assert_called_once_with("query", documents)
        self.encoder.assert_not_called()

    def test_rerank_reuses_encoder_and_does_not_mutate_documents(self):
        documents = [self.b, self.c, self.a, self.d]
        for query in ("first", "second"):
            self.assertEqual(self.hybrid.rerank(query, documents), [self.c, self.a])
        self.assertEqual(documents, [self.b, self.c, self.a, self.d])
        self.encoder.assert_called_once()
        self.assertEqual(self.encoder.return_value.predict.call_count, 2)
        self.sparse.assert_not_called()
        self.dense.assert_not_called()

    def test_rerank_short_pool_returns_at_most_available_documents(self):
        self.encoder.return_value.predict.return_value = [0.5]
        self.assertEqual(self.hybrid.rerank("query", [self.a]), [self.a])
        self.encoder.return_value.predict.assert_called_once_with(
            [("query", "same text")], batch_size=4, show_progress_bar=False
        )

    def test_empty_stages_include_empty_reranking_pool_without_loading_encoder(self):
        self.sparse.return_value = []
        self.dense.return_value = []
        self.assertEqual(
            self.hybrid.retrieve_stages("query"),
            {
                "bm25": [],
                "faiss": [],
                "fusion": [],
                "reranker_candidates": [],
                "reranker": [],
            },
        )
        self.sparse.assert_called_once_with("query")
        self.dense.assert_called_once_with("query")
        self.encoder.assert_not_called()

    def test_rerank_empty_input_is_returned_unchanged(self):
        documents = []
        self.assertIs(self.hybrid.rerank("query", documents), documents)
        self.encoder.assert_not_called()

    def test_disabled_rerank_returns_all_documents_unchanged(self):
        self.hybrid.reranker_model = None
        documents = [self.a, self.b, self.c, self.d]
        self.assertIs(self.hybrid.rerank("query", documents), documents)
        with patch.object(self.hybrid, "rerank") as rerank:
            stages = self.hybrid.retrieve_stages("query")
        self.assertEqual(set(stages), {"bm25", "faiss", "fusion"})
        self.assertEqual(len(stages["fusion"]), 4)
        rerank.assert_not_called()
        self.encoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
