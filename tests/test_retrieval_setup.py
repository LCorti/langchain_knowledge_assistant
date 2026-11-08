"""Offline tests for paths, corpus loading, and index/retriever setup."""

import contextlib
import copy
import io
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag import cli, config, data, index, retriever


class TinyEmbeddings(Embeddings):
    """Deterministic vectors for exercising real FAISS persistence offline."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [float(text.count("alpha")), float(text.count("beta")), 1.0]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]


class ConfigurationTests(unittest.TestCase):
    def test_config_and_artifact_paths_do_not_depend_on_working_directory(self):
        expected = config.get_bm25_config()
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(config.get_bm25_config(), expected)
                self.assertEqual(
                    config.resolve_project_path("data/bm25_index"),
                    config.PROJECT_ROOT / "data/bm25_index",
                )
                self.assertEqual(
                    config.get_faiss_config()["main"]["persist_path"],
                    str(config.PROJECT_ROOT / "data/faiss_vs"),
                )
                self.assertEqual(
                    config.get_evaluation_config()["cache_dir"],
                    str(config.PROJECT_ROOT / "data/langchain_queries"),
                )
            finally:
                os.chdir(previous)

    def test_absolute_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()
            self.assertEqual(config.resolve_project_path(path), path)

    def test_invalid_yaml_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("- not\n- a mapping\n")
            with self.assertRaisesRegex(ValueError, "configuration mapping"):
                config.get_bm25_config(path)


class CorpusTests(unittest.TestCase):
    def test_preserves_chunk_id_text_and_source_metadata(self):
        metadata = {
            "url": "https://example.com/source",
            "start_byte": 0,
            "end_byte": 10,
        }
        source = Dataset.from_dict(
            {"_id": ["repo/file_0_10"], "text": ["alpha beta"], "metadata": [metadata]}
        )
        with patch.object(data, "load_dataset", return_value=source) as load:
            docs = data.load_source_corpus()
        load.assert_called_once_with(
            path="freshstack/corpus-oct-2024", name="langchain", split="train"
        )
        self.assertEqual(docs[0].page_content, "alpha beta")
        self.assertEqual(
            docs[0].metadata,
            {"doc_id": "repo/file_0_10", "chunk_metadata": metadata},
        )

    def test_missing_corpus_columns_raise_a_useful_error(self):
        source = Dataset.from_dict({"text": ["alpha"]})
        with patch.object(data, "load_dataset", return_value=source):
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                data.load_source_corpus()


class IndexSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sparse = config.get_bm25_config()
        self.sparse["main"]["persist_path"] = str(self.root / "nested" / "bm25")
        self.dense = config.get_faiss_config()
        self.dense["main"]["persist_path"] = str(self.root / "faiss")
        self.docs = [
            Document(page_content="alpha alpha", metadata={"doc_id": "a"}),
            Document(page_content="beta beta", metadata={"doc_id": "b"}),
            Document(page_content="gamma", metadata={"doc_id": "c"}),
        ]
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_build_both_missing_indexes_loads_corpus_once(self):
        with patch.object(index, "load_source_corpus", return_value=self.docs) as load:
            with patch.object(index, "build_bm25_index") as sparse:
                with patch.object(index, "build_faiss_index") as dense:
                    index.prepare_indexes(self.sparse, self.dense)
        load.assert_called_once_with()
        sparse.assert_called_once_with(self.docs, self.sparse)
        dense.assert_called_once_with(self.docs, self.dense)

    def test_existing_indexes_do_not_load_corpus_or_build(self):
        with patch.object(index, "bm25_index_exists", return_value=True):
            with patch.object(index, "faiss_index_exists", return_value=True):
                with patch.object(index, "load_source_corpus") as load:
                    with patch.object(index, "build_bm25_index") as sparse:
                        with patch.object(index, "build_faiss_index") as dense:
                            index.prepare_indexes(self.sparse, self.dense)
        load.assert_not_called()
        sparse.assert_not_called()
        dense.assert_not_called()

    def test_builds_only_the_missing_index(self):
        with patch.object(index, "bm25_index_exists", return_value=True):
            with patch.object(
                index, "load_source_corpus", return_value=self.docs
            ) as load:
                with patch.object(index, "build_bm25_index") as sparse:
                    with patch.object(index, "build_faiss_index") as dense:
                        index.prepare_indexes(self.sparse, self.dense)
        load.assert_called_once_with()
        sparse.assert_not_called()
        dense.assert_called_once_with(self.docs, self.dense)

    def test_explicit_rebuild_replaces_both_existing_indexes(self):
        with patch.object(index, "bm25_index_exists", return_value=True):
            with patch.object(index, "faiss_index_exists", return_value=True):
                with patch.object(
                    index, "load_source_corpus", return_value=self.docs
                ) as load:
                    with patch.object(index, "build_bm25_index") as sparse:
                        with patch.object(index, "build_faiss_index") as dense:
                            index.prepare_indexes(self.sparse, self.dense, rebuild=True)
        load.assert_called_once_with()
        sparse.assert_called_once()
        dense.assert_called_once()

    def test_bm25_round_trip_creates_parent_and_applies_current_k(self):
        self.sparse["retriever_params"]["k"] = 1
        index.build_bm25_index(self.docs, self.sparse)
        self.assertTrue(index.bm25_index_exists(self.sparse))
        self.sparse["retriever_params"]["k"] = 2
        with patch.object(index, "load_source_corpus") as load:
            loaded = retriever.get_bm25_retriever(self.sparse)
        load.assert_not_called()
        self.assertEqual(loaded.k, 2)
        self.assertEqual(len(loaded.invoke("alpha")), 2)
        self.assertEqual(loaded.docs[0].metadata["doc_id"], "a")
        self.assertEqual(index.load_or_create_bm25_index(self.sparse).k, 1)

    def test_faiss_round_trip_preserves_settings_metadata_and_ranking(self):
        with patch.object(index, "_get_embeddings", return_value=TinyEmbeddings()):
            built = index.build_faiss_index(self.docs, self.dense)
            loaded = index.load_faiss_index(self.dense)
        self.assertTrue(index.faiss_index_exists(self.dense))
        self.assertEqual(loaded.distance_strategy, built.distance_strategy)
        self.assertEqual(loaded._normalize_L2, built._normalize_L2)
        self.assertEqual(
            loaded.similarity_search("alpha", k=1)[0].metadata["doc_id"], "a"
        )

    def test_threshold_search_keeps_strong_matches_not_weak_ones(self):
        with patch.object(index, "_get_embeddings", return_value=TinyEmbeddings()):
            index.build_faiss_index(self.docs, self.dense)
            loaded = index.load_faiss_index(self.dense)
        search = loaded.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.9},
        )
        self.assertEqual(
            [doc.metadata["doc_id"] for doc in search.invoke("alpha")], ["a"]
        )
        self.assertEqual(
            [index._normalized_inner_product_relevance(score) for score in (-1, 0, 1)],
            [0.0, 0.5, 1.0],
        )

    def test_incompatible_build_settings_require_rebuilding_before_embedding_load(self):
        with patch.object(index, "_get_embeddings", return_value=TinyEmbeddings()):
            index.build_faiss_index(self.docs, self.dense)
        changes = (
            ("embeddings", "emb_model", "another/model"),
            ("embeddings", "norm_embs", False),
            ("index_params", "distance_strategy", "EUCLIDEAN_DISTANCE"),
            ("index_params", "normalize_L2", True),
        )
        for section, key, value in changes:
            with self.subTest(setting=key):
                changed = copy.deepcopy(self.dense)
                changed[section][key] = value
                with patch.object(index, "_get_embeddings") as embeddings:
                    with self.assertRaisesRegex(ValueError, "Rebuild"):
                        index.load_faiss_index(changed)
                embeddings.assert_not_called()
                with self.assertRaisesRegex(ValueError, "Rebuild"):
                    index.prepare_indexes(self.sparse, changed)

    def test_legacy_index_loads_with_warning_but_metric_mismatch_is_rejected(self):
        with patch.object(index, "_get_embeddings", return_value=TinyEmbeddings()):
            index.build_faiss_index(self.docs, self.dense)
            (Path(self.dense["main"]["persist_path"]) / "build_config.json").unlink()
            with self.assertWarnsRegex(RuntimeWarning, "embedding compatibility"):
                index.load_faiss_index(self.dense)
            self.dense["index_params"]["distance_strategy"] = "EUCLIDEAN_DISTANCE"
            with self.assertWarns(RuntimeWarning):
                with self.assertRaisesRegex(ValueError, "distance metric"):
                    index.load_faiss_index(self.dense)

    def test_retrieval_and_device_settings_do_not_invalidate_index(self):
        with patch.object(index, "_get_embeddings", return_value=TinyEmbeddings()):
            index.build_faiss_index(self.docs, self.dense)
            self.dense["embeddings"]["device"] = "cpu"
            self.dense["embeddings"]["batch_size"] = 1
            self.dense["retriever_params"]["k"] = 2
            index.load_faiss_index(self.dense)

    def test_missing_indexes_fail_before_corpus_or_embedding_loading(self):
        with patch.object(index, "load_source_corpus") as load:
            with patch.object(index, "_get_embeddings") as embeddings:
                with self.assertRaisesRegex(FileNotFoundError, "build_index.py"):
                    index.load_or_create_bm25_index(self.sparse)
                with self.assertRaisesRegex(FileNotFoundError, "build_index.py"):
                    index.load_faiss_index(self.dense)
        load.assert_not_called()
        embeddings.assert_not_called()

    def test_partial_faiss_directory_is_not_a_valid_index(self):
        path = Path(self.dense["main"]["persist_path"])
        path.mkdir()
        (path / "index.faiss").touch()
        self.assertFalse(index.faiss_index_exists(self.dense))
        with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
            index.load_faiss_index(self.dense)


class RetrieverSetupTests(unittest.TestCase):
    def setUp(self):
        self.config = config.get_faiss_config()
        self.store = Mock()
        self.enterContext(
            patch.object(retriever, "load_faiss_index", return_value=self.store)
        )
        self.enterContext(patch.object(retriever.faiss, "get_num_gpus", return_value=0))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_only_search_mode_specific_parameters_are_forwarded(self):
        cases = (
            ("similarity", {"k": 20}),
            ("mmr", {"k": 20, "fetch_k": 40, "lambda_mult": 0.5}),
            ("similarity_score_threshold", {"k": 20, "score_threshold": 0.7}),
        )
        original = copy.deepcopy(self.config)
        for search_type, expected in cases:
            with self.subTest(search_type=search_type):
                self.config["retriever_params"]["search_type"] = search_type
                result = retriever.get_faiss_retriever(self.config)
                self.assertIs(result, self.store.as_retriever.return_value)
                self.store.as_retriever.assert_called_with(
                    search_type=search_type, search_kwargs=expected
                )
        original["retriever_params"]["search_type"] = "similarity_score_threshold"
        self.assertEqual(self.config, original)

    def test_unnormalized_inner_product_threshold_search_is_rejected(self):
        self.config["retriever_params"]["search_type"] = "similarity_score_threshold"
        self.config["embeddings"]["norm_embs"] = False
        with self.assertRaisesRegex(ValueError, "requires normalized"):
            retriever.get_faiss_retriever(self.config)
        self.store.as_retriever.assert_not_called()

    def test_gpu_resources_are_kept_alive_on_store(self):
        with patch.object(retriever.faiss, "get_num_gpus", return_value=1):
            with patch.object(
                retriever.faiss, "StandardGpuResources", create=True
            ) as resources:
                with patch.object(
                    retriever.faiss, "index_cpu_to_gpu", create=True
                ) as move:
                    cpu_index = self.store.index
                    retriever.get_faiss_retriever(self.config)
        move.assert_called_once_with(resources.return_value, 0, cpu_index)
        self.assertIs(self.store.index, move.return_value)
        self.assertIs(self.store._gpu_resources, resources.return_value)

    def test_hybrid_uses_configured_retrievers_and_document_ids(self):
        sparse_config = config.get_bm25_config()
        with patch.object(retriever, "get_bm25_retriever") as sparse:
            with patch.object(retriever, "get_faiss_retriever") as dense:
                with patch.object(retriever, "EnsembleRetriever") as ensemble:
                    result = retriever.get_hybrid_retriever(sparse_config, self.config)
        sparse.assert_called_once_with(sparse_config)
        dense.assert_called_once_with(self.config)
        ensemble.assert_called_once_with(
            retrievers=[sparse.return_value, dense.return_value],
            weights=[0.5, 0.5],
            c=60,
            id_key="doc_id",
        )
        self.assertIs(result, ensemble.return_value)

    def test_cli_delegates_to_one_hybrid_factory(self):
        sparse_config = config.get_bm25_config()
        with patch.object(cli, "get_bm25_config", return_value=sparse_config):
            with patch.object(cli, "get_faiss_config", return_value=self.config):
                with patch.object(cli, "get_hybrid_retriever") as hybrid:
                    with patch.object(cli, "run_cli_loop") as loop:
                        cli.main()
        hybrid.assert_called_once_with(sparse_config, self.config)
        loop.assert_called_once_with(hybrid.return_value)


if __name__ == "__main__":
    unittest.main()
