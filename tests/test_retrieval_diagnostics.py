"""Offline diagnostics tests with real datasets and metric implementations."""

import contextlib
import copy
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from datasets import Dataset, DatasetDict
from langchain_core.documents import Document

from src.rag.evaluation import retrieval_diagnostics as diagnostics


def nugget(nugget_id="n1", relevant=("a",), negative=("bad",)):
    return {
        "_id": nugget_id,
        "relevant_corpus_ids": list(relevant),
        "non_relevant_corpus_ids": list(negative),
    }


def query(query_id="q1", nuggets=None, title="Same", text="question"):
    return {
        "query_id": query_id,
        "query_title": title,
        "query_text": text,
        "nuggets": [nugget()] if nuggets is None else nuggets,
    }


def document(doc_id, text="document content"):
    return Document(page_content=text, metadata={"doc_id": doc_id})


def stage_retriever(*responses):
    return Mock(spec=["retrieve_stages"], retrieve_stages=Mock(side_effect=responses))


class StageEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_duplicate_texts_replay_in_query_order_and_empty_runs_count(self):
        queries = Dataset.from_list(
            [
                query("q1"),
                query("q2", [nugget(relevant=("b",))]),
            ]
        )
        retriever = stage_retriever(
            {"bm25": [document("a")], "reranked": []},
            {"bm25": [], "reranked": [document("b")]},
        )
        report = diagnostics.evaluate_stages(queries, retriever, [1, 2])
        self.assertEqual(
            retriever.retrieve_stages.call_args_list,
            [call("Same question"), call("Same question")],
        )
        for stage, rankings in (("bm25", [["a"], []]), ("reranked", [[], ["b"]])):
            with self.subTest(stage=stage):
                result = report[stage]
                expected = {
                    f"{metric}@{k}": 0.5
                    for metric in ("Recall", "NDCG", "Coverage", "alpha-nDCG")
                    for k in (1, 2)
                }
                expected.update({"Hole@1": 0.0, "Hole@2": 0.0})
                self.assertEqual(result["metrics"], expected)
                self.assertEqual(
                    [row["query_id"] for row in result["per_query"]], ["q1", "q2"]
                )
                self.assertEqual(
                    [row["ranking"] for row in result["per_query"]], rankings
                )
                empty_index = rankings.index([])
                empty = result["per_query"][empty_index]
                self.assertEqual(empty["num_returned"], 0)
                self.assertTrue(all(value == 0 for value in empty["metrics"].values()))
                self.assertEqual(
                    empty["missed_relevant_doc_ids"], ["a" if empty_index == 0 else "b"]
                )

    def test_real_discounted_metrics_and_nugget_coverage(self):
        queries = Dataset.from_list(
            [
                query(
                    nuggets=[
                        nugget("n1", ("a",)),
                        nugget("n2", ("b",)),
                    ]
                )
            ]
        )
        retriever = stage_retriever(
            {
                "fusion": [
                    document("bad"),
                    document("a"),
                    document("b"),
                ]
            }
        )
        result = diagnostics.evaluate_stages(queries, retriever, [1, 2, 3])["fusion"]
        scores = result["metrics"]
        discount2, discount3 = 1 / math.log2(3), 1 / math.log2(4)
        self.assertEqual(scores["Recall@1"], 0.0)
        self.assertEqual(scores["Recall@2"], 0.5)
        self.assertEqual(scores["Recall@3"], 1.0)
        self.assertEqual(scores["Coverage@1"], 0.0)
        self.assertEqual(scores["Coverage@2"], 0.5)
        self.assertEqual(scores["Coverage@3"], 1.0)
        self.assertEqual(scores["NDCG@1"], 0.0)
        self.assertEqual(scores["NDCG@2"], round(discount2 / (1 + discount2), 5))
        self.assertEqual(
            scores["NDCG@3"], round((discount2 + discount3) / (1 + discount2), 5)
        )
        self.assertEqual(scores["alpha-nDCG@2"], round(discount2 / (1 + discount2), 4))
        self.assertEqual(
            scores["alpha-nDCG@3"], round((discount2 + discount3) / (1 + discount2), 4)
        )
        self.assertEqual(result["per_query"][0]["metrics"], scores)

    def test_dedup_precedes_clipping_and_original_unique_count_is_retained(self):
        queries = Dataset.from_list([query(nuggets=[nugget(relevant=("a", "z"))])])
        documents = [
            document("a", "first version"),
            document("a", "duplicate"),
            document("unknown", "short"),
            document("bad"),
            document("z"),
        ]
        retriever = stage_retriever({"fusion": documents})
        result = diagnostics.evaluate_stages(
            queries,
            retriever,
            [1, 2],
            examples_per_kind=2,
            text_chars=5,
        )["fusion"]
        row = result["per_query"][0]
        self.assertEqual(row["ranking"], ["a", "unknown"])
        self.assertEqual(row["num_returned"], 4)
        self.assertEqual(row["missed_relevant_doc_ids"], ["z"])
        self.assertEqual(result["metrics"]["Recall@2"], 0.5)
        self.assertEqual(result["metrics"]["Hole@2"], 0.5)
        self.assertEqual(result["examples"]["bad"], [])
        good = result["examples"]["good"][0]
        self.assertEqual(good["rank"], 1)
        self.assertEqual(good["document_text"], "first")
        self.assertTrue(good["text_truncated"])
        self.assertEqual(good["relevant_nugget_ids"], ["n1"])
        unknown = result["examples"]["unjudged"][0]
        self.assertEqual(unknown["rank"], 2)
        self.assertIsNone(unknown["relevance"])
        self.assertEqual(unknown["document_text"], "short")
        self.assertFalse(unknown["text_truncated"])
        self.assertEqual(len(documents), 5)
        self.assertEqual(documents[0].page_content, "first version")

    def test_numeric_and_string_doc_ids_deduplicate_keep_first(self):
        first = document(7)
        other = document("other")
        result = diagnostics.unique_documents([first, document("7"), other], "q1")
        self.assertEqual(result, [first, other])
        self.assertIs(result[0], first)

    def test_missing_doc_ids_rejected_even_beyond_cutoff(self):
        for metadata in ({}, {"doc_id": None}, {"doc_id": ""}):
            with self.subTest(metadata=metadata):
                retriever = stage_retriever(
                    {
                        "fusion": [
                            document("a"),
                            Document(page_content="invalid", metadata=metadata),
                        ]
                    }
                )
                with self.assertRaisesRegex(ValueError, "query q1.*missing.*doc_id"):
                    diagnostics.evaluate_stages(
                        Dataset.from_list([query()]), retriever, [1]
                    )

    def test_positive_beats_cross_nugget_negative_in_either_order(self):
        positive = nugget("positive", ("a",), ("bad",))
        negative = nugget("negative", ("b",), ("a", "bad"))
        for nuggets in ([positive, negative], [negative, positive]):
            with self.subTest(nuggets=nuggets):
                row = query(nuggets=nuggets)
                self.assertEqual(diagnostics.judgments(row), {"a": 1, "b": 1, "bad": 0})
                result = diagnostics.evaluate_stages(
                    Dataset.from_list([row]),
                    stage_retriever(
                        {
                            "fusion": [
                                document("a"),
                                document("bad"),
                                document("unknown"),
                            ]
                        }
                    ),
                    [3],
                )["fusion"]
                self.assertEqual(
                    [pair["doc_id"] for pair in result["examples"]["good"]], ["a"]
                )
                self.assertEqual(
                    result["examples"]["good"][0]["relevant_nugget_ids"], ["positive"]
                )
                self.assertEqual(
                    [pair["doc_id"] for pair in result["examples"]["bad"]], ["bad"]
                )
                self.assertEqual(result["examples"]["bad"][0]["relevance"], 0)
                self.assertEqual(result["examples"]["unjudged"][0]["doc_id"], "unknown")
                self.assertIsNone(result["examples"]["unjudged"][0]["relevance"])
                self.assertEqual(result["metrics"]["Recall@3"], 0.5)

    def test_hole_uses_split_wide_labels_but_examples_use_query_labels(self):
        queries = Dataset.from_list([query(), query("q2", [nugget(relevant=("b",))])])
        report = diagnostics.evaluate_stages(
            queries,
            stage_retriever(
                {"fusion": [document("b"), document("never-judged")]},
                {"fusion": []},
            ),
            [2],
        )["fusion"]
        self.assertEqual(report["metrics"]["Hole@2"], 0.25)
        self.assertEqual(report["per_query"][0]["metrics"]["Hole@2"], 0.5)
        self.assertEqual(report["examples"]["bad"], [])
        self.assertEqual(
            [pair["doc_id"] for pair in report["examples"]["unjudged"]],
            ["b", "never-judged"],
        )

    def test_example_limit_samples_round_robin_across_queries(self):
        queries = Dataset.from_list([query(), query("q2")])
        responses = [
            {"fusion": [document("a"), document("u1"), document("u2")]}
            for _ in range(2)
        ]
        result = diagnostics.evaluate_stages(
            queries,
            stage_retriever(*responses),
            [3],
            examples_per_kind=2,
        )["fusion"]
        self.assertEqual(
            [
                (pair["query_id"], pair["doc_id"])
                for pair in result["examples"]["unjudged"]
            ],
            [("q1", "u1"), ("q2", "u1")],
        )

    def test_cutoffs_above_twenty_skip_only_alpha_ndcg(self):
        result = diagnostics.evaluate_stages(
            Dataset.from_list([query()]),
            stage_retriever({"fusion": [document("a")]}),
            [1, 21],
        )["fusion"]["metrics"]
        self.assertEqual(result["alpha-nDCG@1"], 1.0)
        self.assertNotIn("alpha-nDCG@21", result)
        for name in ("Recall", "NDCG", "Coverage"):
            self.assertEqual(result[f"{name}@21"], 1.0)

    def test_invalid_data_and_options_fail_before_retrieval(self):
        valid = Dataset.from_list([query()])
        cases = [
            (valid.select([]), [1], {}),
            (Dataset.from_list([query(), query()]), [1], {}),
            (Dataset.from_list([query(nuggets=[])]), [1], {}),
            (Dataset.from_list([query(nuggets=[nugget(), nugget()])]), [1], {}),
            (Dataset.from_list([query(nuggets=[nugget(relevant=())])]), [1], {}),
            (Dataset.from_list([query(title="", text=" ")]), [1], {}),
            (valid, [], {}),
            (valid, [0], {}),
            (valid, [True], {}),
            (valid, [1.5], {}),
            (valid, (1,), {}),
            (valid, [1], {"examples_per_kind": 0}),
            (valid, [1], {"text_chars": True}),
        ]
        for queries, cutoffs, options in cases:
            with self.subTest(rows=len(queries), cutoffs=cutoffs, options=options):
                retriever = stage_retriever()
                with self.assertRaises(ValueError):
                    diagnostics.evaluate_stages(queries, retriever, cutoffs, **options)
                retriever.retrieve_stages.assert_not_called()

    def test_inconsistent_stage_names_rejected(self):
        retriever = stage_retriever({"bm25": []}, {"fusion": []})
        with self.assertRaisesRegex(ValueError, "inconsistent stages"):
            diagnostics.evaluate_stages(
                Dataset.from_list([query(), query("q2")]),
                retriever,
                [1],
            )


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "bm25": {
                "main": {"persist_path": "data/sparse"},
                "retriever_params": {"k": 3},
            },
            "faiss": {
                "main": {"persist_path": "data/dense"},
                "retriever_params": {"k": 4, "search_type": "mmr", "fetch_k": 2},
                "embeddings": {"device": "cpu", "options": ["keep"]},
            },
            "retriever": {"weights": [0.5, 0.5], "c": 60},
        }
        for section in self.base:
            self.enterContext(
                patch.object(
                    diagnostics.project_config,
                    f"get_{section}_config",
                    return_value=self.base[section],
                )
            )

    def test_deep_merge_copies_both_inputs_and_replaces_lists(self):
        base = {"nested": {"keep": [1], "replace": [2]}, "scalar": 3}
        overrides = {"nested": {"replace": [4]}, "scalar": {"new": [5]}}
        before = copy.deepcopy((base, overrides))
        merged = diagnostics.merge_config(base, overrides)
        self.assertEqual(
            merged, {"nested": {"keep": [1], "replace": [4]}, "scalar": {"new": [5]}}
        )
        merged["nested"]["keep"].append(6)
        merged["nested"]["replace"].append(7)
        merged["scalar"]["new"].append(8)
        self.assertEqual((base, overrides), before)

    def test_configurations_are_isolated_normalize_mmr_and_resolve_paths(self):
        settings = {
            "defaults": {"bm25": {"retriever_params": {"k": 8}}},
            "configurations": {
                "wide": {
                    "faiss": {"retriever_params": {"k": 9}},
                    "retriever": {"weights": [0.25, 0.75]},
                },
                "baseline": {},
            },
        }
        before = copy.deepcopy((self.base, settings))
        resolved = dict(diagnostics.configurations(settings, 7))
        wide, baseline = resolved["wide"], resolved["baseline"]
        self.assertEqual(wide["faiss"]["retriever_params"]["fetch_k"], 9)
        self.assertEqual(baseline["faiss"]["retriever_params"]["fetch_k"], 4)
        self.assertEqual(baseline["bm25"]["retriever_params"]["k"], 8)
        self.assertEqual(baseline["reranker"]["top_n"], 7)
        self.assertEqual(baseline["reranker"]["rerank_k"], 100)
        self.assertEqual(baseline["retriever"]["weights"], [0.5, 0.5])
        for section in ("bm25", "faiss"):
            self.assertEqual(
                wide[section]["main"]["persist_path"],
                str(
                    diagnostics.project_config.resolve_project_path(
                        self.base[section]["main"]["persist_path"]
                    )
                ),
            )
        wide["faiss"]["embeddings"]["options"].append("changed")
        wide["retriever"]["weights"].append(1)
        wide["reranker"]["top_n"] = 1
        self.assertEqual(baseline["faiss"]["embeddings"]["options"], ["keep"])
        self.assertEqual(baseline["reranker"]["top_n"], 7)
        self.assertEqual((self.base, settings), before)
        self.assertEqual(
            dict(diagnostics.configurations({}, 120))["baseline"]["reranker"][
                "rerank_k"
            ],
            120,
        )

    def test_mmr_preserves_larger_fetch_k_and_other_search_does_not_normalize(self):
        for search_type, fetch_k, expected in (("mmr", 20, 20), ("similarity", 2, 2)):
            with self.subTest(search_type=search_type):
                settings = {
                    "defaults": {
                        "faiss": {
                            "retriever_params": {
                                "search_type": search_type,
                                "fetch_k": fetch_k,
                            }
                        }
                    }
                }
                effective = dict(diagnostics.configurations(settings, 1))["baseline"]
                self.assertEqual(
                    effective["faiss"]["retriever_params"]["fetch_k"], expected
                )

    def test_invalid_configuration_setup_is_rejected(self):
        invalid = [
            {"defaults": []},
            {"configurations": {}},
            {"configurations": []},
            {"configurations": {"../escape": {}}},
            {"configurations": {"": {}}},
            {"configurations": {1: {}}},
            {"configurations": {"name": None}},
            {"configurations": {"name": {"unknown": {}}}},
        ]
        for section in ("bm25", "faiss"):
            for value in (0, -1, True, "2", 1.5):
                invalid.append(
                    {"defaults": {section: {"retriever_params": {"k": value}}}}
                )
        invalid.extend(
            [
                {"defaults": {"faiss": {"retriever_params": {"fetch_k": 0}}}},
                {"defaults": {"reranker": {"top_n": 3, "rerank_k": 2}}},
            ]
        )
        for key in ("top_n", "rerank_k", "reranker_batch_size"):
            invalid.append({"defaults": {"reranker": {key: True}}})
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                diagnostics.configurations(settings, 2)

    def test_positive_integer_rejects_nonintegers_and_nonpositive_values(self):
        self.assertEqual(diagnostics.positive_integer(2, "depth"), 2)
        for value in (False, True, 0, -1, 1.0, "1", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "depth must be a positive integer"
                ):
                    diagnostics.positive_integer(value, "depth")


class DiagnosticsMainTests(unittest.TestCase):
    def setUp(self):
        ConfigurationTests.setUp(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "reports"
        self.settings = {"split": "test", "cutoffs": [2, 1, 2]}
        self.queries = Dataset.from_list([query(), query("q2")])
        self.config = self.enterContext(
            patch.object(
                diagnostics,
                "evaluation_config",
                return_value=self.settings,
            )
        )
        self.prepare = self.enterContext(
            patch.object(
                diagnostics,
                "prepare_from_config",
                return_value=DatasetDict(
                    {
                        "dev": self.queries.select([0]),
                        "test": self.queries,
                    }
                ),
            )
        )
        self.hybrid = self.enterContext(patch.object(diagnostics, "HybridRetriever"))
        self.stdout = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.stdout))

    def test_main_writes_json_and_csv_for_each_configuration(self):
        experiments = self.root / "experiments.yaml"
        experiments.write_text(
            "examples_per_kind: 1\ntext_chars: 4\nconfigurations:\n"
            "  baseline: {}\n  alternate:\n    retriever:\n      weights: [0.2, 0.8]\n",
            encoding="utf-8",
        )
        fakes = [
            stage_retriever({"bm25": [document("a", "long text")], "reranked": []})
            for _ in range(2)
        ]
        self.hybrid.side_effect = fakes
        diagnostics.main(
            [
                "--config",
                "chosen.yaml",
                "--experiments",
                str(experiments),
                "--output-dir",
                str(self.output),
                "--limit",
                "1",
                "--overwrite",
            ]
        )
        self.config.assert_called_once_with(Path("chosen.yaml"))
        self.prepare.assert_called_once_with(self.settings, overwrite=True)
        self.assertEqual(self.hybrid.call_count, 2)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"baseline.json", "alternate.json", "summary.csv"},
        )
        expected_summary = {}
        for index, name in enumerate(("baseline", "alternate")):
            payload = json.loads(
                (self.output / f"{name}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["configuration"], name)
            self.assertEqual(payload["evaluation_settings"], self.settings)
            self.assertEqual(payload["query_ids"], ["q1"])
            self.assertEqual(payload["num_queries"], 1)
            self.assertEqual(payload["split"], "test")
            self.assertEqual(payload["cutoffs"], [1, 2])
            self.assertEqual(payload["examples_per_kind"], 1)
            self.assertEqual(payload["text_chars"], 4)
            effective = payload["effective_config"]
            self.assertEqual(
                effective["retriever"]["weights"],
                [0.5, 0.5] if index == 0 else [0.2, 0.8],
            )
            self.assertEqual(
                self.hybrid.call_args_list[index],
                call(
                    effective["bm25"],
                    effective["faiss"],
                    effective["retriever"],
                    **effective["reranker"],
                ),
            )
            fakes[index].retrieve_stages.assert_called_once_with("Same question")
            stages = payload["stages"]
            self.assertEqual(stages["bm25"]["metrics"]["Recall@1"], 1.0)
            self.assertEqual(stages["reranked"]["metrics"]["Recall@1"], 0.0)
            example = stages["bm25"]["examples"]["good"][0]
            self.assertEqual(example["document_text"], "long")
            self.assertTrue(example["text_truncated"])
            self.assertIn("not necessarily non-relevant", payload["notes"]["unjudged"])
            for stage, result in stages.items():
                expected_summary[name, stage] = result["metrics"]
        with (self.output / "summary.csv").open(newline="", encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual(len(summary), 4)
        self.assertEqual(
            {(row["configuration"], row["stage"]) for row in summary},
            set(expected_summary),
        )
        for row in summary:
            metrics = expected_summary[row.pop("configuration"), row.pop("stage")]
            self.assertEqual({key: float(value) for key, value in row.items()}, metrics)
        self.assertIn("Saved reports and summary.csv", self.stdout.getvalue())

    def test_invalid_cli_setup_fails_before_prepare_or_model_load(self):
        cases = [
            ({"split": "train"}, []),
            ({"cutoffs": []}, []),
            ({"cutoffs": "1"}, []),
            ({"cutoffs": [0]}, []),
            ({"cutoffs": [True]}, []),
            ({"cutoffs": [1.5]}, []),
            ({}, ["--limit", "0"]),
            ({}, ["--limit", "-1"]),
        ]
        for change, args in cases:
            with self.subTest(change=change, args=args):
                self.config.return_value = self.settings | change
                with self.assertRaises(ValueError):
                    diagnostics.main(["--output-dir", str(self.output), *args])
                self.prepare.assert_not_called()
                self.hybrid.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_invalid_experiments_fail_before_prepare_or_model_load(self):
        path = self.root / "invalid.yaml"
        for contents in (
            "- not-a-mapping\n",
            "",
            "configurations: {}\n",
            "examples_per_kind: 0\n",
            "text_chars: false\n",
        ):
            with self.subTest(contents=contents):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    diagnostics.main(
                        ["--experiments", str(path), "--output-dir", str(self.output)]
                    )
                self.prepare.assert_not_called()
                self.hybrid.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_invalid_query_data_does_not_create_outputs_or_load_model(self):
        for queries in (
            self.queries.select([]),
            Dataset.from_list([query(), query()]),
            Dataset.from_list([query(nuggets=[])]),
        ):
            with self.subTest(rows=len(queries)):
                self.prepare.return_value = {"test": queries}
                with self.assertRaises(ValueError):
                    diagnostics.main(["--output-dir", str(self.output)])
                self.hybrid.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_existing_output_directory_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "baseline.json"
        sentinel.write_text("precious existing results", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            diagnostics.main(["--output-dir", str(self.output)])
        self.hybrid.assert_not_called()
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"), "precious existing results"
        )


if __name__ == "__main__":
    unittest.main()
