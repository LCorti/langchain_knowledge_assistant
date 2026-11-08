"""Model-free tests: python -m unittest discover -s src -p test_generation.py."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from rag.evaluation import generation


def freshstack_row(**overrides):
    return {
        "query_id": "q-1",
        "query_title": "How do I fetch?",
        "query_text": "Include a timeout.",
        "answer_text": "Use fetch(timeout=5).",
        "nuggets": [{"_id": "n-1", "text": "Set a timeout of five seconds."}],
        **overrides,
    }


def answer_record(**overrides):
    return {
        **generation.prepare_rows([freshstack_row()])[0],
        "answer": "Use fetch(timeout=5). [1]",
        "sources": [],
        **overrides,
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PrepareRowsTests(unittest.TestCase):
    def test_real_schema_normalizes_ids_and_preserves_reference_text(self):
        raw = freshstack_row(
            query_id=12,
            answer_text="  Gold code: fetch(timeout=5) [1]\n",
            nuggets=[{"_id": 34, "text": "  Keep this text.\n", "extra": "ignored"}],
        )
        original = copy.deepcopy(raw)
        self.assertEqual(
            generation.prepare_rows(iter([raw])),
            [
                {
                    "query_id": "12",
                    "question": "How do I fetch? Include a timeout.",
                    "gold_answer": raw["answer_text"],
                    "nuggets": [{"_id": "34", "text": "  Keep this text.\n"}],
                }
            ],
        )
        self.assertEqual(raw, original)

    def test_title_or_body_alone_is_enough(self):
        for fields, expected in (
            ({"query_title": "", "query_text": " Body only "}, "Body only"),
            ({"query_title": " Title only ", "query_text": ""}, "Title only"),
        ):
            with self.subTest(fields=fields):
                self.assertEqual(
                    generation.prepare_rows([freshstack_row(**fields)])[0]["question"],
                    expected,
                )
        raw = freshstack_row()
        del raw["query_text"]
        self.assertEqual(
            generation.prepare_rows([raw])[0]["question"], raw["query_title"]
        )

    def test_empty_split_and_invalid_query_fields(self):
        cases = [("empty split", [])]
        for key, values in (
            ("query_id", (None, "", True, False, 1.5, [])),
            ("query_title", (None, 12, [])),
            ("query_text", (None, 12, [])),
            ("answer_text", (None, "", " \n", 12, [])),
            ("nuggets", (None, [], {}, "nugget")),
        ):
            cases.extend(
                (f"{key}={value!r}", [freshstack_row(**{key: value})])
                for value in values
            )
        cases.append(
            ("blank question", [freshstack_row(query_title=" ", query_text="\n")])
        )
        for key in ("query_id", "answer_text", "nuggets"):
            row = freshstack_row()
            del row[key]
            cases.append((f"missing {key}", [row]))
        for name, rows in cases:
            with self.subTest(case=name), self.assertRaises(ValueError):
                generation.prepare_rows(rows)

    def test_bad_and_duplicate_nuggets(self):
        invalid = [None, "claim", {}, {"_id": "n"}, {"text": "claim"}]
        invalid.extend(
            {"_id": value, "text": "claim"}
            for value in (None, "", True, False, 1.5, [])
        )
        invalid.extend({"_id": "n", "text": value} for value in (None, "", " \n", 12))
        for nugget in invalid:
            with (
                self.subTest(nugget=nugget),
                self.assertRaises((TypeError, ValueError)),
            ):
                generation.prepare_rows([freshstack_row(nuggets=[nugget])])
        for ids in (("n", "n"), (1, "1")):
            with (
                self.subTest(ids=ids),
                self.assertRaisesRegex(ValueError, "duplicate nugget"),
            ):
                generation.prepare_rows(
                    [
                        freshstack_row(
                            nuggets=[
                                {"_id": nugget_id, "text": "claim"} for nugget_id in ids
                            ]
                        )
                    ]
                )

    def test_duplicate_query_ids_include_normalized_collisions(self):
        for ids in (("q", "q"), (1, "1")):
            with (
                self.subTest(ids=ids),
                self.assertRaisesRegex(ValueError, "Duplicate query ID"),
            ):
                generation.prepare_rows(
                    [freshstack_row(query_id=query_id) for query_id in ids]
                )

    def test_main_validates_entire_split_before_constructing_models(self):
        prepare_module = ModuleType("rag.evaluation.prepare_queries")
        generator_module = ModuleType("rag.generate")
        generator_module.QwenGenerator = Mock()
        pipeline_module = ModuleType("rag.pipeline")
        pipeline_module.RAGPipeline = Mock()
        retriever_module = ModuleType("rag.retriever")
        retriever_module.HybridRetriever = Mock()
        modules = {
            module.__name__: module
            for module in (
                prepare_module,
                generator_module,
                pipeline_module,
                retriever_module,
            )
        }
        cases = (
            [],
            [freshstack_row(), freshstack_row(query_id="q-2", answer_text="")],
            [freshstack_row(), freshstack_row()],
        )
        for rows in cases:
            with self.subTest(rows=rows):
                prepare_module.prepare_from_config = Mock(return_value={"dev": rows})
                config = SimpleNamespace(
                    get_generation_evaluation_config=Mock(
                        return_value={"split": "dev"}
                    ),
                    get_generator_config=Mock(),
                )
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(generation, "project_config", config),
                    patch.object(generation, "JinaAnswerSimilarity") as similarity,
                    patch.object(generation, "QwenNuggetJudge") as judge,
                    self.assertRaises(ValueError),
                ):
                    generation.main([])
                config.get_generator_config.assert_not_called()
                similarity.assert_not_called()
                judge.assert_not_called()
                generator_module.QwenGenerator.assert_not_called()
                pipeline_module.RAGPipeline.assert_not_called()
                retriever_module.HybridRetriever.assert_not_called()


class AnswerFilesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "answers.jsonl"
        self.rows = generation.prepare_rows(
            [freshstack_row(), freshstack_row(query_id="q-2")]
        )

    def save(self, records):
        self.path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            ),
            encoding="utf-8",
        )

    def test_load_reorders_saved_answers_to_query_order(self):
        records = [{**row, "answer": "", "sources": []} for row in self.rows]
        self.save(list(reversed(records)))
        self.assertEqual(generation.load_answers(self.path, self.rows), records)

    def test_load_rejects_each_provenance_mismatch(self):
        for key, value in (
            ("question", "Changed question"),
            ("gold_answer", "Changed reference"),
            ("nuggets", [{"_id": "other", "text": "Set a timeout of five seconds."}]),
            ("nuggets", [{"_id": "n-1", "text": "Changed claim"}]),
        ):
            with self.subTest(key=key, value=value):
                self.save([answer_record(**{key: value})])
                with self.assertRaisesRegex(ValueError, "query/reference differs"):
                    generation.load_answers(self.path, self.rows[:1])
        for key in ("question", "gold_answer", "nuggets"):
            with self.subTest(missing=key):
                record = answer_record()
                del record[key]
                self.save([record])
                with self.assertRaisesRegex(ValueError, "query/reference differs"):
                    generation.load_answers(self.path, self.rows[:1])

    def test_load_rejects_missing_duplicate_and_nonstring_answers(self):
        self.save([answer_record()])
        with self.assertRaisesRegex(ValueError, "No saved answer.*q-2"):
            generation.load_answers(self.path, self.rows)
        self.save([answer_record(), answer_record()])
        with self.assertRaisesRegex(ValueError, "Duplicate saved answer"):
            generation.load_answers(self.path, self.rows[:1])
        for value in (None, 123, [], {}):
            with self.subTest(answer=value):
                self.save([answer_record(answer=value)])
                with self.assertRaisesRegex(TypeError, "Invalid saved answer"):
                    generation.load_answers(self.path, self.rows[:1])
        record = answer_record()
        del record["answer"]
        self.save([record])
        with self.assertRaisesRegex(TypeError, "Invalid saved answer"):
            generation.load_answers(self.path, self.rows[:1])

    def test_generate_serializes_document_like_sources_and_checkpoints_each_answer(
        self,
    ):
        sources = [
            SimpleNamespace(
                metadata={"doc_id": "doc-1", "ignored": True},
                page_content="Café code\nfetch()",
            ),
            SimpleNamespace(metadata={}, page_content="Second source"),
        ]
        first = {
            **self.rows[0],
            "answer": "Réponse [1]",
            "sources": [
                {"citation": 1, "doc_id": "doc-1", "text": "Café code\nfetch()"},
                {"citation": 2, "doc_id": None, "text": "Second source"},
            ],
        }

        def run(question):
            if pipeline.run.call_count == 1:
                return {"answer": first["answer"], "sources": sources}
            self.assertEqual(read_jsonl(self.path), [first])
            return {"answer": "", "sources": []}

        pipeline = SimpleNamespace(run=Mock(side_effect=run))
        records = generation.generate_answers(self.rows, pipeline, self.path)
        self.assertEqual(
            records, [first, {**self.rows[1], "answer": "", "sources": []}]
        )
        self.assertEqual(read_jsonl(self.path), records)
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))
        self.assertIn("Réponse", self.path.read_text(encoding="utf-8"))
        self.assertEqual(
            [call.args for call in pipeline.run.call_args_list],
            [(row["question"],) for row in self.rows],
        )
        self.assertEqual(generation.load_answers(self.path, self.rows), records)

    def test_pipeline_failure_preserves_completed_checkpoint(self):
        pipeline = SimpleNamespace(
            run=Mock(
                side_effect=[
                    {"answer": "Completed", "sources": []},
                    RuntimeError("pipeline failed"),
                ]
            )
        )
        with self.assertRaisesRegex(RuntimeError, "pipeline failed"):
            generation.generate_answers(self.rows, pipeline, self.path)
        self.assertEqual(
            read_jsonl(self.path),
            [{**self.rows[0], "answer": "Completed", "sources": []}],
        )

    def test_nonstring_generation_is_not_checkpointed(self):
        for answer in (None, 12, []):
            with self.subTest(answer=answer):
                pipeline = SimpleNamespace(
                    run=Mock(return_value={"answer": answer, "sources": []})
                )
                with self.assertRaisesRegex(TypeError, "non-string answer"):
                    generation.generate_answers(self.rows[:1], pipeline, self.path)
                self.assertFalse(self.path.exists())


class EvaluateAnswersTests(unittest.TestCase):
    def setUp(self):
        self.answers = [
            answer_record(nuggets=[{"_id": "only", "text": "One claim"}]),
            answer_record(
                query_id="q-2",
                answer="Other answer",
                gold_answer="Other gold",
                nuggets=[
                    {"_id": str(index), "text": f"Claim {index}"} for index in range(4)
                ],
            ),
        ]
        self.judgments = [
            [{"nugget_id": "only", "label": "fully_covered"}],
            [
                {"nugget_id": str(index), "label": label}
                for index, label in enumerate(generation.LABELS)
            ],
        ]
        self.similarity = SimpleNamespace(score=Mock(return_value=[0.8, -0.2]))
        self.judge = SimpleNamespace(judge=Mock(side_effect=self.judgments))

    def test_macro_micro_and_separate_unweighted_labels_with_jsonl_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_query.jsonl"
            report = generation.evaluate_answers(
                self.answers, self.similarity, self.judge, path
            )
            self.assertEqual(read_jsonl(path), report["per_query"])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
        self.assertEqual(report["num_queries"], 2)
        self.assertEqual(report["num_nuggets"], 5)
        self.assertEqual(
            report["nugget_counts"],
            {
                "fully_covered": 2,
                "partially_covered": 1,
                "absent": 1,
                "contradicted": 1,
            },
        )
        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["jina_cosine_similarity"], 0.3)
        self.assertEqual(metrics["nugget_coverage"], (1.0 + 0.25) / 2)
        self.assertEqual(metrics["nugget_coverage_micro"], 2 / 5)
        self.assertNotEqual(
            metrics["nugget_coverage"], metrics["nugget_coverage_micro"]
        )
        self.assertEqual(
            set(metrics),
            {
                "jina_cosine_similarity",
                "nugget_coverage",
                "nugget_coverage_micro",
                "nugget_partially_covered_rate",
                "nugget_absent_rate",
                "nugget_contradicted_rate",
            },
        )
        for label in generation.LABELS[1:]:
            self.assertEqual(metrics[f"nugget_{label}_rate"], 0.125)
        for index, result in enumerate(report["per_query"]):
            self.assertEqual(result["nugget_judgments"], self.judgments[index])
            for key, value in self.answers[index].items():
                self.assertEqual(result[key], value)
            rates = result["metrics"]
            self.assertEqual(rates["nugget_coverage"], (1.0, 0.25)[index])
            self.assertAlmostEqual(
                rates["nugget_coverage"]
                + sum(rates[f"nugget_{label}_rate"] for label in generation.LABELS[1:]),
                1.0,
            )
        self.similarity.score.assert_called_once_with(
            [row["answer"] for row in self.answers],
            [row["gold_answer"] for row in self.answers],
        )
        self.assertEqual(
            [call.args for call in self.judge.judge.call_args_list],
            [(row["question"], row["answer"], row["nuggets"]) for row in self.answers],
        )

    def test_empty_answers_fail_before_any_scoring_or_judging(self):
        with self.assertRaisesRegex(ValueError, "empty answers"):
            generation.evaluate_answers([], self.similarity, self.judge)
        self.similarity.score.assert_not_called()
        self.judge.judge.assert_not_called()

    def test_bad_similarity_length_or_nonfinite_scores_fail_before_judging(self):
        for scores in (
            [],
            [0.1],
            [0.1, 0.2, 0.3],
            [float("nan"), 0.2],
            [0.1, float("inf")],
            [float("-inf"), 0.2],
        ):
            with (
                self.subTest(scores=scores),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.similarity.score.return_value = scores
                path = Path(directory) / "per_query.jsonl"
                with self.assertRaisesRegex(ValueError, "one finite score per answer"):
                    generation.evaluate_answers(
                        self.answers, self.similarity, self.judge, path
                    )
                self.judge.judge.assert_not_called()
                self.assertFalse(path.exists())

    def test_judgment_ids_must_match_exactly_including_order_and_count(self):
        valid = self.judgments[1]
        invalid = (
            [],
            valid[:-1],
            valid + [valid[0]],
            list(reversed(valid)),
            [{**valid[0], "nugget_id": "unknown"}, *valid[1:]],
            [{**valid[0], "nugget_id": 0}, *valid[1:]],
            [valid[0], valid[0], *valid[2:]],
            [{**valid[0], "label": "unknown"}, *valid[1:]],
        )
        for judgments in invalid:
            with (
                self.subTest(judgments=judgments),
                tempfile.TemporaryDirectory() as directory,
            ):
                judge = SimpleNamespace(judge=Mock(return_value=judgments))
                similarity = SimpleNamespace(score=Mock(return_value=[0.5]))
                path = Path(directory) / "per_query.jsonl"
                with self.assertRaisesRegex(ValueError, "Invalid nugget judgments"):
                    generation.evaluate_answers(
                        self.answers[1:], similarity, judge, path
                    )
                self.assertFalse(path.exists())

    def test_empty_nuggets_cannot_be_evaluated(self):
        with self.assertRaisesRegex(ValueError, "Invalid nugget judgments"):
            generation.evaluate_answers(
                [answer_record(nuggets=[])],
                SimpleNamespace(score=Mock(return_value=[0.5])),
                SimpleNamespace(judge=Mock(return_value=[])),
            )


if __name__ == "__main__":
    unittest.main()
