"""Model-free contracts for standalone saved-answer citation evaluation."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from rag import config as project_config
from rag.evaluation import generation
from rag.evaluation.citation_judge import QwenCitationJudge
from rag.evaluation.citation_parsing import QwenCitationExtractor


def answer_record(**overrides):
    return {
        "query_id": "q-1",
        "question": "How do I fetch?",
        "answer": "Use a timeout. [1]",
        "sources": [{"citation": 1, "text": "Use a timeout."}],
        **overrides,
    }


def unit(claim="Use a timeout.", refs=None, span=None):
    return {
        "claim": claim,
        "answer_span": span if span is not None else claim + " [1]",
        "citation_numbers": [1] if refs is None else refs,
    }


def extraction(claims, abstained=False):
    return {
        "claims": claims,
        "abstained": abstained,
        "raw_response": "raw extraction",
    }


def verdict(label="fully_supported"):
    return {
        "label": label,
        "evidence": []
        if label == "insufficient_evidence"
        else [{"citation": 1, "quote": "Use a timeout."}],
        "reason": "Source explanation.",
        "raw_response": "raw judgment",
    }


def read_jsonl(path):
    # File iteration, unlike splitlines(), preserves Unicode line separators.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class TemporaryFilesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.answers_path = self.directory / "input.jsonl"
        self.enterContext(redirect_stdout(io.StringIO()))

    def save(self, records):
        with self.answers_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.answers_path


class LoadCitationAnswersTests(TemporaryFilesTests):
    def test_minimal_records_need_no_gold_nuggets_or_document_ids(self):
        rows = [answer_record(), answer_record(query_id="q-2", sources=[])]
        self.save(rows)
        self.assertEqual(generation.load_citation_answers(self.answers_path), rows)
        self.assertEqual(set(rows[0]), {"query_id", "question", "answer", "sources"})

    def test_unicode_newlines_and_separators_are_preserved_verbatim(self):
        text = "  café 雪\nsecond\r\nthird\u0085fourth\u2028fifth\u2029end  "
        row = answer_record(
            question=text,
            answer=text,
            sources=[{"citation": 1, "text": text, "doc_id": "雪"}],
        )
        self.save([row, answer_record(query_id="next")])
        result = generation.load_citation_answers(self.answers_path)
        self.assertEqual(result, [row, answer_record(query_id="next")])
        self.assertEqual(result[0]["sources"][0]["text"], text)

    def test_ids_normalized_to_strings_without_trimming_and_extra_fields_preserved(
        self,
    ):
        rows = [
            answer_record(query_id=0, metadata={"custom": ["x"]}),
            answer_record(query_id=-7),
            answer_record(query_id=" q "),
        ]
        self.save(rows)
        result = generation.load_citation_answers(self.answers_path)
        self.assertEqual([row["query_id"] for row in result], ["0", "-7", " q "])
        self.assertEqual(result[0]["metadata"], {"custom": ["x"]})

    def test_invalid_or_missing_ids(self):
        for value in (None, True, False, "", " \t", 1.5, [], {}):
            with self.subTest(value=value):
                self.save([answer_record(query_id=value)])
                with self.assertRaisesRegex(ValueError, "query_id.*line 1"):
                    generation.load_citation_answers(self.answers_path)
        row = answer_record()
        del row["query_id"]
        self.save([row])
        with self.assertRaisesRegex(ValueError, "query_id"):
            generation.load_citation_answers(self.answers_path)

    def test_duplicate_ids_including_normalized_collision(self):
        for first, second in (("q", "q"), (7, "7"), ("7", 7)):
            with self.subTest(first=first, second=second):
                self.save(
                    [answer_record(query_id=first), answer_record(query_id=second)]
                )
                with self.assertRaisesRegex(ValueError, "duplicate.*line 2"):
                    generation.load_citation_answers(self.answers_path)

    def test_missing_or_nonlist_sources_are_not_reconstructed(self):
        for value in (None, {}, "passage", 1, False):
            with self.subTest(value=value):
                self.save([answer_record(sources=value)])
                with self.assertRaisesRegex(ValueError, "q-1.*saved sources"):
                    generation.load_citation_answers(self.answers_path)
        row = answer_record()
        del row["sources"]
        self.save([row])
        with self.assertRaisesRegex(ValueError, "saved sources"):
            generation.load_citation_answers(self.answers_path)

    def test_source_citations_must_be_consecutive_one_based_integers(self):
        for numbers in (
            [0],
            [-1],
            [2],
            [2, 1],
            [1, 1],
            [1, 3],
            [True],
            [False],
            [1, True],
            [1.0],
            ["1"],
            [None],
        ):
            with self.subTest(numbers=numbers):
                self.save(
                    [
                        answer_record(
                            sources=[
                                {"citation": number, "text": "Passage"}
                                for number in numbers
                            ]
                        )
                    ]
                )
                with self.assertRaisesRegex(ValueError, "q-1 source .*one-based"):
                    generation.load_citation_answers(self.answers_path)

    def test_source_objects_require_nonblank_string_text_and_citation(self):
        bad_sources = [None, [], "passage", 1, {}, {"text": "Passage"}, {"citation": 1}]
        bad_sources += [
            {"citation": 1, "text": text}
            for text in (None, "", " \n\t", 3, False, [], {})
        ]
        for source in bad_sources:
            with self.subTest(source=source):
                self.save([answer_record(sources=[source])])
                with self.assertRaisesRegex(ValueError, "q-1 source 1"):
                    generation.load_citation_answers(self.answers_path)

    def test_question_is_required_nonblank_string(self):
        rows = [
            answer_record(question=value)
            for value in (None, "", " \n", 1, False, [], {})
        ]
        missing = answer_record()
        del missing["question"]
        for row in rows + [missing]:
            with self.subTest(row=row):
                self.save([row])
                with self.assertRaisesRegex(ValueError, "q-1.*question"):
                    generation.load_citation_answers(self.answers_path)

    def test_answer_must_be_present_string_but_may_be_empty(self):
        for value in (None, 1, False, [], {}):
            with self.subTest(value=value):
                self.save([answer_record(answer=value)])
                with self.assertRaisesRegex(ValueError, "q-1.*string answer"):
                    generation.load_citation_answers(self.answers_path)
        missing = answer_record()
        del missing["answer"]
        self.save([missing])
        with self.assertRaisesRegex(ValueError, "string answer"):
            generation.load_citation_answers(self.answers_path)
        self.save(
            [answer_record(answer=""), answer_record(query_id="q-2", answer=" \n")]
        )
        self.assertEqual(
            [
                row["answer"]
                for row in generation.load_citation_answers(self.answers_path)
            ],
            ["", " \n"],
        )

    def test_malformed_json_duplicate_keys_and_nonstandard_constants(self):
        valid = json.dumps(answer_record())
        malformed = [
            "{",
            valid + " trailing",
            valid + valid,
            valid[:-1] + ', "question": "duplicate"}',
            valid.replace('"citation": 1', '"citation": 1, "citation": 1'),
        ]
        malformed += [
            valid[:-1] + ', "extra": ' + constant + "}"
            for constant in ("NaN", "Infinity", "-Infinity")
        ]
        for text in malformed:
            with self.subTest(text=text):
                self.answers_path.write_text("\n" + text + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid saved JSON") as caught:
                    generation.load_citation_answers(self.answers_path)
                self.assertIn(f"{self.answers_path}:2:", str(caught.exception))
                self.assertIsNotNone(caught.exception.__cause__)

    def test_top_level_json_must_be_object(self):
        for value in ([], None, "answer", 42, True):
            with self.subTest(value=value):
                self.save([value])
                with self.assertRaisesRegex(ValueError, "line 1 must be an object"):
                    generation.load_citation_answers(self.answers_path)

    def test_limit_preserves_order_and_can_exceed_file_size(self):
        rows = [answer_record(query_id=index) for index in range(3)]
        self.save(rows)
        for limit, expected in (
            (1, ["0"]),
            (2, ["0", "1"]),
            (10, ["0", "1", "2"]),
            (None, ["0", "1", "2"]),
        ):
            with self.subTest(limit=limit):
                self.assertEqual(
                    [
                        row["query_id"]
                        for row in generation.load_citation_answers(
                            self.answers_path, limit
                        )
                    ],
                    expected,
                )

    def test_limit_must_be_positive_integer_before_file_is_opened(self):
        for limit in (0, -1, True, False, 1.0, "1"):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    generation.load_citation_answers(self.answers_path, limit)

    def test_limit_does_not_hide_invalid_later_records_or_collisions(self):
        for later in (
            answer_record(query_id="q-2", sources=None),
            answer_record(query_id="q-1"),
        ):
            with self.subTest(later=later):
                self.save([answer_record(), later])
                with self.assertRaises(ValueError):
                    generation.load_citation_answers(self.answers_path, limit=1)
        self.save([answer_record()])
        with self.answers_path.open("a", encoding="utf-8") as stream:
            stream.write("not JSON\n")
        with self.assertRaisesRegex(ValueError, "Invalid saved JSON"):
            generation.load_citation_answers(self.answers_path, limit=1)

    def test_empty_or_whitespace_only_file_rejected_and_blank_lines_ignored(self):
        for text in ("", "\n \t\n"):
            with self.subTest(text=text):
                self.answers_path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "empty saved answers"):
                    generation.load_citation_answers(self.answers_path)
        self.answers_path.write_text(
            "\n" + json.dumps(answer_record()) + "\n \t\n", encoding="utf-8"
        )
        self.assertEqual(
            generation.load_citation_answers(self.answers_path), [answer_record()]
        )


class EvaluateCitationsTests(TemporaryFilesTests):
    def setUp(self):
        super().setUp()
        self.extractor = Mock()
        self.judge = Mock()
        self.judge.judge.return_value = verdict()

    def evaluate(self, rows, extractions, output_path=None):
        self.extractor.extract.side_effect = extractions
        return generation.evaluate_citations(
            rows, self.extractor, self.judge, output_path
        )

    def test_exact_question_claim_and_only_valid_cited_passages_in_occurrence_order(
        self,
    ):
        row = answer_record(
            question="  Question 雪?\nDo not trim. ",
            answer="Claim. [2, 99][1][2]",
            gold_answer="NEVER SEND GOLD",
            nuggets=[{"text": "NEVER SEND NUGGET"}],
            sources=[
                {"citation": 1, "text": "  first\n雪  ", "doc_id": "d1"},
                {"citation": 2, "text": "second\u2028passage", "secret": "metadata"},
                {"citation": 3, "text": "UNCITED PASSAGE"},
            ],
        )
        claim = unit("Claim.", [2, 99, 1, 2], row["answer"])
        report = self.evaluate([row], [extraction([claim])])
        self.extractor.extract.assert_called_once_with(row["answer"])
        self.judge.judge.assert_called_once_with(
            row["question"],
            "Claim.",
            [
                {"citation": 2, "text": "second\u2028passage"},
                {"citation": 1, "text": "  first\n雪  "},
            ],
        )
        result = report["per_query"][0]
        self.assertEqual(
            result["citation_claims"][0],
            {
                **claim,
                "claim_id": 1,
                "invalid_citation_numbers": [99],
                **verdict(),
                "skip_reason": None,
            },
        )
        self.assertEqual(result["extraction_raw_response"], "raw extraction")
        self.assertEqual(report["counts"]["num_references"], 4)
        self.assertEqual(report["counts"]["num_valid_references"], 3)

    def test_duplicate_refs_count_occurrences_but_passages_are_deduplicated(self):
        report = self.evaluate([answer_record()], [extraction([unit(refs=[1, 1, 1])])])
        self.judge.judge.assert_called_once_with(
            "How do I fetch?",
            "Use a timeout.",
            [{"citation": 1, "text": "Use a timeout."}],
        )
        self.assertEqual(
            report["counts"],
            {
                "num_claims": 1,
                "num_cited_claims": 1,
                "num_references": 3,
                "num_valid_references": 3,
                "num_fully_supported_claims": 1,
                "num_contradicted_claims": 0,
            },
        )
        self.assertEqual(
            report["per_query"][0]["citation_claims"][0]["citation_numbers"], [1, 1, 1]
        )

    def test_mixed_valid_invalid_group_has_verdict_but_no_full_support_credit(self):
        report = self.evaluate([answer_record()], [extraction([unit(refs=[1, 99])])])
        claim = report["per_query"][0]["citation_claims"][0]
        self.assertEqual(claim["label"], "fully_supported")
        self.assertEqual(claim["invalid_citation_numbers"], [99])
        self.assertIsNone(claim["skip_reason"])
        self.assertEqual(report["counts"]["num_fully_supported_claims"], 0)
        self.assertEqual(report["metrics"]["citation_validity"], 0.5)
        self.assertEqual(report["metrics"]["cited_claim_support"], 0)
        self.assertEqual(report["metrics"]["supported_citation_coverage"], 0)

    def test_uncited_and_all_invalid_groups_skip_judge_without_fabricated_verdict(self):
        units = [unit("Uncited.", [], "Uncited."), unit("Invalid.", [0, 99, 99])]
        report = self.evaluate([answer_record()], [extraction(units)])
        self.judge.judge.assert_not_called()
        claims = report["per_query"][0]["citation_claims"]
        for index, (claim, reason, invalid) in enumerate(
            zip(
                claims,
                ("uncited", "invalid_references"),
                ([], [0, 99, 99]),
                strict=True,
            ),
            1,
        ):
            self.assertEqual(claim["claim_id"], index)
            self.assertEqual(claim["skip_reason"], reason)
            self.assertEqual(claim["invalid_citation_numbers"], invalid)
            self.assertIsNone(claim["label"])
            self.assertEqual(claim["evidence"], [])
            self.assertEqual(claim["reason"], "")
            self.assertIsNone(claim["raw_response"])
        self.assertEqual(report["counts"]["num_claims"], 2)
        self.assertEqual(report["counts"]["num_cited_claims"], 1)
        self.assertEqual(report["counts"]["num_references"], 3)
        self.assertTrue(all(value == 0 for value in report["metrics"].values()))

    def test_all_four_labels_retained_without_treating_partial_as_support(self):
        labels = (
            "fully_supported",
            "partially_supported",
            "contradicted",
            "insufficient_evidence",
        )
        self.judge.judge.side_effect = [verdict(label) for label in labels]
        report = self.evaluate(
            [answer_record()],
            [extraction([unit(f"Claim {index}.") for index in range(4)])],
        )
        claims = report["per_query"][0]["citation_claims"]
        self.assertEqual([claim["label"] for claim in claims], list(labels))
        self.assertEqual([claim["claim_id"] for claim in claims], [1, 2, 3, 4])
        self.assertEqual(self.judge.judge.call_count, 4)
        self.assertEqual(report["counts"]["num_fully_supported_claims"], 1)
        self.assertEqual(report["counts"]["num_contradicted_claims"], 1)
        self.assertEqual(report["metrics"]["cited_claim_support"], 0.25)
        self.assertEqual(report["metrics"]["supported_citation_coverage"], 0.25)
        self.assertEqual(report["metrics"]["contradiction_rate"], 0.25)

    def test_mixed_invalid_contradicted_group_still_counts_contradiction(self):
        self.judge.judge.return_value = verdict("contradicted")
        report = self.evaluate([answer_record()], [extraction([unit(refs=[1, 99])])])
        self.assertEqual(report["counts"]["num_contradicted_claims"], 1)
        self.assertEqual(report["metrics"]["contradiction_rate"], 1)
        self.assertEqual(report["counts"]["num_fully_supported_claims"], 0)

    def test_pure_abstention_empty_and_nonfactual_answers_have_zeros_separate_counts(
        self,
    ):
        rows = [
            answer_record(answer="I cannot answer without evidence."),
            answer_record(query_id="empty", answer=" \n\t"),
            answer_record(query_id="greeting", answer="Hello!"),
        ]
        report = self.evaluate(
            rows, [extraction([], True), extraction([]), extraction([])]
        )
        self.judge.judge.assert_not_called()
        self.assertEqual(report["num_queries"], 3)
        self.assertEqual(report["num_abstained_answers"], 1)
        self.assertEqual(report["num_empty_answers"], 1)
        self.assertEqual(
            [row["abstained"] for row in report["per_query"]], [True, False, False]
        )
        self.assertEqual(
            [row["empty_answer"] for row in report["per_query"]], [False, True, False]
        )
        self.assertTrue(all(value == 0 for value in report["counts"].values()))
        self.assertTrue(all(value == 0 for value in report["metrics"].values()))
        for row in report["per_query"]:
            self.assertEqual(row["citation_claims"], [])
            self.assertTrue(all(value == 0 for value in row["metrics"].values()))

    def test_mixed_abstention_retains_and_judges_factual_claims(self):
        row = answer_record(answer="I cannot answer all of this. Use a timeout. [1]")
        report = self.evaluate([row], [extraction([unit()], abstained=True)])
        self.assertEqual(report["num_abstained_answers"], 1)
        self.assertEqual(report["num_empty_answers"], 0)
        self.assertEqual(report["counts"]["num_claims"], 1)
        self.assertEqual(report["metrics"]["supported_citation_coverage"], 1)
        self.judge.judge.assert_called_once()

    def test_macro_includes_zero_answers_and_micro_pools_claims_and_reference_occurrences(
        self,
    ):
        rows = [answer_record(query_id=str(index)) for index in range(4)]
        rows[2]["answer"] = "I cannot answer."
        rows[3]["answer"] = ""
        self.judge.judge.side_effect = [
            verdict(),
            verdict("contradicted"),
            verdict("partially_supported"),
        ]
        report = self.evaluate(
            rows,
            [
                extraction([unit()]),
                extraction([unit(refs=[1, 99]), unit(refs=[1, 1]), unit(refs=[])]),
                extraction([], True),
                extraction([]),
            ],
        )
        self.assertEqual(
            report["counts"],
            {
                "num_claims": 4,
                "num_cited_claims": 3,
                "num_references": 5,
                "num_valid_references": 4,
                "num_fully_supported_claims": 1,
                "num_contradicted_claims": 1,
            },
        )
        expected_macro = {
            "citation_validity": 7 / 16,
            "cited_claim_support": 1 / 4,
            "supported_citation_coverage": 1 / 4,
            "contradiction_rate": 1 / 12,
        }
        expected_micro = {
            "citation_validity": 4 / 5,
            "cited_claim_support": 1 / 3,
            "supported_citation_coverage": 1 / 4,
            "contradiction_rate": 1 / 4,
        }
        self.assertEqual(
            set(report["metrics"]),
            set(expected_macro) | {f"{key}_micro" for key in expected_micro},
        )
        for key, value in expected_macro.items():
            self.assertAlmostEqual(report["metrics"][key], value)
            self.assertAlmostEqual(
                report["metrics"][f"{key}_micro"], expected_micro[key]
            )

    def test_checkpoint_contains_only_completed_queries_after_later_failure(self):
        for stage in ("extraction", "judgment"):
            with self.subTest(stage=stage):
                output = self.directory / f"{stage}.jsonl"
                error = RuntimeError("later failure")
                self.extractor.extract.side_effect = [
                    extraction([unit()]),
                    error if stage == "extraction" else extraction([unit()]),
                ]
                self.judge.judge.side_effect = [verdict(), error]
                with self.assertRaises(RuntimeError) as caught:
                    generation.evaluate_citations(
                        [answer_record(), answer_record(query_id="q-2")],
                        self.extractor,
                        self.judge,
                        output,
                    )
                self.assertIs(caught.exception, error)
                expected_note = (
                    "Citation extraction for query q-2"
                    if stage == "extraction"
                    else "Citation judgment for query q-2, claim 1"
                )
                self.assertIn(expected_note, error.__notes__)
                saved = read_jsonl(output)
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["query_id"], "q-1")
                self.assertEqual(saved[0]["counts"]["num_fully_supported_claims"], 1)

    def test_judge_exception_note_identifies_actual_claim_index_and_no_partial_query(
        self,
    ):
        error = ValueError("bad judgment")
        self.extractor.extract.return_value = extraction([unit(refs=[]), unit()])
        self.judge.judge.side_effect = error
        output = self.directory / "failed.jsonl"
        with self.assertRaises(ValueError) as caught:
            generation.evaluate_citations(
                [answer_record()], self.extractor, self.judge, output
            )
        self.assertIs(caught.exception, error)
        self.assertIn("Citation judgment for query q-1, claim 2", error.__notes__)
        self.assertFalse(output.exists())

    def test_inputs_and_collaborator_results_not_mutated_and_checkpoint_matches_report(
        self,
    ):
        rows = [answer_record(sources=[{"citation": 1, "text": "雪\u2028line\nnext"}])]
        extracted = extraction([unit(refs=[1, 1, 99])])
        judged = verdict()
        before = copy.deepcopy((rows, extracted, judged))
        self.judge.judge.return_value = judged
        output = self.directory / "per_query.jsonl"
        report = self.evaluate(rows, [extracted], output)
        self.assertEqual((rows, extracted, judged), before)
        self.assertEqual(read_jsonl(output), report["per_query"])
        self.assertEqual(report["per_query"][0]["sources"], rows[0]["sources"])

    def test_real_extractor_preserves_disabled_claim_in_actual_judge_payload(self):
        row = answer_record(
            question="Is caching enabled?",
            answer="Caching is disabled. [1]",
            sources=[{"citation": 1, "text": "Caching is enabled."}],
        )
        model_extraction = json.dumps(
            {
                "claims": [unit("Caching is enabled.", [1], row["answer"])],
                "abstained": False,
            }
        )
        model_judgment = json.dumps(
            {
                "label": "contradicted",
                "evidence": [{"citation": 1, "quote": "Caching is enabled."}],
                "reason": "The passage says enabled, but the answer says disabled.",
            }
        )
        generator = Mock()
        generator.generate.side_effect = [model_extraction, model_judgment]
        extractor = QwenCitationExtractor(generator, max_retries=0)
        judge = QwenCitationJudge(generator, max_retries=0)

        report = generation.evaluate_citations([row], extractor, judge)

        calls = generator.generate.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            json.loads(calls[0].args[0][1]["content"]), {"answer": row["answer"]}
        )
        self.assertEqual(
            json.loads(calls[1].args[0][1]["content"]),
            {
                "question": row["question"],
                "claim": "Caching is disabled.",
                "cited_passages": row["sources"],
            },
        )
        result = report["per_query"][0]
        claim = result["citation_claims"][0]
        self.assertEqual(claim["claim"], "Caching is disabled.")
        self.assertEqual(claim["answer_span"], row["answer"])
        self.assertEqual(claim["citation_numbers"], [1])
        self.assertEqual(claim["invalid_citation_numbers"], [])
        self.assertEqual(claim["label"], "contradicted")
        self.assertEqual(claim["evidence"], json.loads(model_judgment)["evidence"])
        self.assertEqual(claim["raw_response"], model_judgment)
        self.assertEqual(result["extraction_raw_response"], model_extraction)
        self.assertEqual(
            json.loads(result["extraction_raw_response"])["claims"][0]["claim"],
            "Caching is enabled.",
        )
        self.assertEqual(report["counts"]["num_contradicted_claims"], 1)
        self.assertEqual(report["counts"]["num_fully_supported_claims"], 0)
        self.assertEqual(report["metrics"]["contradiction_rate"], 1)
        self.assertEqual(report["metrics"]["supported_citation_coverage"], 0)

    def test_empty_input_fails_before_extraction_judging_or_checkpoint(self):
        output = self.directory / "empty.jsonl"
        with self.assertRaisesRegex(ValueError, "empty answers"):
            generation.evaluate_citations([], self.extractor, self.judge, output)
        self.extractor.extract.assert_not_called()
        self.judge.judge.assert_not_called()
        self.assertFalse(output.exists())


class CitationCliTests(TemporaryFilesTests):
    def setUp(self):
        super().setUp()
        self.output = self.directory / "new-run"
        self.generator = Mock()
        self.generator_constructor = Mock(return_value=self.generator)
        generator_module = ModuleType("rag.generate")
        generator_module.QwenGenerator = self.generator_constructor
        self.forbidden = {}
        modules = {"rag.generate": generator_module}
        for module_name, symbol in (
            ("rag.evaluation.prepare_queries", "prepare_from_config"),
            ("rag.retriever", "HybridRetriever"),
            ("rag.pipeline", "RAGPipeline"),
        ):
            module = ModuleType(module_name)
            constructor = Mock(side_effect=AssertionError(f"Must not call {symbol}"))
            setattr(module, symbol, constructor)
            modules[module_name] = module
            self.forbidden[symbol] = constructor
        self.enterContext(patch.dict(sys.modules, modules))
        for name in (
            "JinaAnswerSimilarity",
            "QwenNuggetJudge",
            "prepare_rows",
            "generate_answers",
            "load_answers",
            "evaluate_answers",
        ):
            self.forbidden[name] = self.enterContext(
                patch.object(
                    generation,
                    name,
                    side_effect=AssertionError(f"Must not call {name}"),
                )
            )
        self.settings = {
            "results_dir": str(self.directory / "results"),
            "generator_config": "generator-test.yaml",
        }
        self.generator_config = {"model_id": "fake-model", "device": "cpu"}
        self.config = SimpleNamespace(
            GENERATOR_CONFIG_PATH=Path("default-generator.yaml"),
            resolve_project_path=Mock(side_effect=Path),
            get_generation_evaluation_config=Mock(return_value=self.settings),
            get_generator_config=Mock(return_value=self.generator_config),
        )
        for name in ("get_bm25_config", "get_faiss_config", "get_retriever_config"):
            blocked = Mock(side_effect=AssertionError(f"Must not call {name}"))
            setattr(self.config, name, blocked)
            self.forbidden[name] = blocked
        self.enterContext(patch.object(generation, "project_config", self.config))
        self.addCleanup(self.assert_no_generation_or_retrieval)

    def assert_no_generation_or_retrieval(self):
        for name, blocked in self.forbidden.items():
            with self.subTest(forbidden=name):
                blocked.assert_not_called()

    def arguments(self, *extra):
        return ["--citations-only", "--answers", str(self.answers_path), *extra]

    def script_success(self):
        extracted = {"claims": [unit()], "abstained": False}
        judged = {
            key: value for key, value in verdict().items() if key != "raw_response"
        }
        self.generator.generate.side_effect = [
            json.dumps(extracted),
            json.dumps(judged),
        ]

    def test_requires_answers_and_disallows_overwrite_before_config_or_models(self):
        for arguments, message in (
            (["--citations-only"], "--citations-only requires --answers"),
            (self.arguments("--overwrite"), "cannot be used with --citations-only"),
        ):
            with (
                self.subTest(arguments=arguments),
                redirect_stderr(io.StringIO()) as stderr,
            ):
                with self.assertRaises(SystemExit) as caught:
                    generation.main(arguments)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        self.config.get_generation_evaluation_config.assert_not_called()
        self.config.get_generator_config.assert_not_called()
        self.generator_constructor.assert_not_called()

    def test_nonpositive_cli_limit_rejected_before_config(self):
        for limit in ("0", "-1"):
            with self.subTest(limit=limit), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as caught:
                    generation.main(self.arguments("--limit", limit))
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--limit must be positive", stderr.getvalue())
        self.config.get_generation_evaluation_config.assert_not_called()
        self.generator_constructor.assert_not_called()

    def test_missing_evidence_validated_before_generator_even_beyond_limit(self):
        missing = answer_record(query_id="q-2")
        del missing["sources"]
        for bad_row in (
            missing,
            answer_record(query_id="q-2", sources=None),
            answer_record(query_id="q-2", sources=[{"citation": 1}]),
        ):
            with self.subTest(bad_row=bad_row):
                self.save([answer_record(), bad_row])
                with self.assertRaisesRegex(ValueError, "q-2"):
                    generation.main(
                        self.arguments("--limit", "1", "--output-dir", str(self.output))
                    )
        self.config.get_generator_config.assert_not_called()
        self.generator_constructor.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_minimal_config_real_extractor_and_judge_write_all_artifacts(self):
        row = answer_record(query_id=7)
        self.save([row])
        original_input = self.answers_path.read_bytes()
        self.script_success()
        generation.main(
            self.arguments(
                "--config", "evaluation-test.yaml", "--output-dir", str(self.output)
            )
        )
        self.config.get_generation_evaluation_config.assert_called_once_with(
            Path("evaluation-test.yaml")
        )
        self.config.get_generator_config.assert_called_once_with("generator-test.yaml")
        self.generator_constructor.assert_called_once_with(self.generator_config)
        self.assertEqual(self.answers_path.read_bytes(), original_input)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"settings.json", "answers.jsonl", "per_query.jsonl", "summary.json"},
        )
        saved_answers = read_jsonl(self.output / "answers.jsonl")
        self.assertEqual(saved_answers, [{**row, "query_id": "7"}])
        settings = json.loads(
            (self.output / "settings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(settings["evaluation"], "citation_support")
        self.assertEqual(settings["answers_input"], str(self.answers_path))
        self.assertEqual(settings["query_ids"], ["7"])
        self.assertEqual(settings["generator"], self.generator_config)
        self.assertEqual(
            settings["extractor"], {"max_new_tokens": 4096, "max_retries": 1}
        )
        self.assertEqual(settings["judge"], {"max_new_tokens": 1024, "max_retries": 1})
        self.assertEqual(settings["policies"]["zero_denominator"], 0)
        per_query = read_jsonl(self.output / "per_query.jsonl")
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(per_query), 1)
        self.assertEqual(per_query[0]["query_id"], "7")
        self.assertEqual(per_query[0]["citation_claims"][0]["label"], "fully_supported")
        self.assertEqual(summary["num_queries"], 1)
        self.assertEqual(summary["num_abstained_answers"], 0)
        self.assertEqual(summary["num_empty_answers"], 0)
        self.assertEqual(summary["counts"], per_query[0]["counts"])
        self.assertEqual(summary["metrics"]["supported_citation_coverage"], 1)
        self.assertNotIn("per_query", summary)
        calls = self.generator.generate.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            json.loads(calls[0].args[0][1]["content"]), {"answer": row["answer"]}
        )
        self.assertEqual(
            json.loads(calls[1].args[0][1]["content"]),
            {
                "question": row["question"],
                "claim": "Use a timeout.",
                "cited_passages": row["sources"],
            },
        )
        for invocation, tokens in zip(calls, (4096, 1024), strict=True):
            self.assertEqual(
                invocation.kwargs,
                {
                    "enable_thinking": False,
                    "max_new_tokens": tokens,
                    "generation_kwargs": {"do_sample": False, "num_beams": 1},
                },
            )

    def test_default_output_dir_limit_and_custom_citation_settings(self):
        self.settings["citations"] = {
            "extractor": {"max_new_tokens": 123, "max_retries": 0},
            "judge": {"max_new_tokens": 456, "max_retries": 0},
        }
        self.save([answer_record(), answer_record(query_id="q-2")])
        self.script_success()
        generation.main(self.arguments("--limit", "1"))
        self.config.get_generation_evaluation_config.assert_called_once_with()
        outputs = list(Path(self.settings["results_dir"]).iterdir())
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].name.startswith("citations_"))
        settings = json.loads(
            (outputs[0] / "settings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(settings["query_ids"], ["q-1"])
        self.assertEqual(settings["extractor"], self.settings["citations"]["extractor"])
        self.assertEqual(settings["judge"], self.settings["citations"]["judge"])
        self.assertEqual(len(read_jsonl(outputs[0] / "answers.jsonl")), 1)
        self.assertEqual(
            [
                entry.kwargs["max_new_tokens"]
                for entry in self.generator.generate.call_args_list
            ],
            [123, 456],
        )

    def test_default_generator_config_path_requires_no_dataset_or_split_settings(self):
        del self.settings["generator_config"]
        self.save([answer_record(answer="", sources=[])])
        generation.main(self.arguments("--output-dir", str(self.output)))
        self.config.get_generator_config.assert_called_once_with(
            self.config.GENERATOR_CONFIG_PATH
        )
        self.generator.generate.assert_not_called()
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["num_empty_answers"], 1)
        self.assertTrue(all(value == 0 for value in summary["metrics"].values()))

    def test_real_config_getters_accept_minimal_yaml_without_cache_or_datasets(self):
        config_path = self.directory / "evaluation.yaml"
        generator_path = self.directory / "generator.yaml"
        results_dir = self.directory / "real-config-results"
        config_path.write_text(
            f"results_dir: {json.dumps(str(results_dir))}\n"
            f"generator_config: {json.dumps(str(generator_path))}\n",
            encoding="utf-8",
        )
        generator_path.write_text(
            "model_id: fake-model\ndevice: cpu\n", encoding="utf-8"
        )
        row = answer_record()
        self.save([row])
        self.script_success()

        with patch.object(generation, "project_config", project_config):
            for name in ("get_bm25_config", "get_faiss_config", "get_retriever_config"):
                self.enterContext(
                    patch.object(project_config, name, self.forbidden[name])
                )
            generation.main(self.arguments("--config", str(config_path)))

        self.config.get_generation_evaluation_config.assert_not_called()
        self.config.get_generator_config.assert_not_called()
        self.generator_constructor.assert_called_once_with(self.generator_config)
        self.assertEqual(self.generator.generate.call_count, 2)
        outputs = list(results_dir.iterdir())
        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self.assertTrue(output.name.startswith("citations_"))
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {"settings.json", "answers.jsonl", "per_query.jsonl", "summary.json"},
        )
        settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(settings["evaluation"], "citation_support")
        self.assertEqual(settings["generator"], self.generator_config)
        self.assertEqual(settings["answers_input"], str(self.answers_path.resolve()))
        self.assertEqual(settings["query_ids"], [row["query_id"]])
        self.assertEqual(read_jsonl(output / "answers.jsonl"), [row])
        per_query = read_jsonl(output / "per_query.jsonl")
        self.assertEqual(len(per_query), 1)
        self.assertEqual(per_query[0]["citation_claims"][0]["claim"], "Use a timeout.")
        self.assertEqual(per_query[0]["citation_claims"][0]["label"], "fully_supported")
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["num_queries"], 1)
        self.assertEqual(summary["counts"]["num_fully_supported_claims"], 1)
        self.assertEqual(summary["metrics"]["supported_citation_coverage"], 1)

    def test_existing_output_directory_is_never_overwritten(self):
        self.save([answer_record()])
        self.script_success()
        generation.main(self.arguments("--output-dir", str(self.output)))
        sentinel = self.output / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            generation.main(self.arguments("--output-dir", str(self.output)))
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.output.iterdir()}, before
        )
        self.assertEqual(self.generator.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
