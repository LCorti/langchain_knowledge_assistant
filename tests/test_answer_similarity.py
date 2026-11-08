"""Model-free tests: python -m unittest discover -s src -p test_answer_similarity.py."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rag.evaluation.answer_similarity import JinaAnswerSimilarity


class JinaAnswerSimilarityTests(unittest.TestCase):
    def scorer(self, vectors, **kwargs):
        scorer = JinaAnswerSimilarity(**kwargs)
        scorer._model = SimpleNamespace(encode=Mock(return_value=vectors))
        return scorer

    def test_normalized_paired_cosine_not_cross_product_or_raw_dot(self):
        scorer = self.scorer(
            [
                [3, 4],
                [0, 2],
                [2, 0],
                [1, 0],
                [6, 8],
                [7, 0],
                [-9, 0],
                [3, 4],
            ]
        )
        scores = scorer.score(["a", "b", "c", "d"], ["ra", "rb", "rc", "rd"])
        np.testing.assert_allclose(scores, [1.0, 0.0, -1.0, 0.6], atol=1e-14)
        self.assertEqual(len(scores), 4)
        self.assertTrue(all(type(score) is float for score in scores))
        self.assertTrue(all(-1.0 <= score <= 1.0 for score in scores))
        scorer._model.encode.assert_called_once()

    def test_text_matching_adapter_kwargs_preserve_original_text_and_pair_order(self):
        scorer = self.scorer(
            [[3, 4], [0, 2], [6, 8], [-3, 0]],
            batch_size=3,
            max_length=128,
        )
        answers = ["  fetch(timeout=5) [1]\n", "\t ", "Other answer [2]"]
        references = ["Reference code\nfetch()", "Skipped reference", "  Other gold  "]
        original_answers, original_references = answers.copy(), references.copy()
        np.testing.assert_allclose(scorer.score(answers, references), [1.0, 0.0, 0.0])
        scorer._model.encode.assert_called_once_with(
            [answers[0], answers[2], references[0], references[2]],
            task="text-matching",
            batch_size=3,
            max_length=128,
            convert_to_numpy=True,
        )
        self.assertEqual(answers, original_answers)
        self.assertEqual(references, original_references)

    def test_empty_inputs_or_all_blank_answers_return_zero_without_loading(self):
        for answers, references, expected in (
            ([], [], []),
            ([""], ["gold"], [0.0]),
            (["", " \n\t"], ["gold", "other gold"], [0.0, 0.0]),
        ):
            with self.subTest(answers=answers):
                scorer = JinaAnswerSimilarity()
                with patch.object(
                    scorer, "_load", side_effect=AssertionError("model must not load")
                ) as load:
                    self.assertEqual(scorer.score(answers, references), expected)
                load.assert_not_called()
                self.assertIsNone(scorer._model)

    def test_invalid_inputs_fail_before_loading_even_for_empty_answers(self):
        cases = (
            (["answer"], []),
            ([], ["gold"]),
            (["a", "b"], ["gold"]),
            ([None], ["gold"]),
            ([123], ["gold"]),
            ([{}], ["gold"]),
            (["answer"], [None]),
            (["answer"], [123]),
            (["answer"], [""]),
            (["answer"], [" \n\t"]),
            ([""], [""]),
            ([""], [None]),
            (["valid", ""], ["gold", " "]),
        )
        for answers, references in cases:
            with self.subTest(answers=answers, references=references):
                scorer = JinaAnswerSimilarity()
                with (
                    patch.object(
                        scorer,
                        "_load",
                        side_effect=AssertionError("model must not load"),
                    ) as load,
                    self.assertRaises(ValueError),
                ):
                    scorer.score(answers, references)
                load.assert_not_called()
                self.assertIsNone(scorer._model)

    def test_constructor_validates_limits_without_loading(self):
        for kwargs in (
            {"batch_size": 0},
            {"batch_size": -1},
            {"batch_size": True},
            {"batch_size": 1.5},
            {"batch_size": "8"},
            {"batch_size": None},
            {"max_length": 0},
            {"max_length": -1},
            {"max_length": 8193},
            {"max_length": False},
            {"max_length": 128.0},
            {"max_length": None},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                JinaAnswerSimilarity(**kwargs)
        for max_length in (1, 8192):
            scorer = JinaAnswerSimilarity(batch_size=1, max_length=max_length)
            self.assertIsNone(scorer._model)
            self.assertEqual(scorer.max_length, max_length)
        scorer = JinaAnswerSimilarity()
        self.assertEqual(scorer.device, "cpu")
        self.assertEqual(scorer.batch_size, 8)
        self.assertEqual(scorer.max_length, 8192)
        self.assertIsNone(scorer._model)

    def test_invalid_embedding_shapes(self):
        for vectors in (1.0, [1.0, 2.0], [[1, 0]], [[1, 0]] * 3, [[[1, 0]], [[1, 0]]]):
            with self.subTest(vectors=vectors):
                scorer = self.scorer(vectors)
                with self.assertRaisesRegex(ValueError, "unexpected embedding shape"):
                    scorer.score(["answer"], ["gold"])

    def test_nonfinite_zero_or_overflowing_embeddings_are_rejected(self):
        invalid = (
            [[0, 0], [1, 0]],
            [[1, 0], [0, 0]],
            [[], []],
            [[float("nan"), 0], [1, 0]],
            [[1, 0], [float("inf"), 0]],
            [[float("-inf"), 0], [1, 0]],
            [[1e308, 1e308], [1, 0]],
        )
        for vectors in invalid:
            with (
                self.subTest(vectors=vectors),
                np.errstate(over="ignore", invalid="ignore"),
            ):
                scorer = self.scorer(vectors)
                with self.assertRaisesRegex(
                    ValueError, "nonfinite or zero-length embeddings"
                ):
                    scorer.score(["answer"], ["gold"])

    def test_roundoff_is_clamped_to_cosine_range(self):
        for dot, expected in ((1.0000000000000002, 1.0), (-1.0000000000000002, -1.0)):
            with self.subTest(dot=dot):
                scorer = self.scorer([[1, 0], [1, 0]])
                with patch.object(np, "dot", return_value=dot):
                    self.assertEqual(scorer.score(["answer"], ["gold"]), [expected])

    def test_nonfinite_dot_product_is_rejected(self):
        for dot in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(dot=dot):
                scorer = self.scorer([[1, 0], [1, 0]])
                with (
                    patch.object(np, "dot", return_value=dot),
                    self.assertRaisesRegex(ValueError, "Nonfinite answer similarity"),
                ):
                    scorer.score(["answer"], ["gold"])

    def test_injected_encoder_is_reused_without_importing_transformers(self):
        scorer = self.scorer([[1, 0], [1, 0]])
        encoder = scorer._model
        with patch.dict(sys.modules, {"transformers": None}):
            self.assertIs(scorer._load(), encoder)
            self.assertEqual(scorer.score(["answer"], ["gold"]), [1.0])
            self.assertEqual(scorer.score(["another"], ["reference"]), [1.0])
        self.assertIs(scorer._model, encoder)
        self.assertEqual(encoder.encode.call_count, 2)

    def test_lazy_loader_configuration_and_cache_with_stubbed_transformers(self):
        transformers = ModuleType("transformers")
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        transformers.AutoModel = SimpleNamespace(
            from_pretrained=Mock(return_value=model)
        )
        with patch.dict(sys.modules, {"transformers": transformers}):
            scorer = JinaAnswerSimilarity(
                model_id="fake-model", device="cpu", revision="pinned-revision"
            )
            transformers.AutoModel.from_pretrained.assert_not_called()
            self.assertIs(scorer._load(), model)
            self.assertIs(scorer._load(), model)
        transformers.AutoModel.from_pretrained.assert_called_once_with(
            "fake-model", trust_remote_code=True, revision="pinned-revision"
        )
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
