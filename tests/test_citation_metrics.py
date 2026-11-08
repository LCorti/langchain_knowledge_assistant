"""Model-free tests for occurrence validity and conservative group support."""

import copy
import itertools
import subprocess
import sys
import unittest
from pathlib import Path

from rag.evaluation.citation_metrics import compute_citation_metrics


def record(refs, invalid=None, label=None):
    return {
        "citation_numbers": refs,
        "invalid_citation_numbers": [] if invalid is None else invalid,
        "label": label,
    }


class CitationMetricsTests(unittest.TestCase):
    def test_empty_and_abstention_claim_lists_are_all_zero(self):
        # The API accepts extracted claims, so a pure abstention is also [].
        result = compute_citation_metrics([])
        self.assertEqual(
            result,
            {
                "counts": {
                    "num_claims": 0,
                    "num_cited_claims": 0,
                    "num_references": 0,
                    "num_valid_references": 0,
                    "num_fully_supported_claims": 0,
                    "num_contradicted_claims": 0,
                },
                "metrics": {
                    "citation_validity": 0.0,
                    "cited_claim_support": 0.0,
                    "supported_citation_coverage": 0.0,
                    "contradiction_rate": 0.0,
                },
            },
        )

    def test_uncited_claims_do_not_get_perfect_scores(self):
        result = compute_citation_metrics([record([]), record([])])
        self.assertEqual(result["counts"]["num_claims"], 2)
        self.assertEqual(result["counts"]["num_cited_claims"], 0)
        self.assertTrue(all(value == 0 for value in result["metrics"].values()))

    def test_all_invalid_claims_are_cited_but_never_supported(self):
        result = compute_citation_metrics([record([0, 99, 99], [0, 99, 99])])
        self.assertEqual(
            result["counts"],
            {
                "num_claims": 1,
                "num_cited_claims": 1,
                "num_references": 3,
                "num_valid_references": 0,
                "num_fully_supported_claims": 0,
                "num_contradicted_claims": 0,
            },
        )
        self.assertTrue(all(value == 0 for value in result["metrics"].values()))

    def test_combined_counts_and_exact_denominators(self):
        claims = [
            record([1, 2], label="fully_supported"),
            record([3, 99], [99], "fully_supported"),
            record([99, 99], [99, 99]),
            record([]),
            record([4], label="partially_supported"),
            record([5, 100], [100], "contradicted"),
            record([6], label="insufficient_evidence"),
        ]
        result = compute_citation_metrics(claims)
        self.assertEqual(
            result["counts"],
            {
                "num_claims": 7,
                "num_cited_claims": 6,
                "num_references": 10,
                "num_valid_references": 6,
                "num_fully_supported_claims": 1,
                "num_contradicted_claims": 1,
            },
        )
        self.assertEqual(
            result["metrics"],
            {
                "citation_validity": 6 / 10,
                "cited_claim_support": 1 / 6,
                "supported_citation_coverage": 1 / 7,
                "contradiction_rate": 1 / 7,
            },
        )

    def test_duplicate_occurrences_are_not_deduplicated(self):
        claims = [
            record([1, 1, 1, 2, 2], [2, 2], "fully_supported"),
            record([3, 3], label="fully_supported"),
        ]
        result = compute_citation_metrics(claims)
        self.assertEqual(result["counts"]["num_references"], 7)
        self.assertEqual(result["counts"]["num_valid_references"], 5)
        self.assertEqual(result["metrics"]["citation_validity"], 5 / 7)
        self.assertEqual(result["metrics"]["cited_claim_support"], 1 / 2)
        # Invalid list is a multiset: even one excess occurrence is invalid input.
        for refs, invalid in (
            ([1, 2], [2, 2]),
            ([1, 1], [1, 1, 1]),
            ([], [1]),
            ([1], [2]),
        ):
            with (
                self.subTest(refs=refs, invalid=invalid),
                self.assertRaisesRegex(ValueError, "multiset"),
            ):
                compute_citation_metrics([record(refs, invalid)])

    def test_invalid_subset_is_order_independent(self):
        result = compute_citation_metrics(
            [record([1, 2, 3], [3, 1], "partially_supported")]
        )
        self.assertEqual(result["counts"]["num_valid_references"], 1)
        self.assertEqual(result["metrics"]["citation_validity"], 1 / 3)

    def test_multiple_references_are_one_joint_group_not_individual_credits(self):
        claims = [
            record([1, 2, 3, 4], label="fully_supported"),
            record([5], label="partially_supported"),
        ]
        result = compute_citation_metrics(claims)
        self.assertEqual(result["counts"]["num_cited_claims"], 2)
        self.assertEqual(result["counts"]["num_fully_supported_claims"], 1)
        self.assertEqual(result["metrics"]["cited_claim_support"], 1 / 2)
        self.assertEqual(result["metrics"]["supported_citation_coverage"], 1 / 2)
        self.assertEqual(result["metrics"]["citation_validity"], 1)

    def test_every_label_with_all_valid_or_mixed_references(self):
        for label, mixed in itertools.product(
            (
                "fully_supported",
                "partially_supported",
                "contradicted",
                "insufficient_evidence",
            ),
            (False, True),
        ):
            with self.subTest(label=label, mixed=mixed):
                result = compute_citation_metrics(
                    [record([1, 9], [9] if mixed else [], label)]
                )
                supported = int(label == "fully_supported" and not mixed)
                self.assertEqual(
                    result["counts"]["num_fully_supported_claims"], supported
                )
                self.assertEqual(result["metrics"]["cited_claim_support"], supported)
                self.assertEqual(
                    result["metrics"]["supported_citation_coverage"], supported
                )
                self.assertEqual(
                    result["metrics"]["contradiction_rate"],
                    int(label == "contradicted"),
                )
                self.assertEqual(
                    result["metrics"]["citation_validity"], 0.5 if mixed else 1
                )

    def test_uncited_units_reduce_coverage_and_contradiction_rate_only(self):
        cited = [
            record([1], label="fully_supported"),
            record([2], label="contradicted"),
        ]
        result = compute_citation_metrics(cited + [record([]), record([])])
        self.assertEqual(
            result["metrics"],
            {
                "citation_validity": 1,
                "cited_claim_support": 1 / 2,
                "supported_citation_coverage": 1 / 4,
                "contradiction_rate": 1 / 4,
            },
        )

    def test_missing_verdict_for_any_valid_reference_is_rejected(self):
        for refs, invalid in (([1], []), ([1, 99], [99]), ([1, 1], [1])):
            with (
                self.subTest(refs=refs, invalid=invalid),
                self.assertRaisesRegex(ValueError, "verdict"),
            ):
                compute_citation_metrics([record(refs, invalid)])

    def test_no_verdict_allowed_for_skipped_groups(self):
        for refs, invalid, label in itertools.product(
            ([], [99]),
            ([], [99]),
            (
                "fully_supported",
                "partially_supported",
                "contradicted",
                "insufficient_evidence",
            ),
        ):
            if refs == invalid:
                with (
                    self.subTest(refs=refs, label=label),
                    self.assertRaisesRegex(ValueError, "label=None"),
                ):
                    compute_citation_metrics([record(refs, invalid, label)])

    def test_unknown_and_wrong_type_labels_rejected(self):
        for label in (
            "absent",
            "fully_covered",
            "",
            "FULLY_SUPPORTED",
            1,
            True,
            [],
            {},
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                compute_citation_metrics([record([1], label=label)])

    def test_required_records_and_fields_are_validated(self):
        for claims in (None, {}, (), "claims", [None], [[]], ["claim"], [{}]):
            with self.subTest(claims=claims), self.assertRaises(ValueError):
                compute_citation_metrics(claims)
        valid = record([1], label="fully_supported")
        for key in valid:
            incomplete = {k: v for k, v in valid.items() if k != key}
            with self.subTest(missing=key), self.assertRaises(ValueError):
                compute_citation_metrics([incomplete])

    def test_reference_lists_require_actual_integers_not_bools(self):
        for key in ("citation_numbers", "invalid_citation_numbers"):
            for value in (
                None,
                {},
                (),
                "1",
                [True],
                [False],
                [1.0],
                ["1"],
                [None],
                [[]],
            ):
                bad = record([1], label="fully_supported")
                bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    compute_citation_metrics([bad])

    def test_validity_is_caller_resolved_not_inferred_from_number(self):
        result = compute_citation_metrics([record([0, -1, 9999], [0, -1, 9999])])
        self.assertEqual(result["counts"]["num_valid_references"], 0)

    def test_extra_metadata_ignored_and_input_not_mutated(self):
        claims = [record([1, 2], label="fully_supported")]
        claims[0].update(
            {"claim": "Compound claim", "raw_response": "unused", "evidence": []}
        )
        original = copy.deepcopy(claims)
        self.assertEqual(
            compute_citation_metrics(claims)["metrics"]["cited_claim_support"], 1
        )
        self.assertEqual(claims, original)

    def test_order_does_not_change_results(self):
        claims = [
            record([1, 2], label="fully_supported"),
            record([]),
            record([3], label="contradicted"),
            record([9], [9]),
        ]
        expected = compute_citation_metrics(claims)
        for permutation in itertools.permutations(claims):
            self.assertEqual(compute_citation_metrics(list(permutation)), expected)

    def test_import_does_not_load_model_or_judge(self):
        src = str(Path(__file__).resolve().parents[1] / "src")
        script = (
            f"import sys; sys.path.insert(0, {src!r}); "
            "from rag.evaluation.citation_metrics import compute_citation_metrics; "
            "compute_citation_metrics([]); "
            "assert not any(name in sys.modules for name in "
            "('torch', 'transformers', 'langchain_core', 'rag.generate', "
            "'rag.evaluation.nugget_judge', 'rag.evaluation.citation_judge'))"
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-c", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
