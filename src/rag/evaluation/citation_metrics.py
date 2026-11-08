"""Deterministic citation metrics over evaluated claim groups; no model calls."""

from __future__ import annotations

from collections import Counter

_LABELS = frozenset(
    {
        "fully_supported",
        "partially_supported",
        "contradicted",
        "insufficient_evidence",
        "invalid_judgment",
    }
)


def compute_citation_metrics(claims: list[dict]) -> dict:
    """Return separate ``counts`` and ``metrics`` mappings.

    Each record requires citation_numbers, invalid_citation_numbers (integer
    occurrence lists, not sets), and label. Invalid references must be a multiset
    subset of references. Reference validity is resolved by the caller, not by
    this function. Booleans are not citation integers. Other record keys are ignored.
    A label is required for every group containing valid references; groups with
    no valid references must have label=None because their judge was skipped.
    invalid_judgment records output-validation failure, retaining the claim and
    references in denominators without support credit or a contradiction verdict.

    num_fully_supported_claims counts credited groups: fully_supported AND no
    invalid references. Mixed valid/invalid groups receive no support credit,
    but can still count in num_contradicted_claims. Reference counts include
    duplicates. Cited groups include all-invalid groups. Uncited groups remain
    in num_claims (the coverage and contradiction denominator). Empty claims,
    including pure abstentions, yield all zeros; every zero denominator yields 0.
    """
    if not isinstance(claims, list):
        raise TypeError("claims must be a list of evaluated claim dicts")
    counts = {
        "num_claims": len(claims),
        "num_cited_claims": 0,
        "num_references": 0,
        "num_valid_references": 0,
        "num_fully_supported_claims": 0,
        "num_contradicted_claims": 0,
    }
    for claim in claims:
        if not isinstance(claim, dict) or not {
            "citation_numbers",
            "invalid_citation_numbers",
            "label",
        }.issubset(claim):
            raise ValueError(
                "each claim requires citation_numbers, invalid_citation_numbers, label"
            )
        refs = claim["citation_numbers"]
        invalid = claim["invalid_citation_numbers"]
        for name, numbers in (
            ("citation_numbers", refs),
            ("invalid_citation_numbers", invalid),
        ):
            if not isinstance(numbers, list) or any(
                type(n) is not int for n in numbers
            ):
                raise ValueError(f"{name} must be a list of integers")
        if Counter(invalid) - Counter(refs):
            raise ValueError(
                "invalid_citation_numbers must be a multiset subset of citation_numbers"
            )
        label = claim["label"]
        if label is not None and (not isinstance(label, str) or label not in _LABELS):
            raise ValueError(
                "label must be fully_supported, partially_supported, contradicted, "
                "insufficient_evidence, invalid_judgment, or None for a skipped group"
            )
        num_valid = len(refs) - len(invalid)
        if num_valid and label is None:
            raise ValueError(
                "a verdict is required for every group with valid references"
            )
        if not num_valid and label is not None:
            raise ValueError("groups with no valid references must have label=None")
        counts["num_cited_claims"] += bool(refs)
        counts["num_references"] += len(refs)
        counts["num_valid_references"] += num_valid
        counts["num_fully_supported_claims"] += (
            label == "fully_supported" and not invalid
        )
        counts["num_contradicted_claims"] += label == "contradicted"

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    supported = counts["num_fully_supported_claims"]
    return {
        "counts": counts,
        "metrics": {
            "citation_validity": ratio(
                counts["num_valid_references"], counts["num_references"]
            ),
            "cited_claim_support": ratio(supported, counts["num_cited_claims"]),
            "supported_citation_coverage": ratio(supported, counts["num_claims"]),
            "contradiction_rate": ratio(
                counts["num_contradicted_claims"], counts["num_claims"]
            ),
        },
    }
