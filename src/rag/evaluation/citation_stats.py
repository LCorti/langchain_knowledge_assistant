#!/usr/bin/env python3
"""
Re-derive and audit the citation-support eval from per_query.jsonl + summary.json.

Usage:
    python analyze_citation_eval.py --run-dir ./results/citations_<split>_<timestamp>/ \
        [--review-csv contradicted.csv] [--out report.json]

Sections:
  1. Reconcile recomputed counts/metrics against summary.json (macro and micro).
  2. Partition answers: abstained / extraction-failed / scored.
  3. Claim-level breakdown (uncited, judge labels, invalid judgments, invalid refs).
  4. Extraction-failure categories and a length-bias check.
  5. Invalid-judgment categories + sensitivity analysis using the judge's raw
     (quote-unvalidated) label.
  6. Support rates under different denominators, with Wilson 95% intervals.
  7. Contradicted claims dumped for manual review.

Only the standard library is required.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics as st
from collections import Counter


# ----------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def frac(k, n):
    return k / n if n else float("nan")


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {frac(k, n):.3f}  (95% Wilson CI {lo:.3f}-{hi:.3f})"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def header(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def categorize_extraction_error(err):
    d = (err or {}).get("details", "") if isinstance(err, dict) else str(err)
    if "not a literal substring" in d:
        return "answer_span not a literal substring of the answer"
    if "violates original order" in d:
        return "claims violate original order"
    if "overlap" in d:
        return "overlapping spans"
    if "valid JSON" in d:
        return "invalid JSON output"
    if "citation_numbers" in d and "actual citation markers" in d:
        return "citation_numbers do not match markers in span"
    if "complete actual citation markers" in d:
        return "span omits trailing citation markers"
    return "other: " + d[:60]


def categorize_judge_error(reason):
    r = reason or ""
    if "literal substring" in r:
        return "evidence quote not a literal substring of passage"
    if "valid JSON" in r:
        return "invalid JSON output"
    if "citation must be an integer" in r:
        return "evidence citation id invalid"
    if "exactly the keys" in r or "exactly citation and quote" in r:
        return "schema violation"
    return "other: " + r[:60]


LABELS = (
    "fully_supported",
    "partially_supported",
    "contradicted",
    "insufficient_evidence",
)


def raw_label(raw):
    """Best-effort extraction of the judge's label from an unvalidated raw response."""
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        lab = json.loads(s).get("label")
    except Exception:
        m = re.search(r'"label"\s*:\s*"([a-z_]+)"', raw)
        lab = m.group(1) if m else None
    return lab if lab in LABELS else None


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="results directory containing per_query.jsonl and summary.json",
    )
    ap.add_argument("--review-csv", default=None, help="write contradicted claims here")
    ap.add_argument("--out", default=None, help="write machine-readable report here")
    a = ap.parse_args()
    a.per_query = os.path.join(a.run_dir, "per_query.jsonl")
    a.summary = os.path.join(a.run_dir, "summary.json")

    with open(a.per_query) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(a.summary) as f:
        summ = json.load(f)
    report = {}

    # ------------------------------------------------ 1. reconcile with summary
    header("1. Reconciliation with summary.json")
    keys = [
        "num_claims",
        "num_cited_claims",
        "num_references",
        "num_valid_references",
        "num_fully_supported_claims",
        "num_contradicted_claims",
    ]
    tot = {k: sum(r["counts"][k] for r in rows) for k in keys}
    for k in keys:
        ok = "OK " if tot[k] == summ["counts"][k] else "MISMATCH"
        print(f"  [{ok}] {k:30s} recomputed={tot[k]:5d} summary={summ['counts'][k]:5d}")

    micro = {
        "citation_validity_micro": frac(
            tot["num_valid_references"], tot["num_references"]
        ),
        "cited_claim_support_micro": frac(
            tot["num_fully_supported_claims"], tot["num_cited_claims"]
        ),
        "supported_citation_coverage_micro": frac(
            tot["num_fully_supported_claims"], tot["num_claims"]
        ),
        "contradiction_rate_micro": frac(
            tot["num_contradicted_claims"], tot["num_claims"]
        ),
    }
    macro = {
        m: mean(r["metrics"][m] for r in rows)
        for m in (
            "citation_validity",
            "cited_claim_support",
            "supported_citation_coverage",
            "contradiction_rate",
        )
    }
    for k, v in {**macro, **micro}.items():
        s = summ["metrics"][k]
        print(
            f"  [{'OK ' if abs(v - s) < 1e-9 else 'MISMATCH'}] {k:36s} recomputed={v:.4f} summary={s:.4f}"
        )

    # ----------------------------------------------- 2. partition of answers
    header("2. Answer partition")
    ext_fail = [r for r in rows if r["extraction_failed"]]
    abst = [r for r in rows if r["abstained"]]
    empty = [r for r in rows if r["empty_answer"]]
    scored = [
        r
        for r in rows
        if not r["extraction_failed"] and not r["abstained"] and not r["empty_answer"]
    ]
    overlap = [r for r in rows if r["extraction_failed"] and r["abstained"]]
    print(f"  total answers          : {len(rows)}")
    print(
        f"  extraction failed      : {len(ext_fail)}  ({len(ext_fail) / len(rows):.1%})"
    )
    print(f"  abstained              : {len(abst)}  ({len(abst) / len(rows):.1%})")
    print(f"  empty                  : {len(empty)}")
    print(f"  extraction&abstained   : {len(overlap)} (overlap)")
    print(f"  scored (usable)        : {len(scored)}  ({len(scored) / len(rows):.1%})")
    zero_rows = [r for r in rows if r["counts"]["num_claims"] == 0]
    abst_with_claims = sum(1 for r in abst if r["counts"]["num_claims"] > 0)
    scored_zero = sum(1 for r in scored if r["counts"]["num_claims"] == 0)
    print(
        f"  answers with 0 claims  : {len(zero_rows)} "
        f"(abstained answers that still have claims: {abst_with_claims}; "
        f"scored answers with 0 claims: {scored_zero})"
    )

    print(
        "\n  Macro metrics: all answers vs scored-only (extraction-failed/abstained/empty dropped)"
    )
    for m in macro:
        print(
            f"    {m:30s} all={macro[m]:.4f}   scored-only="
            f"{mean(r['metrics'][m] for r in scored):.4f}"
        )
    report["partition"] = dict(
        total=len(rows),
        extraction_failed=len(ext_fail),
        abstained=len(abst),
        scored=len(scored),
    )

    # --------------------------------------------- 3. claim-level breakdown
    header("3. Claim-level breakdown (all extracted factual units)")
    claims = [c for r in rows for c in r["citation_claims"]]
    n_claims = len(claims)
    skip = Counter(c["skip_reason"] for c in claims)
    lab = Counter(c["label"] for c in claims)
    print(f"  factual units: {n_claims}")
    print(f"  skip reasons : {dict(skip)}")
    print(f"  labels       : {dict(lab)}")
    uncited = skip.get("uncited", 0)
    cited = n_claims - uncited
    print(f"\n  uncited units          : {fmt(uncited, n_claims)}")
    print(f"  cited units            : {cited}")
    for l in (
        "fully_supported",
        "partially_supported",
        "insufficient_evidence",
        "contradicted",
        "invalid_judgment",
    ):
        print(f"    {l:24s}: {fmt(lab.get(l, 0), cited)} of cited")
    n_inv_ref = skip.get("invalid_references", 0)
    print(f"    {'(all refs invalid)':24s}: {n_inv_ref}")
    assert (
        cited
        == sum(
            lab.get(l, 0)
            for l in (
                "fully_supported",
                "partially_supported",
                "insufficient_evidence",
                "contradicted",
                "invalid_judgment",
            )
        )
        + n_inv_ref
    ), "partition leak"

    # per-answer citation rate among scored answers
    rates = [
        frac(r["counts"]["num_cited_claims"], r["counts"]["num_claims"])
        for r in scored
        if r["counts"]["num_claims"]
    ]
    print(
        f"\n  per scored answer, fraction of units cited: "
        f"mean={mean(rates):.3f} median={st.median(rates):.3f} "
        f"| answers with zero citations: {sum(1 for x in rates if x == 0)}"
    )
    report["claims"] = dict(
        total=n_claims, uncited=uncited, cited=cited, labels=dict(lab)
    )

    # ------------------------------------------ 4. extraction failure analysis
    header("4. Extraction failures")
    ec = Counter(categorize_extraction_error(r["extraction_error"]) for r in ext_fail)
    for k, v in ec.most_common():
        print(f"  {v:3d}  {k}")
    print(
        f"  attempts per failure: {dict(Counter(r['extraction_error'].get('attempts') for r in ext_fail))}"
    )
    L = lambda rs: [len(r["answer"]) for r in rs]
    print(
        f"\n  answer length (chars), median: failed={st.median(L(ext_fail)):.0f} "
        f"scored={st.median(L(scored)):.0f}"
    )
    print(
        f"  answer length (chars), mean  : failed={mean(L(ext_fail)):.0f} "
        f"scored={mean(L(scored)):.0f}"
    )
    code_fail = mean("```" in r["answer"] for r in ext_fail)
    code_ok = mean("```" in r["answer"] for r in scored)
    print(
        f"  share of answers containing a code fence: failed={code_fail:.2f} scored={code_ok:.2f}"
    )
    # fraction of failed answers by length quartile of all answers
    qs = st.quantiles(L(rows), n=4)
    print("  failure rate by answer-length quartile:")
    bins = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e12)]
    for i, (lo, hi) in enumerate(bins, 1):
        sub = [r for r in rows if lo < len(r["answer"]) <= hi]
        print(
            f"    Q{i} (<= {hi if hi < 1e11 else 'inf'}): "
            f"{sum(r['extraction_failed'] for r in sub)}/{len(sub)}"
        )
    report["extraction_failures"] = dict(ec)

    # ------------------------------------------ 5. invalid judgments
    header("5. Judgment failures and sensitivity")
    inv = [c for c in claims if c["label"] == "invalid_judgment"]
    print(
        f"  invalid judgments (units): {len(inv)}  "
        f"in {sum(any(c['label'] == 'invalid_judgment' for c in r['citation_claims']) for r in rows)} answers"
    )
    jc = Counter(categorize_judge_error(c["reason"]) for c in inv)
    for k, v in jc.most_common():
        print(f"  {v:3d}  {k}")

    raw = Counter(raw_label(c.get("raw_response")) for c in inv)
    print(f"\n  Judge's raw (quote-unvalidated) label on those {len(inv)} units:")
    for k, v in raw.most_common():
        print(f"    {str(k):24s}: {v}")

    full = lab.get("fully_supported", 0)
    judged = cited - len(inv) - n_inv_ref  # units with a valid verdict
    print("\n  Support rate under different assumptions (cited units):")
    print(f"    (a) protocol as specified (failures = 0)      : {fmt(full, cited)}")
    print(f"    (b) among validly judged units only            : {fmt(full, judged)}")
    r_full = raw.get("fully_supported", 0)
    r_part = raw.get("partially_supported", 0)
    print(
        f"    (c) + raw 'fully_supported' on failures        : {fmt(full + r_full, cited)}"
        "   <- UNVALIDATED upper-end, quotes did not verify"
    )
    print(
        f"    (d) (c) restricted to non-failed + raw-full    : "
        f"{fmt(full + r_full, judged + len(inv))}"
    )
    print("  Verdict distribution among validly judged units:")
    for l in LABELS:
        print(f"    {l:24s}: {fmt(lab.get(l, 0), judged)}")
    print(f"  Overall coverage (supported / all units): (a) {fmt(full, n_claims)}")
    print(
        f"                                            (c) {fmt(full + r_full, n_claims)}"
    )
    report["support"] = dict(
        strict=frac(full, cited),
        among_judged=frac(full, judged),
        with_raw_full_on_failures=frac(full + r_full, cited),
        raw_labels_on_failures=dict((str(k), v) for k, v in raw.items()),
    )

    # ------------------------------------------ 6. citation behavior
    header("6. Citation behavior (valid refs)")
    ref_total = tot["num_references"]
    print(
        f"  reference validity (micro): {fmt(tot['num_valid_references'], ref_total)}"
    )
    bad = [
        (
            r["query_id"],
            c["claim_id"],
            c["citation_numbers"],
            c["invalid_citation_numbers"],
        )
        for r in rows
        for c in r["citation_claims"]
        if c["invalid_citation_numbers"]
    ]
    print(f"  units containing invalid refs: {bad}")
    ncites = Counter(
        len(set(c["citation_numbers"])) for c in claims if c["citation_numbers"]
    )
    print(f"  distinct sources per cited unit: {dict(sorted(ncites.items()))}")
    by_n = {}
    for c in claims:
        n = len(set(c["citation_numbers"]))
        if n and c["label"] in LABELS + ("invalid_judgment",):
            by_n.setdefault(min(n, 3), Counter())[c["label"]] += 1
    print("  label by #sources cited (3 = 3+):")
    for n, cnt in sorted(by_n.items()):
        tot_n = sum(cnt.values())
        print(
            f"    n={n}: total={tot_n:3d} full={cnt['fully_supported']:3d} "
            f"insuff={cnt['insufficient_evidence']:3d} inv={cnt['invalid_judgment']:3d}"
        )

    # ------------------------------------------ 7. contradicted claims
    header("7. Contradicted claims (for manual review)")
    rev = []
    for r in rows:
        for c in r["citation_claims"]:
            if c["label"] == "contradicted":
                rev.append(
                    dict(
                        query_id=r["query_id"],
                        claim_id=c["claim_id"],
                        claim=c["claim"],
                        citations=c["citation_numbers"],
                        quotes=" || ".join(e["quote"] for e in c["evidence"]),
                        judge_reason=c["reason"],
                        human_verdict="",
                    )
                )
    for i, x in enumerate(rev, 1):
        print(
            f"\n  [{i}] query={x['query_id']} claim_id={x['claim_id']} cites={x['citations']}"
        )
        print(f"      CLAIM : {x['claim'][:300]}")
        print(f"      QUOTE : {x['quotes'][:300]}")
        print(f"      REASON: {x['judge_reason'][:400]}")
    if a.review_csv:
        with open(a.review_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rev[0].keys()) if rev else ["none"])
            w.writeheader()
            w.writerows(rev)
        print(
            f"\n  wrote {len(rev)} rows to {a.review_csv} (fill human_verdict column)"
        )

    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote report to {a.out}")


if __name__ == "__main__":
    main()
