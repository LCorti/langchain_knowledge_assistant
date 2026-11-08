#!/usr/bin/env python3
"""Diagnostics for a RAG generation-evaluation run.

Reads the artifacts of a `results/generation_<split>_<timestamp>/` directory
(`per_query.jsonl`, `answers.jsonl`, optionally `summary.json`) and reports:

  1. Headline metrics, recomputed from per-query data (+ partial-credit variants
     and bootstrap confidence intervals over queries)
  2. Cosine similarity vs. nugget coverage (distribution, correlations, refusals,
     high-similarity/zero-coverage queries)
  3. Judge-validation failures (`judge_failed`) and their effect on coverage
  4. Refusal analysis (explicit "I do not know" and refusal-like phrasing)
  5. Contradiction analysis (how many come from refusal-style answers)
  6. Retrieval/citation statistics (source ecosystems, citation usage)

Usage:
    python analyze_generation_eval.py --run-dir results/generation_dev_2026...
    python analyze_generation_eval.py --per-query per_query.jsonl \
        --answers answers.jsonl --summary summary.json --json-out report.json

Requires: numpy, scipy.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

LABELS = ("fully_covered", "partially_covered", "absent", "contradicted")

# Explicit "I don't know" style abstentions.
IDK_RE = re.compile(r"i do not know|i don't know", re.I)
# Broader refusal-like phrasing (superset of IDK_RE).
REFUSAL_RE = re.compile(
    r"(i do not know|i don't know"
    r"|does not (contain|provide|include|specif)"
    r"|cannot (be )?(determine|answer|find)"
    r"|not (enough|sufficient) (information|context)"
    r"|no (specific )?(information|mention)"
    r"|context (does not|doesn't))",
    re.I,
)
CITE_RE = re.compile(r"\[(\d+)\]")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_label(judgment: dict) -> str:
    """Label the judge originally produced, even if validation later overrode it."""
    raw = judgment.get("raw_response", "")
    try:
        return json.loads(raw)["label"]
    except Exception:
        m = re.search(r'"label"\s*:\s*"(\w+)"', raw)
        return m.group(1) if m else "unparsed"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def safe_corr(x, y, method="spearman"):
    """Return NaN rather than raising or emitting warnings for degenerate data."""
    if len(x) < 2 or len(y) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    return float(np.corrcoef(x, y)[0, 1])


def json_safe(value):
    """Convert non-finite floats so the report is valid strict JSON."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- #
# Per-query scoring helpers
# --------------------------------------------------------------------------- #
def per_query_coverage(rows, credit=None, use_raw_for_failed=False, drop_failed=False):
    """Macro (per-query mean) and micro (pooled) coverage.

    credit: dict label -> weight (default: fully_covered=1, everything else 0).
    use_raw_for_failed: replace forced-absent labels by the judge's raw label.
    drop_failed: exclude judge_failed nuggets entirely.
    """
    credit = credit or {"fully_covered": 1.0}
    per_query, pooled = [], []
    for p in rows:
        scores = []
        for j in p["nugget_judgments"]:
            if j.get("judge_failed"):
                if drop_failed:
                    continue
                label = raw_label(j) if use_raw_for_failed else j["label"]
            else:
                label = j["label"]
            scores.append(credit.get(label, 0.0))
        if scores:
            per_query.append(np.mean(scores))
            pooled.extend(scores)
    return np.array(per_query), np.array(pooled)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def headline(rows, summary, out):
    section("1. MAIN METRICS (recomputed from per_query.jsonl)")
    counts = collections.Counter(
        j["label"] for p in rows for j in p["nugget_judgments"]
    )
    n_nug = sum(counts.values())
    if not n_nug:
        raise ValueError("No nugget judgments found in per_query.jsonl")
    print(
        f"queries: {len(rows)} | nuggets: {n_nug} (mean {n_nug / len(rows):.2f}/query)"
    )
    for lab in LABELS:
        print(f"  {lab:<18} {counts[lab]:>4}  ({pct(counts[lab] / n_nug)})")

    macro, micro = per_query_coverage(rows)
    lo, hi = bootstrap_ci(macro)
    print(
        f"\nmacro nugget_coverage : {macro.mean():.4f}  95% bootstrap CI over queries [{lo:.3f}, {hi:.3f}]"
    )
    print(f"micro nugget_coverage : {micro.mean():.4f}")
    print(
        f"queries with 0 coverage: {(macro == 0).sum()} ({pct((macro == 0).mean())}) | "
        f"fully covered: {(macro == 1).sum()}"
    )

    for name, credit in {
        "partial = 0.5": {"fully_covered": 1.0, "partially_covered": 0.5},
        "partial = 1.0 (upper bound)": {"fully_covered": 1.0, "partially_covered": 1.0},
    }.items():
        m, mi = per_query_coverage(rows, credit)
        print(f"coverage with {name:<28}: macro {m.mean():.4f} | micro {mi.mean():.4f}")

    if summary:
        print("\nconsistency check against summary.json:")
        for k in ("nugget_coverage", "nugget_coverage_micro", "jina_cosine_similarity"):
            if k in summary.get("metrics", {}):
                print(f"  {k}: summary={summary['metrics'][k]:.4f}")
    out["headline"] = {
        "macro": float(macro.mean()),
        "micro": float(micro.mean()),
        "macro_ci95": [lo, hi],
        "label_counts": dict(counts),
        "zero_coverage_queries": int((macro == 0).sum()),
    }
    return macro


def cosine_analysis(rows, out, cov):
    section("2. COSINE SIMILARITY vs. NUGGET COVERAGE")
    cos = np.array([p["metrics"]["jina_cosine_similarity"] for p in rows])
    pcts = [0, 10, 25, 50, 75, 90, 100]
    print(
        "cosine percentiles",
        {p: round(float(v), 3) for p, v in zip(pcts, np.percentile(cos, pcts))},
    )
    rho, pval = spearmanr(cos, cov) if len(cos) > 1 else (float("nan"), float("nan"))
    print(
        f"Spearman(cosine, coverage) = {rho:.3f} (p={pval:.3g}) | "
        f"Pearson = {safe_corr(cos, cov, 'pearson'):.3f}"
    )

    idk = np.array([bool(IDK_RE.search(p["answer"])) for p in rows])
    if idk.any() and (~idk).any():
        print(
            f"mean cosine: explicit IDK answers {cos[idk].mean():.3f} (n={idk.sum()}) "
            f"vs others {cos[~idk].mean():.3f}"
        )

    thr = 0.88
    hz = [p for p, c, v in zip(rows, cos, cov) if c > thr and v == 0]
    print(f"\nqueries with cosine > {thr} and zero coverage: {len(hz)}")
    for p in hz[:5]:
        print(f"  {p['query_id']} cos={p['metrics']['jina_cosine_similarity']:.3f}")
        print(f"    nugget: {p['nuggets'][0]['text'][:120]}")
        print(f"    answer: {p['answer'][:140].replace(chr(10), ' ')}")

    ans_w = np.array([len(p["answer"].split()) for p in rows])
    gold_w = np.array([len(p["gold_answer"].split()) for p in rows])
    n_nug = np.array([len(p["nuggets"]) for p in rows])
    print(
        f"\nanswer words  pctl(0/25/50/75/100): {np.percentile(ans_w, [0, 25, 50, 75, 100])}"
    )
    print(
        f"gold words    pctl(0/25/50/75/100): {np.percentile(gold_w, [0, 25, 50, 75, 100])}"
    )
    print(
        f"Spearman: answer_len~coverage {safe_corr(ans_w, cov):.3f} | "
        f"answer_len~cosine {safe_corr(ans_w, cos):.3f} | "
        f"gold_len~coverage {safe_corr(gold_w, cov):.3f} | "
        f"n_nuggets~coverage {safe_corr(n_nug, cov):.3f}"
    )
    out["cosine"] = {
        "spearman_cov": float(rho),
        "p": float(pval),
        "median": float(np.median(cos)),
        "high_cos_zero_cov": len(hz),
    }


def judge_failures(rows, out):
    section("3. JUDGE VALIDATION FAILURES (judge_failed -> forced absent)")
    all_j = [(p, j) for p in rows for j in p["nugget_judgments"]]
    failed = [(p, j) for p, j in all_j if j.get("judge_failed")]
    n = len(all_j)
    print(
        f"failed nuggets: {len(failed)}/{n} ({pct(len(failed) / n)}) "
        f"in {len({p['query_id'] for p, _ in failed})} queries"
    )
    if not failed:
        print("none: coverage is not affected by invalid-as-absent policy")
        out["judge_failures"] = {"n_failed": 0}
        return
    errs = collections.Counter(str(j.get("validation_error"))[:90] for _, j in failed)
    for e, c in errs.most_common():
        print(f"  {c:>3} x {e}")
    raw = collections.Counter(raw_label(j) for _, j in failed)
    print(f"judge's original labels for failed nuggets: {dict(raw)}")

    base, base_micro = per_query_coverage(rows)
    drop, drop_micro = per_query_coverage(rows, drop_failed=True)
    upper, upper_micro = per_query_coverage(rows, use_raw_for_failed=True)
    print("\ncoverage under alternative handling of failed nuggets:")
    print(
        f"  as scored (forced absent)      : macro {base.mean():.4f} | micro {base_micro.mean():.4f}"
    )
    print(
        f"  excluded from the denominator  : macro {drop.mean():.4f} | micro {drop_micro.mean():.4f} "
        f"({len(drop)}/{len(rows)} queries with remaining scored nuggets)"
    )
    print(
        f"  raw judge labels (unvalidated) : macro {upper.mean():.4f} | micro {upper_micro.mean():.4f}"
    )
    print(
        "note: raw labels were rejected because evidence was not a verbatim substring; "
        "spot-check some before trusting them."
    )
    out["judge_failures"] = {
        "n_failed": len(failed),
        "raw_labels": dict(raw),
        "macro_as_scored": float(base.mean()),
        "macro_excluded": float(drop.mean()),
        "macro_raw_unvalidated": float(upper.mean()),
        "micro_raw_unvalidated": float(upper_micro.mean()),
    }


def refusals(rows, out, cov):
    section("4. REFUSAL ANALYSIS")
    idk = np.array([bool(IDK_RE.search(p["answer"])) for p in rows])
    ref = np.array([bool(REFUSAL_RE.search(p["answer"])) for p in rows])
    n = len(rows)
    print(
        f"explicit 'I do not know': {idk.sum()}/{n} ({pct(idk.mean())}) | "
        f"mean coverage {cov[idk].mean() if idk.any() else float('nan'):.3f} | "
        f"with coverage > 0: {int((cov[idk] > 0).sum())}"
    )
    print(
        f"refusal-like phrasing   : {ref.sum()}/{n} ({pct(ref.mean())}) | "
        f"mean coverage {cov[ref].mean() if ref.any() else float('nan'):.3f}"
    )
    print(
        f"non-refusal answers     : {(~ref).sum()} | mean coverage "
        f"{cov[~ref].mean() if (~ref).any() else float('nan'):.3f}"
    )
    print(
        "caveat: refusal detection is regex based; inspect matches if you rely on exact counts."
    )
    out["refusals"] = {
        "explicit_idk": int(idk.sum()),
        "refusal_like": int(ref.sum()),
        "coverage_non_refusal": float(cov[~ref].mean()) if (~ref).any() else None,
    }


def contradictions(rows, out):
    section("5. CONTRADICTIONS")
    cs = [
        (p, j)
        for p in rows
        for j in p["nugget_judgments"]
        if j["label"] == "contradicted"
    ]
    from_ref = sum(bool(REFUSAL_RE.search(p["answer"])) for p, _ in cs)
    print(
        f"contradicted nuggets: {len(cs)} in {len({p['query_id'] for p, _ in cs})} queries"
    )
    print(
        f"  from answers with refusal-like phrasing: {from_ref} "
        "(candidates for relabeling as 'absent')"
    )
    for p, j in cs[:6]:
        print(f"\n  [{p['query_id']}] nugget: {j['text'][:130]}")
        print(f"    reason: {j['reason'][:230]}")
    out["contradictions"] = {"n": len(cs), "from_refusal_like_answers": from_ref}


def retrieval_stats(rows, answers, out):
    section("6. RETRIEVAL / CITATION STATISTICS")
    if not answers:
        print("answers.jsonl not provided; skipping")
        return
    prefixes = collections.Counter()
    n_src = 0
    for a in answers:
        for s in a.get("sources", []):
            prefixes[s.get("doc_id", "unknown").split("/")[0]] += 1
            n_src += 1
    print(f"supplied passages: {n_src} over {len(answers)} queries")
    for k, v in prefixes.most_common(8):
        print(f"  {k:<28} {v:>5} ({pct(v / n_src) if n_src else 'n/a'})")
    # Validate citation indices when answer records can be matched by query_id.
    source_counts = {
        a["query_id"]: len(a.get("sources", [])) for a in answers if "query_id" in a
    }
    invalid_citations = 0
    checked_citations = 0
    for p in rows:
        if p.get("query_id") not in source_counts:
            continue
        n_passages = source_counts[p["query_id"]]
        citations = [int(c) for c in CITE_RE.findall(p["answer"])]
        checked_citations += len(citations)
        invalid_citations += sum(c < 1 or c > n_passages for c in citations)
    if checked_citations:
        print(
            f"invalid citation indices: {invalid_citations}/{checked_citations} "
            "(among queries matched to answers.jsonl)"
        )
    distinct = [len(set(CITE_RE.findall(p["answer"]))) for p in rows]
    print(
        f"\ndistinct citations per answer: mean {np.mean(distinct):.2f} | "
        f"answers with none: {sum(d == 0 for d in distinct)}"
    )
    pos = collections.Counter(
        int(c) for p in rows for c in CITE_RE.findall(p["answer"])
    )
    print("citation counts by passage rank:", sorted(pos.items()))
    out["retrieval"] = {
        "sources_by_prefix": dict(prefixes.most_common(8)),
        "mean_distinct_citations": float(np.mean(distinct)),
        "invalid_citations": invalid_citations,
        "checked_citations": checked_citations,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-dir", type=Path, help="results/generation_<split>_<timestamp>/"
    )
    ap.add_argument("--per-query", type=Path)
    ap.add_argument("--answers", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--json-out", type=Path, help="write machine-readable results here")
    args = ap.parse_args()

    if args.run_dir:
        args.per_query = args.per_query or args.run_dir / "per_query.jsonl"
        args.answers = args.answers or args.run_dir / "answers.jsonl"
        args.summary = args.summary or args.run_dir / "summary.json"
    if not args.per_query or not args.per_query.exists():
        ap.error("per_query.jsonl is required (use --run-dir or --per-query)")

    rows = read_jsonl(args.per_query)
    if not rows:
        ap.error("per_query.jsonl contains no query records")
    answers = (
        read_jsonl(args.answers) if args.answers and args.answers.exists() else None
    )
    summary = (
        json.loads(args.summary.read_text())
        if args.summary and args.summary.exists()
        else None
    )

    out: dict = {}
    cov = headline(rows, summary, out)
    cosine_analysis(rows, out, cov)
    judge_failures(rows, out)
    refusals(rows, out, cov)
    contradictions(rows, out)
    retrieval_stats(rows, answers, out)

    if args.json_out:
        args.json_out.write_text(json.dumps(json_safe(out), indent=2, allow_nan=False))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
