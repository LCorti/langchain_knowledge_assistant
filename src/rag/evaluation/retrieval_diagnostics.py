"""Compare retrieval stages and save judged examples for named configurations.

Run from the project root with:
    uv run python -m src.rag.evaluation.retrieval_diagnostics --experiments config/retrieval_diagnostics.yaml
"""

import argparse
import copy
import csv
import gc
import json
import re
from pathlib import Path

import yaml
from datasets import Dataset

from src.rag import config as project_config
from src.rag.retriever import HybridRetriever

from .prepare_queries import evaluation_config, prepare_from_config
from .retrieval import evaluate_retrieval


def merge_config(base: dict, overrides: dict) -> dict:
    """Recursively override a config without modifying the shared defaults."""
    if not isinstance(overrides, dict):
        raise TypeError("Configuration overrides must be mappings.")
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def positive_integer(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def configurations(settings: dict, max_k: int) -> list[tuple[str, dict]]:
    defaults = {
        "bm25": project_config.get_bm25_config(),
        "faiss": project_config.get_faiss_config(),
        "retriever": project_config.get_retriever_config(),
        "reranker": project_config.get_reranker_config(),
    }
    defaults = merge_config(defaults, settings.get("defaults", {}))
    experiments = settings.get("configurations", {"baseline": {}})
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(
            "configurations must be a nonempty mapping of names to overrides."
        )
    resolved = []
    for name, overrides in experiments.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", name
        ):
            raise ValueError(
                "Configuration names must contain only letters, digits, _ or -."
            )
        effective = merge_config(defaults, overrides)
        if set(effective) != {"bm25", "faiss", "retriever", "reranker"}:
            raise ValueError(f"Unknown configuration section in {name}.")
        for section in ("bm25", "faiss"):
            effective[section]["main"]["persist_path"] = str(
                project_config.resolve_project_path(
                    effective[section]["main"]["persist_path"]
                )
            )
            positive_integer(
                effective[section]["retriever_params"]["k"], f"{name}.{section}.k"
            )
        params = effective["faiss"]["retriever_params"]
        if params["search_type"] == "mmr":
            params["fetch_k"] = max(
                positive_integer(params["fetch_k"], "fetch_k"), params["k"]
            )
        reranker = effective["reranker"]
        for key in ("rerank_k", "top_n", "reranker_batch_size"):
            positive_integer(reranker[key], key)
        if reranker["top_n"] > reranker["rerank_k"]:
            raise ValueError("Require top_n <= rerank_k.")
        resolved.append((name, effective))
    return resolved


class RankingReplay:
    """Replay rankings in dataset order, without assuming query text is unique."""

    def __init__(self, rankings):
        self.rankings = iter(rankings)

    def retrieve(self, query):
        return next(self.rankings)


def query_text(row: dict) -> str:
    return f"{row['query_title']} {row['query_text']}".strip()


def judgments(row: dict) -> dict[str, int]:
    labels = {}
    for nugget in row["nuggets"]:
        for doc_id in nugget["non_relevant_corpus_ids"]:
            labels.setdefault(str(doc_id), 0)
        for doc_id in nugget["relevant_corpus_ids"]:
            labels[str(doc_id)] = 1
    return labels


def unique_documents(documents: list, query_id: str) -> list:
    unique = {}
    for document in documents:
        doc_id = document.metadata.get("doc_id")
        if doc_id is None or str(doc_id) == "":
            raise ValueError(
                f"Retrieved document for query {query_id} is missing metadata['doc_id']."
            )
        unique.setdefault(str(doc_id), document)
    return list(unique.values())


def evaluate_stages(
    queries: Dataset,
    retriever: HybridRetriever,
    cutoffs: list[int],
    examples_per_kind: int = 5,
    text_chars: int = 2000,
) -> dict:
    """Retrieve once/query, evaluate the captured stages, and sample labeled pairs."""
    positive_integer(examples_per_kind, "examples_per_kind")
    positive_integer(text_chars, "text_chars")
    # Reuse the existing evaluator's validation before expensive retrieval/model loads.
    evaluate_retrieval(queries, RankingReplay([[] for _ in queries]), cutoffs)
    max_k = max(cutoffs)
    rows = list(queries)
    labels = [judgments(row) for row in rows]
    annotated = {doc_id for query_labels in labels for doc_id in query_labels}
    stage_rankings = {}
    counts = {}
    for index, row in enumerate(rows):
        stages = retriever.retrieve_stages(query_text(row))
        if index and set(stages) != set(stage_rankings):
            raise ValueError("Retriever returned inconsistent stages across queries.")
        for stage, documents in stages.items():
            documents = unique_documents(documents, str(row["query_id"]))
            stage_rankings.setdefault(stage, []).append(documents[:max_k])
            counts.setdefault(stage, []).append(len(documents))
        print(f"  Retrieved {index + 1}/{len(rows)} queries", flush=True)

    report = {}
    for stage, rankings in stage_rankings.items():
        metrics = evaluate_retrieval(queries, RankingReplay(rankings), cutoffs)
        per_query = []
        # Retain at most N pairs/query/category before round-robin sampling.
        sample_pools = {kind: [] for kind in ("good", "bad", "unjudged")}
        for index, (row, documents, query_labels) in enumerate(
            zip(rows, rankings, labels, strict=True)
        ):
            scores = evaluate_retrieval(
                queries.select([index]), RankingReplay([documents]), cutoffs
            )
            # Match the existing evaluator's split-wide annotation universe for Hole.
            scores.update(
                {
                    f"Hole@{k}": round(
                        sum(
                            str(doc.metadata["doc_id"]) not in annotated
                            for doc in documents[:k]
                        )
                        / k,
                        4,
                    )
                    for k in cutoffs
                }
            )
            relevant_ids = {
                doc_id for doc_id, label in query_labels.items() if label > 0
            }
            retrieved_ids = {str(doc.metadata["doc_id"]) for doc in documents}
            per_query.append(
                {
                    "query_id": str(row["query_id"]),
                    "query": query_text(row),
                    "num_returned": counts[stage][index],
                    "metrics": scores,
                    "ranking": [str(doc.metadata["doc_id"]) for doc in documents],
                    "missed_relevant_doc_ids": sorted(relevant_ids - retrieved_ids),
                }
            )
            pools = {kind: [] for kind in sample_pools}
            for rank, document in enumerate(documents, 1):
                doc_id = str(document.metadata["doc_id"])
                label = query_labels.get(doc_id)
                kind = "good" if label == 1 else "bad" if label == 0 else "unjudged"
                if len(pools[kind]) >= examples_per_kind:
                    continue
                pools[kind].append(
                    {
                        "query_id": str(row["query_id"]),
                        "query": query_text(row),
                        "doc_id": doc_id,
                        "rank": rank,
                        "relevance": label,
                        "document_text": document.page_content[:text_chars],
                        "text_truncated": len(document.page_content) > text_chars,
                        "relevant_nugget_ids": [
                            str(nugget["_id"])
                            for nugget in row["nuggets"]
                            if doc_id
                            in {str(value) for value in nugget["relevant_corpus_ids"]}
                        ],
                    }
                )
            for kind in sample_pools:
                sample_pools[kind].append(pools[kind])
        examples = {}
        for kind, pools in sample_pools.items():
            # Prefer high-ranked pairs, while spreading samples across queries.
            examples[kind] = [
                pool[offset]
                for offset in range(examples_per_kind)
                for pool in pools
                if len(pool) > offset
            ][:examples_per_kind]
        report[stage] = {
            "metrics": metrics,
            "per_query": per_query,
            "examples": examples,
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Evaluation/split YAML (default: config/evaluation.yaml)",
    )
    parser.add_argument(
        "--experiments",
        type=Path,
        help="Named configurations YAML; omit for baseline",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New output directory (must not already exist)"
    )
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N queries for a smoke test"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly recreate cached query splits",
    )
    args = parser.parse_args(argv)
    settings = evaluation_config(args.config)
    split = settings.get("split", "dev")
    if split not in ("dev", "test"):
        raise ValueError("Evaluation split must be 'dev' or 'test'.")
    cutoffs = settings.get("cutoffs", [1, 5, 10])
    if not isinstance(cutoffs, list) or not cutoffs:
        raise ValueError("cutoffs must be a nonempty list of positive integers.")
    cutoffs = sorted({positive_integer(k, "cutoff") for k in cutoffs})
    experiments = {}
    if args.experiments:
        with project_config.resolve_project_path(args.experiments).open() as stream:
            experiments = yaml.safe_load(stream)
        if not isinstance(experiments, dict):
            raise ValueError("Experiments YAML must contain a mapping.")
    configs = configurations(experiments, max(cutoffs))
    examples_per_kind = positive_integer(
        experiments.get("examples_per_kind", 5), "examples_per_kind"
    )
    text_chars = positive_integer(experiments.get("text_chars", 2000), "text_chars")
    if args.limit is not None:
        positive_integer(args.limit, "limit")
    queries = prepare_from_config(settings, overwrite=args.overwrite)[split]
    if args.limit is not None:
        queries = queries.select(range(min(args.limit, len(queries))))
    # Reject invalid data before creating outputs or loading retrievers.
    evaluate_retrieval(queries, RankingReplay([[] for _ in queries]), cutoffs)
    if args.output_dir:
        output_dir = project_config.resolve_project_path(args.output_dir)
    else:
        from datetime import datetime, timezone

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = (
            project_config.resolve_project_path(settings.get("results_dir", "results"))
            / f"retrieval_diagnostics_{split}_{run_id}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Split: {split} ({len(queries)} queries); cutoffs: {cutoffs}")
    print(f"Output: {output_dir}")
    if any(k > 20 for k in cutoffs):
        print("alpha-nDCG is skipped for cutoffs above 20.")
    summary = []
    for name, effective in configs:
        print(f"\nConfiguration: {name}", flush=True)
        for stage, depth in (
            ("BM25", effective["bm25"]["retriever_params"]["k"]),
            ("FAISS", effective["faiss"]["retriever_params"]["k"]),
            ("reranker", effective["reranker"]["top_n"]),
        ):
            if depth < max(cutoffs):
                print(
                    f"  Warning: {stage} depth {depth} is below largest cutoff {max(cutoffs)}."
                )
        retriever = HybridRetriever(
            effective["bm25"],
            effective["faiss"],
            effective["retriever"],
            **effective["reranker"],
        )
        stages = evaluate_stages(
            queries, retriever, cutoffs, examples_per_kind, text_chars
        )
        payload = {
            "configuration": name,
            "effective_config": effective,
            "evaluation_settings": settings,
            "split": split,
            "num_queries": len(queries),
            "query_ids": [str(row["query_id"]) for row in queries],
            "cutoffs": cutoffs,
            "examples_per_kind": examples_per_kind,
            "text_chars": text_chars,
            "notes": {
                "good": "Judged relevant to at least one nugget.",
                "bad": "Explicitly judged non-relevant and not relevant to any nugget.",
                "unjudged": "No judgment for this query; not necessarily non-relevant.",
                "ranking": "Deduplicated by doc_id; saved up to max(cutoffs).",
                "reranker_candidates": "Fusion prefix actually passed to the reranker.",
            },
            "stages": stages,
        }
        (output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        for stage, result in stages.items():
            summary.append({"configuration": name, "stage": stage, **result["metrics"]})
            print(
                f"  {stage}: "
                + " | ".join(
                    f"{key}={value:.4f}" for key, value in result["metrics"].items()
                )
            )
        # Do not retain multiple embedding/reranker models across configurations.
        del retriever
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with (output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(f"\nSaved reports and summary.csv to {output_dir}")


if __name__ == "__main__":
    main()
