import argparse
import json
from pathlib import Path

from datasets import Dataset

from src.rag import config as project_config
from src.rag.retriever import HybridRetriever

from .metrics import alpha_ndcg, coverage, hole, ndcg, recall
from .prepare_queries import evaluation_config, prepare_from_config


def evaluate_retrieval(
    queries: Dataset,
    retriever: HybridRetriever,
    cutoffs: list[int],
) -> dict[str, float]:
    if len(queries) == 0:
        raise ValueError("Cannot evaluate an empty query split.")
    if (
        not isinstance(cutoffs, list)
        or not cutoffs
        or any(type(k) is not int or k <= 0 for k in cutoffs)
    ):
        raise ValueError(
            "Evaluation cutoffs must be a nonempty list of positive integers."
        )

    cutoffs = sorted(set(cutoffs))
    max_k = max(cutoffs)

    qrels_query: dict[str, dict[str, int]] = {}
    qrels_nuggets: dict[str, dict[str, int]] = {}
    query_to_nuggets: dict[str, list[str]] = {}
    results: dict[str, dict[str, float]] = {}

    # Build judgments separately from retrieval so malformed data fails early.
    for row in queries:
        query_id = str(row["query_id"])
        if query_id in qrels_query:
            raise ValueError(f"Duplicate query ID: {query_id}")

        nuggets = row["nuggets"]
        if not nuggets:
            raise ValueError(f"Query {query_id} has no nuggets.")

        query_qrels: dict[str, int] = {}
        query_to_nuggets[query_id] = []

        for nugget in nuggets:
            # Namespace nugget IDs to avoid collisions across queries.
            nugget_id = json.dumps([query_id, str(nugget["_id"])])
            if nugget_id in qrels_nuggets:
                raise ValueError(f"Duplicate nugget ID in query {query_id}")

            nugget_qrels = {
                str(doc_id): 0 for doc_id in nugget["non_relevant_corpus_ids"]
            }
            nugget_qrels.update(
                {str(doc_id): 1 for doc_id in nugget["relevant_corpus_ids"]}
            )
            qrels_nuggets[nugget_id] = nugget_qrels
            query_to_nuggets[query_id].append(nugget_id)

            # A document is query-relevant if it supports any nugget.
            # A negative judgment for one nugget must not overwrite a
            # positive judgment for another.
            for doc_id, relevance in nugget_qrels.items():
                query_qrels[doc_id] = max(query_qrels.get(doc_id, 0), relevance)

        if not any(query_qrels.values()):
            raise ValueError(f"Query {query_id} has no relevant documents.")
        qrels_query[query_id] = query_qrels

    for row in queries:
        query_id = str(row["query_id"])
        query = f"{row['query_title']} {row['query_text']}".strip()
        if not query:
            raise ValueError(f"Query {query_id} has no text.")

        documents = retriever.retrieve(query)
        doc_scores: dict[str, float] = {}

        for rank, document in enumerate(documents):
            doc_id = document.metadata.get("doc_id")
            if doc_id is None or str(doc_id) == "":
                raise ValueError(
                    f"Retrieved document for query {query_id} "
                    "is missing metadata['doc_id']."
                )

            # The hybrid retriever exposes ranking, not numeric scores.
            # Strictly decreasing scores preserve that ranking.
            doc_scores.setdefault(str(doc_id), float(len(documents) - rank))
            if len(doc_scores) == max_k:
                break

        # Keep empty runs: these queries must still count in macro averages.
        results[query_id] = doc_scores

    scores = {}
    scores.update(recall(qrels_query, results, cutoffs))
    scores.update(ndcg(qrels_query, results, cutoffs))
    scores.update(coverage(qrels_nuggets, query_to_nuggets, results, cutoffs))

    alpha_cutoffs = [k for k in cutoffs if k <= 20]
    if alpha_cutoffs:
        # Avoid passing empty runs to the current alpha_ndcg wrapper.
        # Its denominator remains all queries in query_to_nuggets.
        nonempty_results = {query_id: run for query_id, run in results.items() if run}
        scores.update(
            alpha_ndcg(
                qrels_nuggets,
                query_to_nuggets,
                nonempty_results,
                alpha_cutoffs,
            )
        )

    hole_per_query = hole(qrels_query, results, cutoffs)
    scores.update(
        {
            f"Hole@{k}": round(
                sum(per_query[f"Hole@{k}"] for per_query in hole_per_query.values())
                / len(queries),
                4,
            )
            for k in cutoffs
        }
    )

    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional evaluation YAML file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly recreate cached splits"
    )
    args = parser.parse_args(argv)
    settings = evaluation_config(args.config)
    split = settings.get("split", "test")
    if split != "test":
        raise ValueError(
            "Retrieval evaluation requires the 'test' split; "
            "use retrieval_diagnostics for dev tuning."
        )
    cutoffs = settings.get("cutoffs", [1, 5, 10])
    if (
        not isinstance(cutoffs, list)
        or not cutoffs
        or any(type(k) is not int or k <= 0 for k in cutoffs)
    ):
        raise ValueError(
            "Evaluation cutoffs must be a nonempty list of positive integers."
        )
    splits = prepare_from_config(settings, overwrite=args.overwrite)
    results_dir = project_config.resolve_project_path(
        settings.get("results_dir", "results")
    )
    print(f"Evaluation split: {split} ({len(splits[split])} queries)")
    print(f"Cutoffs: {cutoffs}")
    print(f"Results directory: {results_dir}")
    print("Preparing retrievers.")
    cutoffs = sorted(set(cutoffs))
    max_k = max(cutoffs)
    bm25_config = project_config.get_bm25_config()
    faiss_config = project_config.get_faiss_config()
    retriever_config = settings["retriever"]
    reranker_config = {**settings["reranker"], "top_n": max_k}

    # Relevant for FAISS search MMR -- TBD
    faiss_params = faiss_config["retriever_params"]
    if faiss_params["search_type"] == "mmr":
        faiss_params["fetch_k"] = max(faiss_params["fetch_k"], max_k)
    retriever = HybridRetriever(
        bm25_config,
        faiss_config,
        retriever_config,
        reranker_config,
    )
    #  alpha-nDCG seems to ignore anything beyond k = 20...
    unsupported = [k for k in cutoffs if k > 20]
    if unsupported:
        print(f"Skipping alpha-nDCG for cutoffs above 20: {unsupported}")

    print("\nEvaluating retrieval...")
    metrics = evaluate_retrieval(splits[split], retriever, cutoffs)
    for name, score in metrics.items():
        print(f"{name}: {score:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"retrieval_{split}.json"
    output_path.write_text(
        json.dumps(
            {
                "split": split,
                "num_queries": len(splits[split]),
                "cutoffs": cutoffs,
                "settings": settings,
                "retriever": {
                    "bm25": bm25_config,
                    "faiss": faiss_config,
                    "retriever": retriever_config,
                    "reranker": reranker_config,
                },
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
