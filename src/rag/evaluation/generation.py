"""Evaluate generated answers or saved-answer citation support with Qwen."""

import argparse
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .. import config as project_config
from .answer_similarity import JinaAnswerSimilarity
from .citation_judge import CitationJudgeError, QwenCitationJudge
from .citation_metrics import compute_citation_metrics
from .citation_parsing import CitationExtractionError, QwenCitationExtractor
from .nugget_judge import QwenNuggetJudge, _reject_constant, _unique_object

LABELS = ("fully_covered", "partially_covered", "absent", "contradicted")


def prepare_rows(queries) -> list[dict]:
    """Validate the cached FreshStack schema before running expensive models."""
    rows, seen = [], set()
    for row in queries:
        query_id = row.get("query_id")
        if (
            not isinstance(query_id, (str, int))
            or isinstance(query_id, bool)
            or str(query_id) == ""
        ):
            raise ValueError("Every query must have a nonempty query_id")
        query_id = str(query_id)
        if query_id in seen:
            raise ValueError(f"Duplicate query ID: {query_id}")
        seen.add(query_id)
        parts = [row.get("query_title", ""), row.get("query_text", "")]
        if any(not isinstance(part, str) for part in parts):
            raise ValueError(f"Query {query_id} has invalid question text")
        question = " ".join(parts).strip()
        gold_answer = row.get("answer_text")
        if not question or not isinstance(gold_answer, str) or not gold_answer.strip():
            raise ValueError(f"Query {query_id} needs question text and answer_text")
        nuggets, nugget_ids = [], set()
        if not isinstance(row.get("nuggets"), list) or not row["nuggets"]:
            raise ValueError(f"Query {query_id} has no nuggets")
        for nugget in row["nuggets"]:
            if not isinstance(nugget, dict):
                raise TypeError(f"Query {query_id} has an invalid nugget")
            nugget_id, text = nugget.get("_id"), nugget.get("text")
            if (
                not isinstance(nugget_id, (str, int))
                or isinstance(nugget_id, bool)
                or str(nugget_id) == ""
                or str(nugget_id) in nugget_ids
                or not isinstance(text, str)
                or not text.strip()
            ):
                raise ValueError(f"Query {query_id} has an invalid or duplicate nugget")
            nugget_ids.add(str(nugget_id))
            nuggets.append({"_id": str(nugget_id), "text": text})
        rows.append(
            {
                "query_id": query_id,
                "question": question,
                "gold_answer": gold_answer,
                "nuggets": nuggets,
            }
        )
    if not rows:
        raise ValueError("Cannot evaluate an empty query split")
    return rows


def _append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def generate_answers(rows: list[dict], pipeline, output_path: Path) -> list[dict]:
    """Checkpoint answers immediately so judging can be retried without regeneration."""
    answers = []
    for index, row in enumerate(rows, 1):
        print(f"Generating answer {index}/{len(rows)} ({row['query_id']})", flush=True)
        response = pipeline.run(row["question"])
        answer = response["answer"]
        if not isinstance(answer, str):
            raise TypeError(f"Query {row['query_id']} returned a non-string answer")
        record = {
            **row,
            "answer": answer,
            "sources": [
                {
                    "citation": i,
                    "doc_id": doc.metadata.get("doc_id"),
                    "text": doc.page_content,
                }
                for i, doc in enumerate(response["sources"], 1)
            ],
        }
        _append_json(output_path, record)
        answers.append(record)
    return answers


def load_answers(path: Path, rows: list[dict]) -> list[dict]:
    """Reuse this runner's JSONL answers, checking question/reference provenance."""
    saved = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            query_id = str(record["query_id"])
            if query_id in saved:
                raise ValueError(f"Duplicate saved answer for query {query_id}")
            saved[query_id] = record
    answers = []
    for row in rows:
        record = saved.get(row["query_id"])
        if record is None:
            raise ValueError(f"No saved answer for query {row['query_id']}")
        if any(
            record.get(key) != row[key]
            for key in ("question", "gold_answer", "nuggets")
        ):
            raise ValueError(
                f"Saved query/reference differs for query {row['query_id']}"
            )
        if not isinstance(record.get("answer"), str):
            raise TypeError(f"Invalid saved answer for query {row['query_id']}")
        answers.append(record)
    return answers


def evaluate_answers(
    answers: list[dict], similarity, judge, output_path: Path | None = None
) -> dict:
    if not answers:
        raise ValueError("Cannot evaluate empty answers")
    scores = similarity.score(
        [row["answer"] for row in answers], [row["gold_answer"] for row in answers]
    )
    if len(scores) != len(answers) or any(not math.isfinite(score) for score in scores):
        raise ValueError("Similarity must return one finite score per answer")
    per_query, total_counts = [], Counter()
    for index, (row, score) in enumerate(zip(answers, scores, strict=True), 1):
        print(f"Judging answer {index}/{len(answers)} ({row['query_id']})", flush=True)
        judgments = judge.judge(row["question"], row["answer"], row["nuggets"])
        if (
            not row["nuggets"]
            or [item["nugget_id"] for item in judgments]
            != [item["_id"] for item in row["nuggets"]]
            or any(item["label"] not in LABELS for item in judgments)
        ):
            raise ValueError(f"Invalid nugget judgments for query {row['query_id']}")
        counts = Counter(item["label"] for item in judgments)
        total_counts.update(counts)
        result = {
            **row,
            "metrics": {
                "jina_cosine_similarity": float(score),
                "nugget_coverage": counts["fully_covered"] / len(judgments),
                **{
                    f"nugget_{label}_rate": counts[label] / len(judgments)
                    for label in LABELS
                    if label != "fully_covered"
                },
            },
            "nugget_counts": {label: counts[label] for label in LABELS},
            "nugget_judgments": judgments,
        }
        if output_path is not None:
            _append_json(output_path, result)
        per_query.append(result)
    total = sum(total_counts.values())
    return {
        "num_queries": len(per_query),
        "num_nuggets": total,
        "metrics": {
            **{
                key: mean(row["metrics"][key] for row in per_query)
                for key in per_query[0]["metrics"]
            },
            "nugget_coverage_micro": total_counts["fully_covered"] / total,
        },
        "nugget_counts": {label: total_counts[label] for label in LABELS},
        "per_query": per_query,
    }


def load_citation_answers(path: Path, limit: int | None = None) -> list[dict]:
    """Read standalone saved answers, validating evidence before any model calls.

    No dataset, gold answer, or nuggets are required. Sources must retain the
    pipeline's one-based consecutive numbering and exact passage text. An empty
    source list is legitimate; a missing list is not reconstructible evidence.
    """
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    answers, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid saved JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"Saved answer at line {line_number} must be an object")
            query_id = row.get("query_id")
            if (
                not isinstance(query_id, (str, int))
                or isinstance(query_id, bool)
                or not str(query_id).strip()
                or str(query_id) in seen
            ):
                raise ValueError(
                    f"Invalid or duplicate saved query_id at line {line_number}"
                )
            query_id = str(query_id)
            seen.add(query_id)
            if not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"Query {query_id} needs nonempty question text")
            if not isinstance(row.get("answer"), str):
                raise TypeError(f"Query {query_id} needs a string answer")
            if not isinstance(row.get("sources"), list):
                raise TypeError(
                    f"Query {query_id} needs saved sources with exact passage text"
                )
            for expected, source in enumerate(row["sources"], 1):
                if (
                    not isinstance(source, dict)
                    or type(source.get("citation")) is not int
                    or source["citation"] != expected
                    or not isinstance(source.get("text"), str)
                    or not source["text"].strip()
                ):
                    raise ValueError(
                        f"Query {query_id} source {expected} needs consecutive one-based "
                        "citation numbering and nonempty passage text"
                    )
            answers.append({**row, "query_id": query_id})
    if not answers:
        raise ValueError("Cannot evaluate empty saved answers")
    return answers if limit is None else answers[:limit]


def evaluate_citations(
    answers: list[dict], extractor, judge, output_path: Path | None = None
) -> dict:
    """Extract actual-answer units, resolve references, then judge cited groups.

    Mixed valid/invalid groups are judged on their valid passages but receive no
    full-support credit. Uncited/all-invalid groups have no semantic verdict.
    Invalid extractions receive zero per-query metrics and remain in macro
    averaging, but add no pooled claims. Invalid judgments retain their claims
    in the denominators without support credit or a contradiction verdict.
    Generation/runtime errors still propagate.
    """
    if not answers:
        raise ValueError("Cannot evaluate empty answers")
    per_query, all_claims = [], []
    for index, row in enumerate(answers, 1):
        query_id = row["query_id"]
        print(f"Evaluating citations {index}/{len(answers)} ({query_id})", flush=True)
        extraction_error = None
        try:
            extraction = extractor.extract(row["answer"])
        except CitationExtractionError as exc:
            extraction_error = {
                "type": type(exc).__name__,
                "attempts": exc.attempts,
                "details": exc.details,
            }
            extraction = {
                "claims": [],
                "abstained": False,
                "raw_response": exc.raw_response,
            }
        sources = {source["citation"]: source["text"] for source in row["sources"]}
        claims = []
        for claim_index, unit in enumerate(extraction["claims"], 1):
            refs = unit["citation_numbers"]
            invalid = [number for number in refs if number not in sources]
            valid = list(dict.fromkeys(number for number in refs if number in sources))
            if valid:
                try:
                    judgment = judge.judge(
                        row["question"],
                        unit["claim"],
                        [
                            {"citation": number, "text": sources[number]}
                            for number in valid
                        ],
                    )
                except CitationJudgeError as exc:
                    judgment = {
                        "label": "invalid_judgment",
                        "evidence": [],
                        "reason": str(exc.__cause__ or exc),
                        "raw_response": exc.raw_response,
                        "judge_error": {
                            "type": type(exc).__name__,
                            "attempts": exc.attempts,
                        },
                    }
                skip_reason = None
            else:
                judgment = {
                    "label": None,
                    "evidence": [],
                    "reason": "",
                    "raw_response": None,
                }
                skip_reason = "invalid_references" if refs else "uncited"
            claims.append(
                {
                    **unit,
                    "claim_id": claim_index,
                    "invalid_citation_numbers": invalid,
                    **judgment,
                    "skip_reason": skip_reason,
                }
            )
        aggregate = compute_citation_metrics(claims)
        result = {
            **row,
            "abstained": extraction["abstained"],
            "empty_answer": not row["answer"].strip(),
            "extraction_raw_response": extraction["raw_response"],
            "extraction_failed": extraction_error is not None,
            "extraction_error": extraction_error,
            "citation_claims": claims,
            **aggregate,
        }
        if output_path is not None:
            _append_json(output_path, result)
        per_query.append(result)
        all_claims.extend(claims)
    pooled = compute_citation_metrics(all_claims)
    return {
        "num_queries": len(per_query),
        "num_abstained_answers": sum(row["abstained"] for row in per_query),
        "num_empty_answers": sum(row["empty_answer"] for row in per_query),
        "counts": pooled["counts"],
        "metrics": {
            **{
                key: mean(row["metrics"][key] for row in per_query)
                for key in pooled["metrics"]
            },
            **{f"{key}_micro": value for key, value in pooled["metrics"].items()},
        },
        "per_query": per_query,
        "num_extraction_failures": sum(row["extraction_failed"] for row in per_query),
        "num_judgment_failures": sum(
            claim["label"] == "invalid_judgment" for claim in all_claims
        ),
    }


def _run_citation_evaluation(args, settings: dict) -> None:
    answers_path = project_config.resolve_project_path(args.answers)
    answers = load_citation_answers(answers_path, args.limit)
    generator_config = project_config.get_generator_config(
        settings.get("generator_config", project_config.GENERATOR_CONFIG_PATH)
    )
    from ..generate import QwenGenerator

    generator = QwenGenerator(generator_config)
    citation_settings = settings.get("citations", {})
    extractor = QwenCitationExtractor(
        generator, **citation_settings.get("extractor", {})
    )
    judge = QwenCitationJudge(generator, **citation_settings.get("judge", {}))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        project_config.resolve_project_path(args.output_dir)
        if args.output_dir
        else Path(settings["results_dir"])
        / f"citations_{settings.get('split')}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    run_settings = {
        "evaluation": "citation_support",
        "answers_input": str(answers_path),
        "query_ids": [row["query_id"] for row in answers],
        "generator": generator_config,
        "extractor": {
            "max_new_tokens": extractor.max_new_tokens,
            "max_retries": extractor.max_retries,
        },
        "judge": {
            "max_new_tokens": judge.max_new_tokens,
            "max_retries": judge.max_retries,
        },
        "policies": {
            "claim_units": "LLM-extracted factual sentences or short claim groups",
            "attribution": "immediately preceding unit, never whole-paragraph backfill",
            "invalid_reference_group": "judge valid subset; no full-support credit",
            "reference_counting": "occurrences, including repeated numbers",
            "zero_denominator": 0,
            "macro_averaging": "all answers, including abstentions and empty answers",
            "invalid_extraction": "record failure; assign zero per-query metrics; include in macro averaging; exclude from pooled claim counts",
        },
    }
    (output_dir / "settings.json").write_text(
        json.dumps(run_settings, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for row in answers:
        _append_json(output_dir / "answers.jsonl", row)
    report = evaluate_citations(
        answers, extractor, judge, output_dir / "per_query.jsonl"
    )
    summary = {key: value for key, value in report.items() if key != "per_query"}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, score in summary["metrics"].items():
        print(f"{name}: {score:.4f}")
    print(f"Saved citation evaluation to {output_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Generation evaluation YAML")
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate the first N split queries or saved citation answers",
    )
    parser.add_argument(
        "--answers", type=Path, help="Reuse answers.jsonl from a previous run"
    )
    parser.add_argument(
        "--citations-only",
        action="store_true",
        help="Evaluate citation support from --answers without datasets, retrieval, or gold scoring",
    )
    parser.add_argument("--output-dir", type=Path, help="New directory for this run")
    parser.add_argument(
        "--overwrite", action="store_true", help="Recreate query splits only"
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.citations_only and not args.answers:
        parser.error("--citations-only requires --answers")
    if args.citations_only and args.overwrite:
        parser.error(
            "--overwrite recreates query splits and cannot be used with --citations-only"
        )
    settings = (
        project_config.get_generation_evaluation_config()
        if args.config is None
        else project_config.get_generation_evaluation_config(args.config)
    )
    if args.citations_only:
        _run_citation_evaluation(args, settings)
        return
    split = settings.get("split", "dev")
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")

    from .prepare_queries import prepare_from_config

    queries = prepare_from_config(settings, overwrite=args.overwrite)[split]
    if args.limit is not None:
        queries = queries.select(range(min(args.limit, len(queries))))
    rows = prepare_rows(queries)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        project_config.resolve_project_path(args.output_dir)
        if args.output_dir
        else Path(settings["results_dir"]) / f"generation_{split}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    from ..generate import QwenGenerator

    generator_config = project_config.get_generator_config(
        settings.get("generator_config", project_config.GENERATOR_CONFIG_PATH)
    )
    generator = QwenGenerator(generator_config)
    judge = QwenNuggetJudge(generator, **settings.get("judge", {}))
    similarity = JinaAnswerSimilarity(**settings.get("similarity", {}))
    run_settings = {
        "split": split,
        "query_ids": [row["query_id"] for row in rows],
        "settings": settings,
        "generator": generator_config,
        "answers_input": str(project_config.resolve_project_path(args.answers))
        if args.answers
        else None,
    }
    if args.answers:
        answers = load_answers(project_config.resolve_project_path(args.answers), rows)
        for record in answers:
            _append_json(output_dir / "answers.jsonl", record)
    else:
        from ..pipeline import RAGPipeline
        from ..retriever import HybridRetriever

        bm25_config = project_config.get_bm25_config()
        faiss_config = project_config.get_faiss_config()
        retriever_config = project_config.get_retriever_config()
        reranker_config = settings["reranker"]
        run_settings["retrieval"] = {
            "bm25": bm25_config,
            "faiss": faiss_config,
            "retriever": retriever_config,
            "reranker": reranker_config,
        }
        retriever = HybridRetriever(
            bm25_config, faiss_config, retriever_config, reranker_config
        )
        pipeline = RAGPipeline(retriever, generator, generator_config)
    (output_dir / "settings.json").write_text(
        json.dumps(run_settings, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Evaluation split: {split} ({len(rows)} queries); output: {output_dir}",
        flush=True,
    )
    if not args.answers:
        answers = generate_answers(rows, pipeline, output_dir / "answers.jsonl")
    # Similarity runs before judging; the CPU default avoids competing for GPU memory.
    report = evaluate_answers(
        answers, similarity, judge, output_dir / "per_query.jsonl"
    )
    summary = {key: value for key, value in report.items() if key != "per_query"}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for name, score in summary["metrics"].items():
        print(f"{name}: {score:.4f}")
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
