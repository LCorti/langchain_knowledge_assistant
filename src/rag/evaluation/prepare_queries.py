"""Prepare reproducible, provenance-checked query splits for evaluation."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from src.rag import config as project_config

MANIFEST_NAME = "manifest.json"


def _query_ids(dataset: Dataset) -> list:
    column = next(
        (name for name in ("_id", "query_id", "id") if name in dataset.column_names),
        None,
    )
    if column is None:
        raise ValueError("Queries must have an _id, query_id, or id column.")
    ids = list(dataset[column])
    if any(type(value) not in (str, int) or value == "" for value in ids):
        raise ValueError("Query IDs must be nonempty strings or integers.")
    if len(set(ids)) != len(ids):
        raise ValueError("Query IDs must be unique within each split.")
    return ids


def _split_ids(splits: DatasetDict) -> dict:
    if set(splits) != {"dev", "test"}:
        raise ValueError("Expected exactly dev/test splits.")
    if any(not isinstance(part, Dataset) or len(part) == 0 for part in splits.values()):
        raise ValueError("Both splits must be nonempty Dataset instances.")
    ids = {name: _query_ids(splits[name]) for name in ("dev", "test")}
    if set(ids["dev"]) & set(ids["test"]):
        raise ValueError("Dev/test query IDs must be disjoint.")
    return ids


def _check_cache_ownership(cache_path: Path) -> None:
    """Refuse to replace reserved entries unless they form a saved DatasetDict."""
    owned_names = ("dev", "test", "dataset_dict.json", MANIFEST_NAME)
    if not any(
        (cache_path / name).exists() or (cache_path / name).is_symlink()
        for name in owned_names
    ):
        return
    try:
        for name in owned_names:
            if (cache_path / name).is_symlink():
                raise ValueError(f"reserved cache entry is a symlink: {name}")
        metadata = [cache_path / "dataset_dict.json"]
        for split in ("dev", "test"):
            metadata.extend(
                cache_path / split / name
                for name in ("dataset_info.json", "state.json")
            )
        for path in metadata:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing regular Dataset metadata file: {path}")
            if not isinstance(json.loads(path.read_text()), dict):
                raise TypeError(f"invalid Dataset metadata mapping: {path}")
        saved = json.loads((cache_path / "dataset_dict.json").read_text())
        if sorted(saved.get("splits", [])) != ["dev", "test"]:
            raise ValueError("DatasetDict metadata must declare exactly dev/test")
        # Validate the saved metadata and data locally, not by loading the source.
        # Conservatively reject partial caches rather than infer ownership from
        # directory names or a manifest alone.
        existing = load_from_disk(str(cache_path))
        if (
            not isinstance(existing, DatasetDict)
            or set(existing) != {"dev", "test"}
            or any(not isinstance(part, Dataset) for part in existing.values())
        ):
            raise ValueError("expected a saved dev/test DatasetDict")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(
            f"Refusing to overwrite unrecognized evaluation cache at {cache_path}: "
            f"{exc}. Existing entries were not removed. Choose another cache_dir "
            "or inspect and move the conflicting entries manually."
        ) from exc


def load_or_create_retrieval_splits(
    cache_dir: str | Path = "data/langchain_queries",
    dev_size: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
    *,
    dataset_path: str = "freshstack/queries-oct-2024",
    dataset_name: str | None = "langchain",
    dataset_split: str = "test",
    revision: str | None = None,
) -> DatasetDict:
    """Reuse matching cached splits, or create seeded dev/test partitions.

    A cache without a matching manifest is never silently reused. Explicit
    overwrite requires an identifiable saved dev/test DatasetDict (legacy caches
    without a manifest are accepted), and replaces only those datasets and their
    metadata. Unrelated files in cache_dir are preserved. Ambiguous entries are
    rejected before loading the source. No source/network lookup is needed for reuse.
    """
    if (
        isinstance(dev_size, bool)
        or not isinstance(dev_size, (int, float))
        or not 0 < dev_size < 1
    ):
        raise ValueError("dev_size must be a fraction between 0 and 1.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("dataset_path must be a nonempty string.")
    if not isinstance(dataset_split, str) or not dataset_split:
        raise ValueError("dataset_split must be a nonempty string.")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("revision must be a nonempty string or None.")

    cache_path = project_config.resolve_project_path(cache_dir)
    requested = {
        "seed": seed,
        "dev_size": float(dev_size),
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "revision": revision,
    }
    if cache_path.exists() and not overwrite:
        try:
            manifest = json.loads((cache_path / MANIFEST_NAME).read_text())
            if not isinstance(manifest, dict) or manifest.get("version") != 1:
                raise ValueError("missing or unsupported manifest version")
            if manifest.get("requested") != requested:
                raise ValueError("requested provenance differs from the manifest")
            splits = load_from_disk(str(cache_path))
            ids = _split_ids(splits)
            if manifest.get("split_query_ids") != ids:
                raise ValueError("ordered split query IDs differ from the manifest")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"Cannot reuse evaluation cache at {cache_path}: {exc}. "
                "Use overwrite=True (CLI: --overwrite) to recreate it, "
                "or choose another cache_dir."
            ) from exc
        return splits

    if cache_path.exists() and not cache_path.is_dir():
        raise ValueError(f"cache_dir must be a directory: {cache_path}")
    _check_cache_ownership(cache_path)
    source_options = {"split": dataset_split}
    if revision is not None:
        source_options["revision"] = revision
    original = load_dataset(dataset_path, dataset_name, **source_options)
    if not isinstance(original, Dataset):
        raise TypeError(
            "The source must be a Dataset, not a DatasetDict or streaming dataset."
        )
    _query_ids(original)
    partition = original.train_test_split(test_size=dev_size, seed=seed, shuffle=True)
    splits = DatasetDict({"dev": partition["test"], "test": partition["train"]})
    ids = _split_ids(splits)
    manifest = {
        "version": 1,
        "requested": requested,
        "source": {
            "fingerprint": getattr(original, "_fingerprint", None),
            "resolved_revision": getattr(original, "_revision", None),
        },
        "split_query_ids": ids,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Stage before replacing anything so source/serialization errors leave the
    # previous cache intact. Removing the old manifest first fails closed if
    # replacement is interrupted. Never follow symlinks while removing entries.
    with tempfile.TemporaryDirectory(
        prefix="query-splits-", dir=cache_path.parent
    ) as staging:
        staged = Path(staging) / "splits"
        splits.save_to_disk(str(staged))
        (staged / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        cache_path.mkdir(exist_ok=True)
        _check_cache_ownership(cache_path)
        (cache_path / MANIFEST_NAME).unlink(missing_ok=True)
        for name in ("dev", "test", "dataset_dict.json"):
            target = cache_path / name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            (staged / name).replace(target)
        (staged / MANIFEST_NAME).replace(cache_path / MANIFEST_NAME)
    return splits


def evaluation_config(config_path: str | Path | None = None) -> dict:
    """Load default configuration or an explicit YAML file using root-relative paths."""
    if config_path is None:
        return project_config.get_evaluation_config()
    return project_config.get_evaluation_config(config_path)


def prepare_from_config(settings: dict, *, overwrite: bool = False) -> DatasetDict:
    """Pass configuration explicitly to the backwards-compatible split helper."""
    keys = (
        "cache_dir",
        "dev_size",
        "seed",
        "dataset_path",
        "dataset_name",
        "dataset_split",
        "revision",
    )
    return load_or_create_retrieval_splits(
        **{key: settings[key] for key in keys if key in settings},
        overwrite=overwrite,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional evaluation YAML file")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly recreate cached splits"
    )
    args = parser.parse_args(argv)
    settings = evaluation_config(args.config)
    splits = prepare_from_config(settings, overwrite=args.overwrite)
    print(
        f"Query cache: {project_config.resolve_project_path(settings.get('cache_dir', 'data/langchain_queries'))}"
    )
    print(f"Dev examples: {len(splits['dev'])}; test examples: {len(splits['test'])}")


if __name__ == "__main__":
    main()
