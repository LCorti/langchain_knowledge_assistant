"""Offline evaluation tests using real, small Hugging Face Dataset fixtures."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset, DatasetDict, disable_progress_bars, load_from_disk

from src.rag.evaluation import prepare_queries, retrieval

disable_progress_bars()


class RetrievalSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "queries"
        self.source = Dataset.from_dict(
            {
                "_id": [f"q{i}" for i in range(20)],
                "text": [f"Question {i}" for i in range(20)],
            }
        )
        self.loader = self.enterContext(
            patch.object(prepare_queries, "load_dataset", return_value=self.source)
        )

    def prepare(self, **kwargs):
        return prepare_queries.load_or_create_retrieval_splits(self.cache, **kwargs)

    def manifest(self):
        return json.loads((self.cache / prepare_queries.MANIFEST_NAME).read_text())

    def write_manifest(self, manifest):
        (self.cache / prepare_queries.MANIFEST_NAME).write_text(json.dumps(manifest))

    def test_deterministic_disjoint_complete_and_manifest(self):
        first = self.prepare(revision="pinned-commit")
        second = prepare_queries.load_or_create_retrieval_splits(
            self.root / "other", revision="pinned-commit"
        )
        self.assertEqual(len(first["dev"]), 4)
        self.assertEqual(len(first["test"]), 16)
        for name in ("dev", "test"):
            self.assertEqual(list(first[name]["_id"]), list(second[name]["_id"]))
        dev, test = set(first["dev"]["_id"]), set(first["test"]["_id"])
        self.assertFalse(dev & test)
        self.assertEqual(dev | test, set(self.source["_id"]))
        manifest = self.manifest()
        self.assertEqual(
            manifest["requested"],
            {
                "seed": 42,
                "dev_size": 0.2,
                "dataset_path": "freshstack/queries-oct-2024",
                "dataset_name": "langchain",
                "dataset_split": "test",
                "revision": "pinned-commit",
            },
        )
        self.assertEqual(manifest["source"]["fingerprint"], self.source._fingerprint)
        self.assertEqual(manifest["split_query_ids"]["dev"], list(first["dev"]["_id"]))
        self.loader.assert_called_with(
            "freshstack/queries-oct-2024",
            "langchain",
            split="test",
            revision="pinned-commit",
        )

    def test_cache_reuse_never_loads_remote_source(self):
        expected = self.prepare()
        self.loader.reset_mock()
        self.loader.side_effect = AssertionError("Remote loading is forbidden on reuse")
        actual = self.prepare()
        self.loader.assert_not_called()
        for name in ("dev", "test"):
            self.assertEqual(actual[name].to_dict(), expected[name].to_dict())

    def test_every_provenance_mismatch_requires_explicit_overwrite(self):
        self.prepare()
        self.loader.reset_mock()
        for change in (
            {"seed": 7},
            {"dev_size": 0.3},
            {"dataset_path": "other/queries"},
            {"dataset_name": "other"},
            {"dataset_split": "train"},
            {"revision": "v2"},
        ):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "overwrite=True"):
                    self.prepare(**change)
        self.loader.assert_not_called()

    def test_missing_invalid_and_unsupported_manifest(self):
        self.prepare()
        manifest_path = self.cache / prepare_queries.MANIFEST_NAME
        for contents in (None, "not-json", "[]", '{"version": 99}'):
            with self.subTest(contents=contents):
                if contents is None:
                    manifest_path.unlink()
                else:
                    manifest_path.write_text(contents)
                with self.assertRaisesRegex(ValueError, "--overwrite"):
                    self.prepare()

    def test_manifest_order_tampering_is_rejected(self):
        self.prepare()
        manifest = self.manifest()
        manifest["split_query_ids"]["dev"].reverse()
        self.write_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "ordered split query IDs"):
            self.prepare()

    def test_corrupt_cache_type_and_split_names(self):
        self.prepare()
        invalid = (
            self.source,
            DatasetDict({"train": self.source}),
            DatasetDict({"dev": self.source.select([]), "test": self.source}),
            DatasetDict(
                {"dev": self.source.select([0]), "test": self.source.select([0])}
            ),
        )
        for cached in invalid:
            with self.subTest(cached=type(cached).__name__):
                with patch.object(
                    prepare_queries, "load_from_disk", return_value=cached
                ):
                    with self.assertRaisesRegex(ValueError, "overwrite=True"):
                        self.prepare()

    def test_overwrite_preserves_unrelated_files_and_does_not_follow_symlinks(self):
        self.prepare()
        keep = self.cache / "user-notes.txt"
        keep.write_text("keep me")
        outside = self.root / "outside"
        (outside / "nested").mkdir(parents=True)
        sentinel = outside / "nested" / "precious.txt"
        sentinel.write_text("outside data")
        # A symlink under an owned directory must not cause external deletion.
        (self.cache / "dev" / "external").symlink_to(outside, target_is_directory=True)
        self.prepare(seed=7, overwrite=True)
        self.assertEqual(keep.read_text(), "keep me")
        self.assertEqual(sentinel.read_text(), "outside data")
        self.assertEqual(self.manifest()["requested"]["seed"], 7)
        self.assertIsInstance(load_from_disk(str(self.cache)), DatasetDict)

    def test_unrelated_dev_test_directories_are_preserved_without_download(self):
        for split in ("dev", "test"):
            directory = self.cache / split
            directory.mkdir(parents=True)
            (directory / "user-data.txt").write_text(f"precious {split}")
        # Neither a matching directory name, a DatasetDict marker alone, nor
        # empty JSON metadata is enough to authorize recursive deletion.
        for evidence in ("none", "dataset_dict", "empty_metadata", "manifest"):
            with self.subTest(evidence=evidence):
                if evidence == "dataset_dict":
                    (self.cache / "dataset_dict.json").write_text(
                        json.dumps({"splits": ["dev", "test"]})
                    )
                elif evidence == "empty_metadata":
                    for split in ("dev", "test"):
                        for name in ("dataset_info.json", "state.json"):
                            (self.cache / split / name).write_text("{}")
                elif evidence == "manifest":
                    self.write_manifest(
                        {
                            "version": 1,
                            "requested": {"seed": 42},
                            "split_query_ids": {"dev": ["a"], "test": ["b"]},
                        }
                    )
                before = {
                    path.relative_to(self.cache): path.read_bytes()
                    for path in self.cache.rglob("*")
                    if path.is_file()
                }
                with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                    self.prepare(overwrite=True)
                self.assertEqual(
                    before,
                    {
                        path.relative_to(self.cache): path.read_bytes()
                        for path in self.cache.rglob("*")
                        if path.is_file()
                    },
                )
                self.loader.assert_not_called()

    def test_valid_legacy_cache_without_manifest_can_be_overwritten(self):
        legacy = DatasetDict(
            {
                "dev": self.source.select(range(4)),
                "test": self.source.select(range(4, 20)),
            }
        )
        legacy.save_to_disk(str(self.cache))
        self.assertFalse((self.cache / prepare_queries.MANIFEST_NAME).exists())
        keep = self.cache / "user-notes.txt"
        keep.write_text("preserve legacy notes")
        updated = self.prepare(seed=7, overwrite=True)
        self.loader.assert_called_once()
        self.assertEqual(self.manifest()["requested"]["seed"], 7)
        self.assertEqual(keep.read_text(), "preserve legacy notes")
        self.assertEqual(
            load_from_disk(str(self.cache))["dev"].to_dict(),
            updated["dev"].to_dict(),
        )

    def test_incomplete_or_invalid_cache_metadata_is_rejected_before_download(self):
        self.prepare()
        self.loader.reset_mock()
        paths = (
            self.cache / "dataset_dict.json",
            self.cache / "dev" / "dataset_info.json",
            self.cache / "dev" / "state.json",
            self.cache / "test" / "dataset_info.json",
            self.cache / "test" / "state.json",
        )
        for path in paths:
            original = path.read_bytes()
            for replacement in (None, b"not-json", b"[]"):
                with self.subTest(path=path, replacement=replacement):
                    if replacement is None:
                        path.unlink()
                    else:
                        path.write_bytes(replacement)
                    with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                        self.prepare(overwrite=True)
                    self.loader.assert_not_called()
                    self.assertEqual(
                        path.read_bytes() if path.exists() else None, replacement
                    )
                    path.write_bytes(original)
        self.assertIsInstance(load_from_disk(str(self.cache)), DatasetDict)

    def test_bad_source_does_not_destroy_previous_cache(self):
        self.prepare()
        before = (self.cache / prepare_queries.MANIFEST_NAME).read_text()
        self.loader.return_value = Dataset.from_dict(
            {"_id": ["duplicate", "duplicate"]}
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            self.prepare(overwrite=True)
        self.assertEqual(
            (self.cache / prepare_queries.MANIFEST_NAME).read_text(), before
        )

    def test_invalid_source_types_ids_and_parameters(self):
        for source in (
            DatasetDict({"test": self.source}),
            Dataset.from_dict({"text": ["no id"]}),
            Dataset.from_dict({"_id": [None, "valid"]}),
            Dataset.from_dict({"_id": ["", "valid"]}),
        ):
            with self.subTest(source=source):
                self.loader.return_value = source
                with self.assertRaises(ValueError):
                    self.prepare()
        self.loader.reset_mock()
        for kwargs in (
            {"dev_size": 0},
            {"dev_size": 1},
            {"dev_size": True},
            {"dev_size": "0.2"},
            {"seed": True},
            {"overwrite": "yes"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.prepare(**kwargs)
        self.loader.assert_not_called()

    def test_alternate_id_columns(self):
        for column in ("query_id", "id"):
            with self.subTest(column=column):
                self.loader.return_value = self.source.rename_column("_id", column)
                splits = self.prepare(overwrite=True)
                self.assertEqual(len(splits["dev"]), 4)


class EvaluationEntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = {
            "seed": 17,
            "dev_size": 0.3,
            "cache_dir": str(self.root / "cache"),
            "results_dir": str(self.root / "results"),
            "split": "test",
            "cutoffs": [3, 8],
            "dataset_path": "fixture/queries",
            "dataset_name": "sample",
            "dataset_split": "validation",
            "revision": "pinned",
        }
        self.splits = DatasetDict(
            {
                "dev": Dataset.from_dict({"_id": ["a"]}),
                "test": Dataset.from_dict({"_id": ["b", "c"]}),
            }
        )

    def test_config_fields_are_forwarded_explicitly(self):
        with patch.object(
            prepare_queries, "load_or_create_retrieval_splits", return_value=self.splits
        ) as load:
            prepare_queries.prepare_from_config(self.settings, overwrite=True)
        expected = {
            key: value
            for key, value in self.settings.items()
            if key not in ("results_dir", "split", "cutoffs")
        }
        load.assert_called_once_with(**expected, overwrite=True)

    def test_default_config_uses_shared_loader(self):
        with patch.object(
            prepare_queries.project_config,
            "get_evaluation_config",
            return_value=self.settings,
        ) as load:
            self.assertEqual(prepare_queries.evaluation_config(), self.settings)
        load.assert_called_once_with()

    def test_explicit_yaml_resolves_paths_from_project_root(self):
        config = self.root / "evaluation.yaml"
        config.write_text(
            "cache_dir: data/custom\nresults_dir: results/custom\nseed: 9\n"
        )
        settings = prepare_queries.evaluation_config(config)
        resolver = prepare_queries.project_config.resolve_project_path
        self.assertEqual(settings["cache_dir"], str(resolver("data/custom")))
        self.assertEqual(settings["results_dir"], str(resolver("results/custom")))
        self.assertEqual(settings["seed"], 9)

    def test_prepare_main_config_and_overwrite(self):
        output = io.StringIO()
        with patch.object(
            prepare_queries, "evaluation_config", return_value=self.settings
        ) as config:
            with patch.object(
                prepare_queries, "prepare_from_config", return_value=self.splits
            ) as prepare:
                with contextlib.redirect_stdout(output):
                    prepare_queries.main(["--config", "chosen.yaml", "--overwrite"])
        config.assert_called_once_with(Path("chosen.yaml"))
        prepare.assert_called_once_with(self.settings, overwrite=True)
        self.assertIn("Dev examples: 1; test examples: 2", output.getvalue())

    def test_retrieval_main_reports_setup_only_without_results(self):
        output = io.StringIO()
        with patch.object(retrieval, "evaluation_config", return_value=self.settings):
            with patch.object(
                retrieval, "prepare_from_config", return_value=self.splits
            ) as prepare:
                with contextlib.redirect_stdout(output):
                    retrieval.main([])
        prepare.assert_called_once_with(self.settings, overwrite=False)
        self.assertIn("test (2 queries)", output.getvalue())
        self.assertIn("[3, 8]", output.getvalue())
        self.assertIn("metrics (including nDCG) are not implemented", output.getvalue())
        self.assertFalse(Path(self.settings["results_dir"]).exists())

    def test_retrieval_rejects_invalid_setup_before_loading(self):
        for change in (
            {"split": "train"},
            {"cutoffs": []},
            {"cutoffs": [0]},
            {"cutoffs": [True]},
        ):
            with self.subTest(change=change):
                with patch.object(
                    retrieval, "evaluation_config", return_value=self.settings | change
                ):
                    with patch.object(retrieval, "prepare_from_config") as prepare:
                        with self.assertRaises(ValueError):
                            retrieval.main([])
                        prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
