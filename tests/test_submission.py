from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class SubmissionTests(unittest.TestCase):
    def test_source_archive_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.zip"
            result = subprocess.run(
                [sys.executable, str(ROOT / "package_submission.py"),
                 "--source-only", "--output", str(output)],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
            self.assertIn("nlp_homework/requirements.txt", names)
            self.assertIn("nlp_homework/results/data_selection.json", names)
            self.assertFalse(any("/artifacts/" in name for name in names))
            self.assertFalse(any("/data/" in name for name in names))
            self.assertFalse(any("__pycache__" in name for name in names))
            self.assertFalse(any("/.git/" in name for name in names))

    def test_dataset_directory_supports_flat_and_nested_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation", "test"):
                (root / split).mkdir()
            (root / "train" / "train").mkdir()
            environment = dict(os.environ, PG19_DATA_DIR=str(root),
                               PYTHONPATH=str(ROOT))
            result = subprocess.run(
                [sys.executable, "-B", "-c",
                 "import config; print(config.TRAIN_DIR); print(config.VALID_DIR)"],
                cwd=directory, env=environment,
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(result.stdout.splitlines(),
                             [str(root / "train" / "train"),
                              str(root / "validation")])

    def make_processed(self, root):
        processed = root / "data" / "processed"
        processed.mkdir(parents=True)
        documents = {
            "train": [{"book_id": "a", "tokens": ["the", "cat", "."]}],
            "validation": [{"book_id": "b", "tokens": ["the", "<unk>"]}],
            "test": [{"book_id": "c", "tokens": ["cat", "."]}],
        }
        for split, values in documents.items():
            (processed / f"{split}.json").write_text(json.dumps(values))
        (processed / "vocab.json").write_text(
            json.dumps([".", "<bos>", "<eos>", "<unk>", "cat", "the"])
        )
        return processed

    def test_manifest_records_selection_without_modifying_data(self):
        module = importlib.import_module("package_submission")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = self.make_processed(root)
            before = (processed / "train.json").read_bytes()
            output = root / "results" / "selection.json"
            manifest = module.write_manifest(
                processed, output, seed=42,
                token_limits={"train": 3, "validation": 2, "test": 2},
                min_freq=2, max_vocab_size=20_000,
            )
            self.assertEqual(manifest["splits"]["train"]["documents"],
                             [{"book_id": "a", "token_count": 3}])
            self.assertEqual(manifest["splits"]["train"]["sha256"],
                             hashlib.sha256(before).hexdigest())
            self.assertEqual((processed / "train.json").read_bytes(), before)
            self.assertEqual(json.loads(output.read_text()), manifest)

    def test_manifest_rejects_shared_books(self):
        module = importlib.import_module("package_submission")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = self.make_processed(root)
            (processed / "test.json").write_text(json.dumps(
                [{"book_id": "a", "tokens": ["cat"]}]
            ))
            with self.assertRaisesRegex(ValueError, "book"):
                module.write_manifest(
                    processed, root / "selection.json", seed=42,
                    token_limits={"train": 3, "validation": 2, "test": 2},
                    min_freq=2, max_vocab_size=20_000,
                )

    def test_full_archive_contains_models_and_processed_data(self):
        module = importlib.import_module("package_submission")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in module.SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            (root / "lm").mkdir(exist_ok=True)
            (root / "lm" / "ngram.py").write_text("value = 1")
            for name in module.ASSET_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            (root / ".aws").mkdir()
            (root / ".aws" / "credentials").write_text("private")
            output = root / "dist" / "submission.zip"
            module.create_archive(root, output, include_assets=True)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
            self.assertIn("nlp_homework/artifacts/lstm_best.pt", names)
            self.assertIn("nlp_homework/data/processed/train.json", names)
            self.assertFalse(any(".aws" in name for name in names))

    def test_missing_assets_do_not_create_partial_archive(self):
        module = importlib.import_module("package_submission")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in module.SOURCE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            output = root / "submission.zip"
            with self.assertRaises(FileNotFoundError):
                module.create_archive(root, output, include_assets=True)
            self.assertFalse(output.exists())

    def test_trigram_validation_reproduces_selected_smoothing(self):
        from lm.ngram import count_ngrams

        notebook = json.loads((ROOT / "02_ngram_models.ipynb").read_text())
        source = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "trigram_ppl =" in "".join(cell.get("source", []))
        )
        documents = [{"book_id": "fixture", "tokens": ["a", "b"]}]
        bigrams, bigram_contexts = count_ngrams(documents, 2)
        trigrams, trigram_contexts = count_ngrams(documents, 3)
        namespace = {
            "importlib": importlib,
            "valid_documents": documents,
            "bigram_counts": bigrams,
            "bigram_context_counts": bigram_contexts,
            "trigram_counts": trigrams,
            "trigram_context_counts": trigram_contexts,
            "vocab": {"a", "b", "<bos>", "<eos>", "<unk>"},
        }
        with redirect_stdout(io.StringIO()):
            exec(compile(source, "validation_cell", "exec"), namespace)
        self.assertAlmostEqual(namespace["trigram_ppl"], 1.002997002997003)


if __name__ == "__main__":
    unittest.main()
