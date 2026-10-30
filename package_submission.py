import argparse
import hashlib
import json
from pathlib import Path
import zipfile


SOURCE_FILES = (
    ".gitignore",
    "README.md",
    "requirements.txt",
    "config.py",
    "app.py",
    "package_submission.py",
    "01_data_preparation.ipynb",
    "02_ngram_models.ipynb",
    "03_lstm_model.ipynb",
    "04_evaluation.ipynb",
    "lm/__init__.py",
    "lm/preprocessing.py",
    "lm/ngram.py",
    "lm/lstm.py",
    "lm/evaluation.py",
    "lm/generation.py",
    "results/data_selection.json",
    "results/lstm_history.json",
    "results/test_results.csv",
)

ASSET_FILES = (
    "artifacts/bigram.pkl",
    "artifacts/trigram.pkl",
    "artifacts/lstm_best.pt",
    "data/processed/train.json",
    "data/processed/validation.json",
    "data/processed/test.json",
    "data/processed/vocab.json",
)


def write_manifest(
    processed_dir, output_path, *, seed, token_limits,
    min_freq, max_vocab_size,
):
    processed_dir = Path(processed_dir)
    vocab_path = processed_dir / "vocab.json"
    vocab_list = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = set(vocab_list)

    if len(vocab) != len(vocab_list):
        raise ValueError("Duplicate vocabulary tokens.")
    if not {"<bos>", "<eos>", "<unk>"}.issubset(vocab):
        raise ValueError("Missing special vocabulary tokens.")

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "token_limits": token_limits,
        "vocabulary": {
            "size": len(vocab),
            "minimum_frequency": min_freq,
            "maximum_size": max_vocab_size,
            "sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        },
        "splits": {},
    }
    previous_ids = set()

    for split in ("train", "validation", "test"):
        path = processed_dir / f"{split}.json"
        documents = json.loads(path.read_text(encoding="utf-8"))
        ids = [document["book_id"] for document in documents]

        if len(ids) != len(set(ids)) or previous_ids.intersection(ids):
            raise ValueError(f"Duplicate book IDs in {split}.")
        previous_ids.update(ids)

        if any(token not in vocab for document in documents
               for token in document["tokens"]):
            raise ValueError(f"Out-of-vocabulary tokens in {split}.")

        token_count = sum(len(document["tokens"]) for document in documents)
        if not 0 < token_count <= token_limits[split]:
            raise ValueError(f"Invalid token count in {split}.")

        manifest["splits"][split] = {
            "book_count": len(documents),
            "token_count": token_count,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "documents": [
                {"book_id": document["book_id"],
                 "token_count": len(document["tokens"])}
                for document in documents
            ],
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def create_archive(project_dir, output_path, *, include_assets=True):
    project_dir = Path(project_dir).resolve()
    output_path = Path(output_path).resolve()

    if output_path.suffix.lower() != ".zip":
        raise ValueError("The submission output must be a .zip file.")

    names = SOURCE_FILES + (ASSET_FILES if include_assets else ())
    missing = [name for name in names if not (project_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing submission files: " + ", ".join(missing))

    files = [project_dir / name for name in names]
    files.extend((project_dir / "tests").glob("*.py"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
    ) as archive:
        for path in sorted(set(files)):
            archive.write(
                path, Path("nlp_homework") / path.relative_to(project_dir),
            )

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    filename = "nlp_homework_source.zip" if args.source_only else "nlp_homework_submission.zip"
    output = args.output or root / "dist" / filename
    archive = create_archive(root, output, include_assets=not args.source_only)
    print(archive)
    print(f"Size: {archive.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
