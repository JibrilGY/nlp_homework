import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "pg19"

if not DEFAULT_DATA_DIR.is_dir():
    DEFAULT_DATA_DIR = Path.home() / "Desktop" / "dataset"

DATA_DIR = Path(
    os.environ.get("PG19_DATA_DIR", str(DEFAULT_DATA_DIR))
).expanduser().resolve()

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
RESULT_DIR = PROJECT_DIR / "results"


def _split_dir(name):
    directory = DATA_DIR / name
    nested = directory / name
    return nested if nested.is_dir() else directory


TRAIN_DIR = _split_dir("train")
VALID_DIR = _split_dir("validation")
TEST_DIR = _split_dir("test")

SEED = 42

TRAIN_TOKEN_LIMIT = 1_000_000
VALID_TOKEN_LIMIT = 100_000
TEST_TOKEN_LIMIT = 100_000
