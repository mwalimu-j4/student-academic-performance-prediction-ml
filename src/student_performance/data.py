"""Dataset download, validation, and target preparation utilities."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/320/student+performance"
DATASET_DOI = "https://doi.org/10.24432/C5TG7T"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
DATASET_CITATION = (
    "Cortez, P. (2008). Student Performance [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T"
)
ORIGINAL_FILES = ["student-mat.csv", "student-por.csv", "student.txt", "student-merge.R"]
EXPECTED_COLUMNS = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
]

PRIMARY_FEATURES = [column for column in EXPECTED_COLUMNS if column not in {"G1", "G2", "G3"}]
CATEGORICAL_COLUMNS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]


def download_dataset(destination: Path) -> Path:
    """Download and extract the official UCI archive into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "student-performance.zip"
    with urlopen(DATASET_URL, timeout=120) as response:
        archive_path.write_bytes(response.read())

    with ZipFile(archive_path) as outer_archive:
        outer_archive.extractall(destination / "_outer")
    nested_archive = destination / "_outer" / "student.zip"
    with ZipFile(nested_archive) as inner_archive:
        inner_archive.extractall(destination)
    archive_path.unlink()
    nested_archive.unlink()
    (destination / "_outer" / ".student.zip_old").unlink(missing_ok=True)
    (destination / "_outer").rmdir()
    return destination / "student-mat.csv"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_manifest(raw_directory: Path) -> Path:
    """Record the official source, license, citation, files, and local hashes."""
    manifest = {
        "source": {
            "repository": "UCI Machine Learning Repository",
            "dataset": "Student Performance",
            "dataset_id": 320,
            "dataset_page": DATASET_PAGE,
            "download_url": DATASET_URL,
            "doi": DATASET_DOI,
            "license": DATASET_LICENSE,
            "citation": DATASET_CITATION,
        },
        "original_files": ORIGINAL_FILES,
        "local_files": {
            name: {
                "path": f"data/raw/{name}",
                "sha256": sha256_file(raw_directory / name),
            }
            for name in ORIGINAL_FILES
            if (raw_directory / name).exists()
        },
    }
    path = raw_directory / "source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_math_data(path: Path) -> pd.DataFrame:
    """Load and validate the official Mathematics CSV."""
    frame = pd.read_csv(path, sep=";")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected UCI schema. Expected columns exactly: "
            f"{EXPECTED_COLUMNS}; received: {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("The dataset is empty.")
    return frame


def verify_math_data(frame: pd.DataFrame) -> dict:
    """Audit the loaded data before any transformation."""
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Schema verification must pass before auditing the dataset.")
    categorical_values = {
        column: sorted(frame[column].dropna().unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }
    return {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "columns": list(frame.columns),
        "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "duplicate_records": int(frame.duplicated(keep=False).sum()),
        "duplicate_rows_beyond_first": int(frame.duplicated().sum()),
        "missing_values": {column: int(count) for column, count in frame.isna().sum().items()},
        "target": {
            "column": "G3",
            "documented_scale": "0-20",
            "pass_threshold": "G3 >= 10",
            "distribution": {
                str(value): int(count)
                for value, count in frame["G3"].value_counts().sort_index().items()
            },
            "binary_distribution": {
                "fail": int((frame["G3"] < 10).sum()),
                "pass": int((frame["G3"] >= 10).sum()),
            },
        },
        "categorical_unique_values": categorical_values,
    }


def preprocess_primary_data(frame: pd.DataFrame, output_path: Path) -> dict:
    """Create an auditable early-prediction table and save its processing audit."""
    audit = verify_math_data(frame)
    if any(audit["missing_values"].values()):
        raise ValueError(
            "Missing values were found; no rows were silently removed. "
            f"Review the audit: {audit['missing_values']}"
        )
    duplicate_rows_removed = int(frame.duplicated().sum())
    cleaned = frame.drop_duplicates(keep="first").copy()
    features, target = prepare_classification_data(cleaned)
    processed = features.copy()
    processed["pass"] = target
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return {
        "input_rows": int(len(frame)),
        "output_rows": int(len(processed)),
        "duplicate_rows_removed": duplicate_rows_removed,
        "missing_values_handled": "none found",
        "excluded_columns": ["G1", "G2", "G3"],
        "target_definition": "pass = 1 when G3 >= 10; fail = 0 otherwise",
        "output_path": "data/processed/student-mat-early-prediction.csv",
    }


def prepare_classification_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the approved leakage-controlled predictors and binary target."""
    missing = set(EXPECTED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    features = frame[PRIMARY_FEATURES].copy()
    target = (frame["G3"] >= 10).astype("int8").rename("pass")
    return features, target
