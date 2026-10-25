"""Download the official UCI Student Performance archive."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

from student_performance.data import (
    download_dataset,
    load_math_data,
    preprocess_primary_data,
    verify_math_data,
    write_source_manifest,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raw_directory = root / "data" / "raw"
    csv_path = download_dataset(raw_directory)
    manifest_path = write_source_manifest(raw_directory)
    frame = load_math_data(csv_path)
    verification = verify_math_data(frame)
    processing = preprocess_primary_data(
        frame, root / "data" / "processed" / "student-mat-early-prediction.csv"
    )
    report = {
        "source_manifest": "data/raw/source_manifest.json",
        "verification": verification,
        "processing": processing,
    }
    report_path = root / "results" / "data_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
