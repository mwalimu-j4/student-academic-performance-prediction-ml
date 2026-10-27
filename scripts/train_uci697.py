"""Train, compare, and persist the production UCI 697 multiclass model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from student_performance.uci697 import (  # noqa: E402
    DATASET_DOI, DATASET_LICENSE, DATASET_URL, LABELS, PREDICTION_FEATURES, TARGET,
    build_pipeline, evaluate_predictions, load_uci697,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_path = ROOT / "data" / "raw" / "uci-697" / "data.csv"
    if not data_path.exists():
        raise SystemExit("Download UCI 697 first: python scripts/download_uci697.py")
    frame = load_uci697(data_path)
    features, target = frame[PREDICTION_FEATURES], frame["Target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics: dict = {
        "dataset": {
            "name": "Predict Students' Dropout and Academic Success",
            "uci_id": 697, "url": DATASET_URL, "doi": DATASET_DOI,
            "license": DATASET_LICENSE, "rows": len(frame), "columns": len(frame.columns),
            "duplicate_rows": int(frame.duplicated().sum()),
            "missing_values": int(frame.isna().sum().sum()),
            "target_distribution": frame["Target"].value_counts().reindex(LABELS).to_dict(),
        },
        "experiment": {
            "random_state": 42, "test_fraction": 0.25,
            "training_rows": len(x_train), "test_rows": len(x_test),
            "features": PREDICTION_FEATURES,
            "excluded_downstream_features": [c for c in frame.columns if c not in PREDICTION_FEATURES + ["Target"]],
        },
        "models": {},
    }
    for name in ("logistic_regression", "random_forest"):
        model = build_pipeline(x_train, name)
        cv_scores = cross_validate(
            model, x_train, y_train, cv=cv,
            scoring=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        test = evaluate_predictions(model, x_test, y_test)
        metrics["models"][name] = {
            "cross_validation": {
                metric: {"mean": float(cv_scores[f"test_{metric}"].mean()), "std": float(cv_scores[f"test_{metric}"].std())}
                for metric in ("accuracy", "f1_macro", "precision_macro", "recall_macro")
            },
            "test": test,
        }
        joblib.dump(model, ROOT / "models" / f"uci697_{name}.joblib")
    best = max(metrics["models"], key=lambda n: metrics["models"][n]["test"]["macro"]["f1"])
    joblib.dump(
        joblib.load(ROOT / "models" / f"uci697_{best}.joblib"),
        ROOT / "models" / "student_outcome_pipeline.joblib",
    )
    metrics["selected_model"] = best
    (ROOT / "models" / "model_metadata.json").write_text(json.dumps({
        "model": best, "artifact": "models/student_outcome_pipeline.joblib",
        "target": TARGET, "labels": LABELS, "features": PREDICTION_FEATURES,
        "prediction_time": "admission/enrollment; downstream curricular-unit progress excluded",
    }, indent=2), encoding="utf-8")
    (ROOT / "results" / "uci697_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"model": name, "test_accuracy": value["test"]["accuracy"],
         "test_macro_f1": value["test"]["macro"]["f1"],
         "test_weighted_f1": value["test"]["weighted"]["f1"],
         "test_roc_auc_ovr_macro": value["test"]["roc_auc_ovr_macro"]}
        for name, value in metrics["models"].items()
    ]).to_csv(ROOT / "results" / "uci697_model_comparison.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
