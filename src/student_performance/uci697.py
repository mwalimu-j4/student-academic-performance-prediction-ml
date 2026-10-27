"""Leakage-aware data and model helpers for UCI dataset 697."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_URL = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
DATASET_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
DATASET_DOI = "https://doi.org/10.24432/C5MC89"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
TARGET = "Target"
LABELS = ["Dropout", "Enrolled", "Graduate"]

# These are known at admission/enrolment. Curricular-unit progress and grades are
# intentionally excluded because they are downstream of the prediction point.
PREDICTION_FEATURES = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    "Admission grade", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International", "Unemployment rate",
    "Inflation rate", "GDP",
]
DOWNSTREAM_FEATURES = [
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
]


def load_uci697(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    frame.columns = [column.rstrip() for column in frame.columns]
    required = set(PREDICTION_FEATURES + DOWNSTREAM_FEATURES + [TARGET])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"UCI 697 schema is missing columns: {sorted(missing)}")
    if frame.empty or frame[TARGET].isna().any():
        raise ValueError("UCI 697 data is empty or contains missing targets.")
    if sorted(frame[TARGET].unique()) != sorted(LABELS):
        raise ValueError(f"Unexpected target labels: {sorted(frame[TARGET].unique())}")
    if frame.duplicated().any():
        raise ValueError("Duplicate records detected; refusing to train on duplicated data.")
    if frame.isna().any().any():
        raise ValueError("Missing values detected; refusing silent imputation.")
    return frame


def build_pipeline(
    features: pd.DataFrame,
    model_name: str = "random_forest",
    class_weight: str | dict[str, float] | None = "balanced",
) -> Pipeline:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=400, class_weight=class_weight, random_state=42, n_jobs=-1
        )
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=42
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", transformer), ("classifier", classifier)])


def evaluate_predictions(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    report = classification_report(
        y_test, predictions, labels=LABELS, target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_test, predictions, labels=LABELS, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_test, predictions, labels=LABELS, average="weighted", zero_division=0
    )
    return {
        "test_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro": {"precision": float(macro[0]), "recall": float(macro[1]), "f1": float(macro[2])},
        "weighted": {"precision": float(weighted[0]), "recall": float(weighted[1]), "f1": float(weighted[2])},
        "per_class": {label: report[label] for label in LABELS},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=LABELS).tolist(),
        "roc_auc_ovr_macro": float(roc_auc_score(y_test, probabilities, labels=LABELS, multi_class="ovr", average="macro")),
        "roc_auc_ovr_weighted": float(roc_auc_score(y_test, probabilities, labels=LABELS, multi_class="ovr", average="weighted")),
        "labels": LABELS,
    }
