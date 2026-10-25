"""Model construction and evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a transformer whose fitting occurs inside the model pipeline."""
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )


def build_pipeline(
    features: pd.DataFrame, model_name: str = "random_forest", random_state: int = 42
) -> Pipeline:
    """Create a leakage-safe preprocessing and classifier pipeline."""
    if model_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
            max_depth=5,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline([("preprocessor", build_preprocessor(features)), ("classifier", classifier)])


def evaluate_predictions(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return binary test metrics with pass (1) as the positive class."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["fail", "pass"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "positive_class": {"label": 1, "name": "pass"},
        "averaging": {
            "precision": "binary, pos_label=1",
            "recall": "binary, pos_label=1",
            "f1": "binary, pos_label=1",
        },
        "test_examples": int(len(y_test)),
        "actual_class_distribution": {
            "fail": int((y_test == 0).sum()),
            "pass": int((y_test == 1).sum()),
        },
        "predicted_class_distribution": {
            "fail": int((predictions == 0).sum()),
            "pass": int((predictions == 1).sum()),
        },
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "at_risk_fail_class": {
            "class_label": 0,
            "class_name": "fail",
            "recall": report["fail"]["recall"],
            "precision": report["fail"]["precision"],
            "f1": report["fail"]["f1-score"],
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def evaluate_cross_validation(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, cv) -> dict:
    """Evaluate only on training folds; the holdout test set remains untouched."""
    from sklearn.model_selection import cross_validate

    scores = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
        n_jobs=1,
        return_train_score=False,
    )
    return {
        metric: {
            "mean": float(scores[f"test_{metric}"].mean()),
            "std": float(scores[f"test_{metric}"].std()),
        }
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }


def save_confusion_matrix(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, path: Path) -> None:
    """Save a labelled confusion-matrix figure."""
    predictions = model.predict(x_test)
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["fail", "pass"],
        cmap="Blues",
        ax=axis,
    )
    axis.set_title("Student Performance Classification")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_model_comparison_plot(metrics: dict, path: Path) -> None:
    """Save a readable comparison of held-out test metrics."""
    names = list(metrics)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [[metrics[name]["test"][metric] for name in names] for metric in metric_names]
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = range(len(names))
    width = 0.15
    for index, (metric, metric_values) in enumerate(zip(metric_names, values)):
        axis.bar(
            [position + (index - 2) * width for position in positions],
            metric_values,
            width,
            label=metric.upper() if metric == "f1" else metric.replace("_", "-").title(),
        )
    axis.set_xticks(list(positions), [name.replace("_", " ").title() for name in names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Held-out Test Metrics")
    axis.legend(ncol=3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
