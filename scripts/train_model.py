"""Train and evaluate the approved primary classification experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_performance.data import load_math_data, prepare_classification_data
from student_performance.modeling import (
    build_pipeline,
    evaluate_cross_validation,
    evaluate_predictions,
    save_confusion_matrix,
    save_model_comparison_plot,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw_path = ROOT / "data" / "raw" / "student-mat.csv"
    results_path = ROOT / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    frame = load_math_data(raw_path)
    features, target = prepare_classification_data(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = {
        "experiment": {
            "random_state": 42,
            "test_size": 0.2,
            "cv_splits": 5,
            "stratified": True,
            "target": "pass = 1 when G3 >= 10; fail = 0 otherwise",
            "excluded_from_features": ["G1", "G2", "G3"],
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "feature_count_before_encoding": int(features.shape[1]),
            "positive_rate": float(target.mean()),
        },
        "selection_criterion": {
            "primary": "held-out fail-class recall",
            "tie_breaker": "held-out fail-class F1",
            "criterion": "Higher fail recall is preferred for identifying students at risk of failing; if tied, prefer higher fail F1.",
            "objective_threshold": "No externally imposed performance threshold; measured values are reported without claiming target attainment.",
        },
        "models": {},
    }

    for model_name in ["decision_tree", "random_forest"]:
        model = build_pipeline(features, model_name=model_name)
        cv_metrics = evaluate_cross_validation(model, x_train, y_train, cross_validator)
        model.fit(x_train, y_train)
        test_metrics = evaluate_predictions(model, x_test, y_test)
        all_metrics["models"][model_name] = {
            "cross_validation": cv_metrics,
            "test": test_metrics,
        }
        joblib.dump(model, results_path / f"{model_name}_pipeline.joblib")
        reloaded_model = joblib.load(results_path / f"{model_name}_pipeline.joblib")
        reload_predictions = reloaded_model.predict(x_test.iloc[[0, 1]]).tolist()
        all_metrics["models"][model_name]["reload_check"] = {
            "input_rows": 2,
            "predictions_on_held_out_examples": reload_predictions,
        }
        save_confusion_matrix(
            model, x_test, y_test, results_path / f"{model_name}_confusion_matrix.png"
        )

    save_model_comparison_plot(
        all_metrics["models"], results_path / "model_comparison.png"
    )
    summary_rows = []
    for model_name, model_metrics in all_metrics["models"].items():
        row = {"model": model_name}
        row.update({f"test_{metric}": model_metrics["test"][metric]
                    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]})
        row.update({
            "test_fail_recall": model_metrics["test"]["at_risk_fail_class"]["recall"],
            "test_fail_precision": model_metrics["test"]["at_risk_fail_class"]["precision"],
            "test_fail_f1": model_metrics["test"]["at_risk_fail_class"]["f1"],
        })
        row.update({f"cv_{metric}_mean": model_metrics["cross_validation"][metric]["mean"]
                    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(results_path / "model_comparison.csv", index=False)
    ranked = sorted(
        all_metrics["models"],
        key=lambda name: (
            all_metrics["models"][name]["test"]["at_risk_fail_class"]["recall"],
            all_metrics["models"][name]["test"]["at_risk_fail_class"]["f1"],
        ),
        reverse=True,
    )
    all_metrics["selection_criterion"]["ranking"] = ranked
    all_metrics["selection_criterion"]["recommended_model"] = ranked[0]

    with (results_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
