"""Streamlit demonstration for the evaluated early-prediction Random Forest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_pipeline.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"
COMPARISON_PATH = ROOT / "results" / "model_comparison.csv"
MODEL_METADATA_PATH = ROOT / "models" / "model_metadata.json"

CATEGORICAL_OPTIONS = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["course", "home", "reputation", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
}

NUMERIC_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 3),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 75),
}


@st.cache_resource
def load_model() -> Any:
    """Load the committed evaluated pipeline once per application process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact is missing: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def validate_input(values: dict[str, Any], expected_features: list[str]) -> list[str]:
    """Return understandable validation errors without changing submitted values."""
    errors = []
    if set(values) != set(expected_features):
        errors.append("The input schema does not match the approved early-prediction features.")
    forbidden = {"G1", "G2", "G3"} & set(values)
    if forbidden:
        errors.append("G1, G2, and G3 are not accepted because they are unavailable at prediction time.")
    for feature, options in CATEGORICAL_OPTIONS.items():
        if feature in values and values[feature] not in options:
            errors.append(f"{feature} must be one of: {', '.join(options)}.")
    for feature, (minimum, maximum) in NUMERIC_RANGES.items():
        if feature not in values:
            continue
        value = values[feature]
        if not isinstance(value, int) or not minimum <= value <= maximum:
            errors.append(f"{feature} must be an integer from {minimum} to {maximum}.")
    return errors


def render_categorical(feature: str) -> str:
    return st.selectbox(feature, CATEGORICAL_OPTIONS[feature], key=feature)


def render_numeric(feature: str) -> int:
    minimum, maximum = NUMERIC_RANGES[feature]
    default = minimum if feature == "absences" else minimum
    return int(st.number_input(feature, min_value=minimum, max_value=maximum, value=default, step=1))


def collect_form_values() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for feature in [
        "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
        "reason", "guardian",
    ]:
        values[feature] = render_categorical(feature)
    for feature in ["age", "Medu", "Fedu", "traveltime", "studytime", "failures"]:
        values[feature] = render_numeric(feature)
    for feature in [
        "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
        "internet", "romantic",
    ]:
        values[feature] = render_categorical(feature)
    for feature in ["famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]:
        values[feature] = render_numeric(feature)
    return values


def render_evaluation() -> None:
    metrics = load_json(str(METRICS_PATH))
    st.subheader("Verified model evaluation")
    st.caption("Values below are loaded from the executed evaluation artifacts; they are not recomputed by the dashboard.")
    st.write(
        f"Held-out test set: {metrics['experiment']['test_rows']} examples "
        f"({metrics['models']['random_forest']['test']['actual_class_distribution']['fail']} fail, "
        f"{metrics['models']['random_forest']['test']['actual_class_distribution']['pass']} pass)."
    )
    rows = []
    for model_name, model_metrics in metrics["models"].items():
        test = model_metrics["test"]
        rows.append({
            "Model": model_name.replace("_", " ").title(),
            "Accuracy": test["accuracy"],
            "Precision (pass)": test["precision"],
            "Recall (pass)": test["recall"],
            "F1 (pass)": test["f1"],
            "ROC-AUC": test["roc_auc"],
            "Recall (fail)": test["at_risk_fail_class"]["recall"],
            "F1 (fail)": test["at_risk_fail_class"]["f1"],
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.write("Confusion matrices use the order [fail, pass].")
    st.write("Decision Tree:", metrics["models"]["decision_tree"]["test"]["confusion_matrix"])
    st.write("Random Forest:", metrics["models"]["random_forest"]["test"]["confusion_matrix"])
    st.image(str(ROOT / "results" / "model_comparison.png"), caption="Held-out test comparison")


def main() -> None:
    st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
    st.title("Student Academic Performance Prediction")
    st.write(
        "This educational demonstration estimates whether a student is likely to pass or fail "
        "using the evaluated UCI Mathematics Student Performance Random Forest."
    )
    st.info(
        "The prediction is decision support only. It must not be used as the sole basis for "
        "academic decisions, grading, discipline, or access to support."
    )

    try:
        metadata = load_json(str(MODEL_METADATA_PATH))
        model = load_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"The approved model could not be loaded: {error}")
        st.stop()

    features = metadata["features"]
    st.subheader("Early-prediction inputs")
    st.caption("G1, G2, and G3 are intentionally not requested because they are unavailable at the approved prediction point.")
    with st.form("prediction_form"):
        values = collect_form_values()
        submitted = st.form_submit_button("Predict outcome")

    if submitted:
        errors = validate_input(values, features)
        if errors:
            for error in errors:
                st.error(error)
        else:
            input_frame = pd.DataFrame([values], columns=features)
            prediction = int(model.predict(input_frame)[0])
            probability = float(model.predict_proba(input_frame)[0, 1])
            outcome = "Likely pass" if prediction == 1 else "Likely fail"
            st.subheader("Prediction")
            st.success(outcome) if prediction == 1 else st.warning(outcome)
            st.metric("Estimated pass probability", f"{probability:.1%}")
            st.caption(
                "This probability is the fitted model's estimate for this input, not a guarantee "
                "and not a calibrated certainty."
            )

    st.divider()
    render_evaluation()
    st.caption("Source: UCI Student Performance dataset, dataset 320. License: CC BY 4.0.")


if __name__ == "__main__":
    main()
