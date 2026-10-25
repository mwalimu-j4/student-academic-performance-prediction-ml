import pandas as pd
from sklearn.model_selection import train_test_split

from student_performance.modeling import build_pipeline, evaluate_predictions


def test_pipeline_handles_categorical_and_numeric_features():
    features = pd.DataFrame(
        {
            "school": ["GP", "MS", "GP", "MS"],
            "age": [16, 17, 18, 19],
            "absences": [2, 4, 0, 8],
        }
    )
    target = pd.Series([0, 1, 1, 0])
    model = build_pipeline(features)

    model.fit(features, target)

    assert model.predict(pd.DataFrame({"school": ["GP"], "age": [17], "absences": [3]})).shape == (1,)


def test_both_classifiers_are_supported():
    features = pd.DataFrame({"school": ["GP", "MS", "GP", "MS"], "age": [16, 17, 18, 19]})
    target = pd.Series([0, 1, 1, 0])

    for model_name in ["decision_tree", "random_forest"]:
        model = build_pipeline(features, model_name=model_name)
        model.fit(features, target)
        assert model.predict(features).shape == (4,)


def test_evaluation_declares_positive_class_and_test_distribution():
    features = pd.DataFrame({"school": ["GP", "MS", "GP", "MS"], "age": [16, 17, 18, 19]})
    target = pd.Series([0, 1, 1, 0])
    model = build_pipeline(features, model_name="random_forest").fit(features, target)

    metrics = evaluate_predictions(model, features, target)

    assert metrics["positive_class"] == {"label": 1, "name": "pass"}
    assert metrics["averaging"]["f1"] == "binary, pos_label=1"
    assert metrics["test_examples"] == 4
    assert sum(metrics["actual_class_distribution"].values()) == 4


def test_pipeline_does_not_include_target_or_period_grades():
    features = pd.DataFrame({"school": ["GP", "MS", "GP", "MS"], "age": [16, 17, 18, 19]})
    target = pd.Series([0, 1, 1, 0])
    model = build_pipeline(features).fit(features, target)
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()

    assert not any(name.endswith("G1") or name.endswith("G2") or name.endswith("G3")
                   for name in transformed_names)


def test_stratified_split_is_reproducible():
    features = pd.DataFrame({"age": range(20), "school": ["GP", "MS"] * 10})
    target = pd.Series([0, 1] * 10)
    first = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    second = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    for first_part, second_part in zip(first, second):
        if isinstance(first_part, pd.DataFrame):
            pd.testing.assert_frame_equal(first_part, second_part)
        else:
            pd.testing.assert_series_equal(first_part, second_part)
