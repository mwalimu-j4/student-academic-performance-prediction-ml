import json
from pathlib import Path

from student_performance.uci697 import LABELS, PREDICTION_FEATURES, load_uci697


ROOT = Path(__file__).resolve().parents[1]


def test_uci697_is_verified_and_has_genuine_holdout_capacity():
    frame = load_uci697(ROOT / "data" / "raw" / "uci-697" / "data.csv")
    assert len(frame) == 4424
    assert frame.duplicated().sum() == 0
    assert frame.isna().sum().sum() == 0
    assert sorted(frame["Target"].unique()) == sorted(LABELS)
    assert int(round(len(frame) * 0.25)) >= 1000


def test_production_metadata_excludes_downstream_features_and_uses_valid_labels():
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    assert metadata["labels"] == LABELS
    assert metadata["features"] == PREDICTION_FEATURES
    assert not any("Curricular units" in name for name in metadata["features"])


def test_saved_model_accepts_verified_feature_schema():
    import joblib

    frame = load_uci697(ROOT / "data" / "raw" / "uci-697" / "data.csv")
    model = joblib.load(ROOT / "models" / "student_outcome_pipeline.joblib")
    prediction = model.predict(frame[PREDICTION_FEATURES].iloc[[0]])[0]
    assert prediction in LABELS
