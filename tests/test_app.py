from app import MODEL_PATH, validate_input


def test_deployment_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_app_rejects_forbidden_grade_fields_and_invalid_values():
    features = ["school", "age"]
    values = {"school": "GP", "age": 99, "G1": 10}

    errors = validate_input(values, features)

    assert any("G1, G2, and G3" in error for error in errors)
    assert any("schema" in error.lower() for error in errors)


def test_app_validation_accepts_valid_feature_values():
    features = ["school", "age"]
    values = {"school": "GP", "age": 17}

    assert validate_input(values, features) == []
