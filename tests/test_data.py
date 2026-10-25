import pandas as pd
import pytest

from student_performance.data import EXPECTED_COLUMNS, PRIMARY_FEATURES, prepare_classification_data


def test_prepare_classification_data_excludes_grades_and_creates_threshold_label():
    frame = pd.DataFrame(
        [
            ["GP", "F", 18, "U", "GT3", "A", 4, 4, "at_home", "teacher", "course",
             "mother", 2, 2, 0, "yes", "no", "no", "no", "yes", "yes", "no", "no",
             4, 3, 4, 1, 1, 3, 6, 5, 6, 9],
            ["GP", "M", 17, "U", "GT3", "T", 3, 2, "services", "other", "reputation",
             "father", 1, 3, 0, "no", "yes", "no", "yes", "yes", "yes", "yes", "no",
             5, 3, 3, 1, 2, 5, 2, 12, 13, 14],
        ],
        columns=EXPECTED_COLUMNS,
    )
    features, target = prepare_classification_data(frame)

    assert list(features.columns) == PRIMARY_FEATURES
    assert target.tolist() == [0, 1]
    assert "G1" not in features
    assert "G2" not in features
    assert "G3" not in features


def test_prepare_classification_data_rejects_missing_column():
    frame = pd.DataFrame(columns=[column for column in EXPECTED_COLUMNS if column != "G3"])

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_classification_data(frame)

