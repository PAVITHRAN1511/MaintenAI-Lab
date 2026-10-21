import pandas as pd

from core.eda import (
    distribution_feature_columns,
    prepare_correlation_data,
    prepare_distribution_data,
    failure_class_statistics,
)


def eda_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3, 4],
            "Product ID": ["M1", "L2", "M3", "H4"],
            "Type": ["M", "L", "M", "H"],
            "Air temperature [K]": [298.1, 298.2, 298.3, 298.4],
            "Process temperature [K]": [308.6, 308.7, 308.8, 308.9],
            "Rotational speed [rpm]": [1551, 1408, 1498, 1433],
            "Torque [Nm]": [42.8, 46.3, 39.1, 45.7],
            "Tool wear [min]": [0, 3, 5, 7],
            "Machine failure": [0, 0, 0, 1],
            "TWF": [0, 0, 0, 0],
        }
    )


def test_distribution_selector_excludes_identifiers_and_outcomes() -> None:
    features = distribution_feature_columns(eda_frame())
    assert features[:5] == [
        "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
        "Torque [Nm]", "Tool wear [min]",
    ]
    assert "UDI" not in features
    assert "Product ID" not in features
    assert "Machine failure" not in features
    assert "TWF" not in features


def test_distribution_preparation_uses_numeric_observations_and_category_counts() -> None:
    frame = eda_frame()
    numeric = prepare_distribution_data(frame, "Torque [Nm]")
    categorical = prepare_distribution_data(frame, "Type")
    assert numeric.kind == "numeric"
    assert numeric.data["Torque [Nm]"].tolist() == frame["Torque [Nm]"].tolist()
    assert categorical.kind == "categorical"
    assert categorical.data["Count"].sum() == len(frame)


def test_failure_class_statistics_include_percentages() -> None:
    statistics = failure_class_statistics(eda_frame())
    assert statistics is not None
    assert statistics["Count"].sum() == 4
    assert statistics.loc[statistics["Machine failure"] == "Failure", "Percentage"].item() == 25.0


def test_correlation_data_marks_outcome_labels_and_excludes_udi() -> None:
    correlation = prepare_correlation_data(eda_frame())
    assert correlation is not None
    assert "UDI" not in correlation.columns
    assert "Machine failure (target)" in correlation.columns
    assert "TWF (failure subtype label)" in correlation.columns
