import pandas as pd

from core.data_loader import validate_ai4i_schema, validate_feature_selection


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "UDI": [1, 2],
            "Product ID": ["M14860", "L47181"],
            "Type": ["M", "L"],
            "Air temperature [K]": [298.1, 298.2],
            "Process temperature [K]": [308.6, 308.7],
            "Rotational speed [rpm]": [1551, 1408],
            "Torque [Nm]": [42.8, 46.3],
            "Tool wear [min]": [0, 3],
            "Machine failure": [0, 1],
            "TWF": [0, 0],
            "HDF": [0, 1],
            "PWF": [0, 0],
            "OSF": [0, 0],
            "RNF": [0, 0],
        }
    )


def test_valid_ai4i_schema_passes() -> None:
    assert validate_ai4i_schema(valid_frame()).is_valid


def test_schema_reports_missing_required_column() -> None:
    report = validate_ai4i_schema(valid_frame().drop(columns="Torque [Nm]"))
    assert not report.is_valid
    assert "Torque [Nm]" in report.errors[0]


def test_schema_reports_non_numeric_sensor_value() -> None:
    frame = valid_frame()
    frame["Torque [Nm]"] = frame["Torque [Nm]"].astype(object)
    frame.loc[0, "Torque [Nm]"] = "bad-value"
    report = validate_ai4i_schema(frame)
    assert not report.is_valid
    assert any("Torque [Nm]" in error and "numeric" in error for error in report.errors)


def test_machine_failure_target_rejects_failure_mode_features() -> None:
    report = validate_feature_selection(
        ["Air temperature [K]", "TWF", "HDF", "Machine failure"], "Machine failure"
    )
    assert not report.is_valid
    assert "Data leakage detected" in report.errors[0]
    assert "TWF" in report.errors[0]


def test_machine_failure_target_allows_sensor_features() -> None:
    report = validate_feature_selection(
        ["Air temperature [K]", "Torque [Nm]", "Type"], "Machine failure"
    )
    assert report.is_valid
