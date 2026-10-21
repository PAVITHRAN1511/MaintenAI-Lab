"""CSV loading and AI4I 2020 dataset validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
}

NUMERIC_COLUMNS = {
    "UDI",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
}

FAILURE_TYPE_COLUMNS = ("TWF", "HDF", "PWF", "OSF", "RNF")
LEAKAGE_COLUMNS = frozenset(("Machine failure", *FAILURE_TYPE_COLUMNS))


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass
class LoadResult:
    dataframe: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)


def load_csv(source: str | Path | BinaryIO | TextIO) -> LoadResult:
    """Read a CSV while returning user-friendly errors rather than raising into the UI."""
    try:
        return LoadResult(dataframe=pd.read_csv(source))
    except FileNotFoundError:
        return LoadResult(errors=["The specified CSV path does not exist."])
    except PermissionError:
        return LoadResult(errors=["The CSV cannot be read due to file permissions."])
    except pd.errors.EmptyDataError:
        return LoadResult(errors=["The selected file is empty."])
    except UnicodeDecodeError:
        return LoadResult(errors=["The file is not a UTF-8-readable CSV. Re-save it as UTF-8 CSV and try again."])
    except Exception as exc:  # pandas exposes several parser-specific exceptions
        return LoadResult(errors=[f"Unable to parse CSV: {exc}"])


def validate_ai4i_schema(df: pd.DataFrame) -> ValidationReport:
    """Validate the modelling fields and common integrity constraints of AI4I 2020."""
    report = ValidationReport()
    missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        report.errors.append("Missing required AI4I columns: " + ", ".join(missing_columns) + ".")
        return report

    if df.empty:
        report.errors.append("The dataset has no data rows.")

    missing_values = df[list(REQUIRED_COLUMNS)].isna().sum()
    missing_values = missing_values[missing_values > 0]
    for column, count in missing_values.items():
        report.errors.append(f"'{column}' contains {int(count)} missing value(s). Clean or impute this column before modelling.")

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() & df[column].notna()
        if invalid.any():
            report.errors.append(f"'{column}' must be numeric; {int(invalid.sum())} value(s) could not be converted.")

    if not pd.api.types.is_string_dtype(df["Product ID"]) and not pd.api.types.is_object_dtype(df["Product ID"]):
        report.errors.append("'Product ID' must contain text identifiers.")
    if not pd.api.types.is_string_dtype(df["Type"]) and not pd.api.types.is_object_dtype(df["Type"]):
        report.errors.append("'Type' must be a categorical text column (for example L, M, H).")

    failure = pd.to_numeric(df["Machine failure"], errors="coerce")
    if failure.notna().any() and not failure.dropna().isin([0, 1]).all():
        report.errors.append("'Machine failure' must be binary (0 or 1).")

    for column in FAILURE_TYPE_COLUMNS:
        if column in df.columns:
            mode = pd.to_numeric(df[column], errors="coerce")
            if mode.isna().any() or not mode.dropna().isin([0, 1]).all():
                report.errors.append(f"'{column}' must be binary (0 or 1) when supplied.")
        else:
            report.warnings.append(f"Optional failure-mode column '{column}' is absent; failure-type EDA will be limited.")
    return report


def validate_feature_selection(features: list[str], target: str) -> ValidationReport:
    """Reject target-derived features that would leak labels into a supervised model."""
    report = ValidationReport()
    if target not in LEAKAGE_COLUMNS:
        report.errors.append("Unsupported supervised target. Choose 'Machine failure' in the AI4I classification workflow.")
        return report

    leaked = sorted(set(features).intersection(LEAKAGE_COLUMNS))
    if leaked:
        report.errors.append(
            "Data leakage detected: " + ", ".join(leaked) + " cannot be used as a feature when predicting " + target + "."
        )
    return report


def suggested_predictors(df: pd.DataFrame) -> list[str]:
    """Return eligible AI4I sensor/process inputs, excluding IDs and failure-derived fields."""
    candidates = [column for column in df.columns if column not in LEAKAGE_COLUMNS and column not in {"UDI", "Product ID"}]
    return [column for column in candidates if not np.issubdtype(df[column].dtype, np.datetime64)]
