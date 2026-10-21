"""Plotly figures backed by loaded data only."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px

from core.data_loader import FAILURE_TYPE_COLUMNS, LEAKAGE_COLUMNS

IDENTIFIER_COLUMNS = ("UDI", "Product ID")
SENSOR_FEATURE_COLUMNS = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)


@dataclass(frozen=True)
class PreparedDistribution:
    """Data prepared from the currently loaded dataframe for one EDA chart."""

    kind: str
    data: pd.DataFrame


def distribution_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return meaningful, non-leaking variables with AI4I sensors first."""
    excluded = set(IDENTIFIER_COLUMNS).union(LEAKAGE_COLUMNS)
    preferred = [column for column in SENSOR_FEATURE_COLUMNS if column in df.columns]
    remaining = [
        column
        for column in df.columns
        if column not in excluded and column not in preferred
        and (pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_object_dtype(df[column])
             or pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]))
    ]
    return preferred + remaining


def prepare_distribution_data(df: pd.DataFrame, column: str) -> PreparedDistribution:
    """Prepare numeric observations or categorical counts without inventing values."""
    if column not in df.columns:
        raise KeyError(f"'{column}' is not available in the loaded dataset.")

    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return PreparedDistribution("numeric", pd.DataFrame({column: series.dropna()}))

    values = series.astype("string").fillna("Missing")
    counts = values.value_counts(dropna=False).rename_axis(column).reset_index(name="Count")
    return PreparedDistribution("categorical", counts)


def distribution_figure(df: pd.DataFrame, column: str):
    prepared = prepare_distribution_data(df, column)
    if prepared.kind == "numeric":
        return px.histogram(
            prepared.data,
            x=column,
            nbins=40,
            marginal="box",
            title=f"Distribution of {column}",
            labels={column: column},
        )
    return px.bar(prepared.data, x=column, y="Count", title=f"Category counts for {column}", text_auto=True)


def failure_class_statistics(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return observed class counts and percentages for the Machine failure label."""
    if "Machine failure" not in df.columns or df.empty:
        return None

    failure = pd.to_numeric(df["Machine failure"], errors="coerce")
    labels = failure.map({0: "No failure", 1: "Failure"}).fillna("Missing/other")
    order = ["No failure", "Failure", "Missing/other"]
    counts = labels.value_counts().reindex(order, fill_value=0)
    counts = counts[counts > 0].rename_axis("Machine failure").reset_index(name="Count")
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)
    return counts


def correlation_figure(df: pd.DataFrame):
    correlation = prepare_correlation_data(df)
    if correlation is None:
        return None
    return px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature correlations — outcome labels are explicitly marked",
    ).update_xaxes(tickangle=-35)


def prepare_correlation_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """Prepare a correlation matrix, labelling outcome-derived columns explicitly."""
    numeric = df.select_dtypes(include="number").drop(columns=list(IDENTIFIER_COLUMNS), errors="ignore")
    if numeric.shape[1] < 2:
        return None
    correlation = numeric.corr(numeric_only=True)
    labels = {
        "Machine failure": "Machine failure (target)",
        **{column: f"{column} (failure subtype label)" for column in FAILURE_TYPE_COLUMNS},
    }
    return correlation.rename(index=labels, columns=labels)


def failure_count_figure(df: pd.DataFrame):
    counts = failure_class_statistics(df)
    if counts is None:
        return None
    return px.bar(
        counts,
        x="Machine failure",
        y="Count",
        title="Machine failure class balance",
        color="Machine failure",
        text=counts["Percentage"].map(lambda value: f"{value:.2f}%"),
    )


def failure_type_figure(df: pd.DataFrame):
    columns = [column for column in FAILURE_TYPE_COLUMNS if column in df]
    if not columns:
        return None
    counts = df[columns].apply(pd.to_numeric, errors="coerce").sum().rename_axis("Failure type").reset_index(name="Count")
    return px.bar(counts, x="Failure type", y="Count", title="Failure-mode counts", color="Failure type")
