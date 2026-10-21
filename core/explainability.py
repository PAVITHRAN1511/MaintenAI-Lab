"""Honest feature-importance and optional SHAP helpers."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.ml_engine import ExperimentResult


@dataclass(frozen=True)
class ExplanationStatus:
    available: bool
    reason: str


def global_feature_importance(result: ExperimentResult) -> tuple[pd.DataFrame | None, ExplanationStatus]:
    if result.config.task_type not in {"classification", "regression"}:
        return None, ExplanationStatus(False, "Global supervised-model importance is unavailable for this task.")
    try:
        preprocessor = result.model.named_steps["preprocessor"]
        estimator = result.model.named_steps["model"]
        names = preprocessor.get_feature_names_out()
    except (AttributeError, KeyError):
        return None, ExplanationStatus(False, "The fitted pipeline does not expose transformed feature names.")
    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        values = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    if values is None:
        return None, ExplanationStatus(False, f"{type(estimator).__name__} does not expose native global feature importance.")
    importance = pd.DataFrame({"Feature": names, "Importance": np.asarray(values, dtype=float)}).sort_values("Importance", ascending=False, ignore_index=True)
    return importance, ExplanationStatus(True, "Importance is calculated from the fitted estimator.")


def shap_status(result: ExperimentResult) -> ExplanationStatus:
    if result.config.task_type not in {"classification", "regression"} or result.test_features is None:
        return ExplanationStatus(False, "SHAP requires a completed supervised run with held-out feature data.")
    if importlib.util.find_spec("shap") is None:
        return ExplanationStatus(False, "SHAP is not installed in this environment.")
    try:
        estimator = result.model.named_steps["model"]
    except (AttributeError, KeyError):
        return ExplanationStatus(False, "The fitted pipeline does not expose an estimator for SHAP.")
    if not (hasattr(estimator, "feature_importances_") or hasattr(estimator, "coef_")):
        return ExplanationStatus(False, f"SHAP is not configured for {type(estimator).__name__}.")
    return ExplanationStatus(True, "SHAP can calculate held-out global importance for this fitted estimator.")


def shap_global_importance(result: ExperimentResult, max_rows: int = 200) -> tuple[pd.DataFrame | None, ExplanationStatus]:
    """Return mean absolute SHAP values only when SHAP and a compatible model are available."""
    status = shap_status(result)
    if not status.available:
        return None, status
    try:
        import shap  # type: ignore[import-not-found]

        preprocessor = result.model.named_steps["preprocessor"]
        estimator = result.model.named_steps["model"]
        transformed = preprocessor.transform(result.test_features.iloc[:max_rows])
        names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(estimator) if hasattr(estimator, "feature_importances_") else shap.LinearExplainer(estimator, transformed)
        values = explainer.shap_values(transformed)
        values = values[1] if isinstance(values, list) and len(values) > 1 else values
        importance = np.abs(np.asarray(values)).mean(axis=0)
        table = pd.DataFrame({"Feature": names, "Mean |SHAP value|": importance}).sort_values("Mean |SHAP value|", ascending=False, ignore_index=True)
    except Exception as exc:  # SHAP support varies by estimator and installed version
        return None, ExplanationStatus(False, f"SHAP could not explain this fitted model: {exc}")
    return table, ExplanationStatus(True, "Mean absolute SHAP values were calculated from held-out rows.")
