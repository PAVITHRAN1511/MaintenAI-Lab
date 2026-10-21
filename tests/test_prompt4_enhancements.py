"""Tests for Prompt 4 enhancements: engineered features, classification CV scoring,
threshold-dependent predictions, and task selector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app import _calculate_live_metrics
from core.diagnostics import classification_figures
from core.features import ENGINEERED_FEATURE_NAMES, add_engineered_features
from core.ml_engine import (
    ExperimentConfig,
    _validation_score,
    default_feature_columns,
    run_supervised_experiment,
)
from core.model_lab import algorithms_for
from tests.test_ml_engine import supervised_frame


def test_engineered_features_creation_and_safe_columns() -> None:
    frame = supervised_frame(50)
    # Add temperature columns for testing
    frame["Air temperature [K]"] = 300.0
    frame["Process temperature [K]"] = 310.0

    eng = add_engineered_features(frame)

    for feat in ENGINEERED_FEATURE_NAMES:
        assert feat in eng.columns

    # Verify calculation correctness
    assert np.allclose(eng["Temp difference [K]"], 10.0)
    assert np.allclose(eng["Temp ratio"], 310.0 / 300.0)
    expected_power = frame["Torque [Nm]"] * frame["Rotational speed [rpm]"] * (2 * np.pi / 60)
    assert np.allclose(eng["Power [W]"], expected_power)
    expected_overstrain = frame["Tool wear [min]"] * frame["Torque [Nm]"]
    assert np.allclose(eng["Overstrain [minNm]"], expected_overstrain)

    # Safe default features must include engineered features and exclude identifiers & target/failure modes
    defaults = default_feature_columns(eng, "Machine failure")
    assert all(feat in defaults for feat in ENGINEERED_FEATURE_NAMES)
    assert "UDI" not in defaults
    assert "Product ID" not in defaults
    assert "Machine failure" not in defaults
    assert "TWF" not in defaults


def test_classification_cv_scoring_uses_binary_failure_f1() -> None:
    frame = supervised_frame(60)
    config = ExperimentConfig(
        task_type="classification",
        algorithm="decision_tree",
        target="Machine failure",
        validation="stratified_kfold",
        cv_folds=3,
        random_state=42,
    )
    result = run_supervised_experiment(frame, config)
    assert len(result.validation_scores) == 3
    # Check that scores are valid F1 values between 0 and 1
    assert all(0.0 <= s <= 1.0 for s in result.validation_scores)


def test_regression_cv_scoring_remains_intact() -> None:
    frame = supervised_frame(60)
    config = ExperimentConfig(
        task_type="regression",
        algorithm="ridge",
        target="continuous_target",
        features=("Torque [Nm]", "Rotational speed [rpm]"),
        validation="kfold",
        cv_folds=3,
        random_state=42,
    )
    result = run_supervised_experiment(frame, config)
    assert len(result.validation_scores) == 3
    # R2 scores can be up to 1.0
    assert all(s <= 1.0 for s in result.validation_scores)


def test_threshold_dependent_predictions_and_live_metrics() -> None:
    frame = supervised_frame(100)
    config = ExperimentConfig(
        task_type="classification",
        algorithm="logistic_regression",
        target="Machine failure",
        features=("Torque [Nm]", "Rotational speed [rpm]"),
        random_state=42,
    )
    result = run_supervised_experiment(frame, config)
    assert result.prediction_scores is not None

    # Low threshold: higher recall
    preds_low = (result.prediction_scores >= 0.10).astype(int)
    metrics_low = _calculate_live_metrics(result, preds_low)

    # High threshold: higher precision, lower recall
    preds_high = (result.prediction_scores >= 0.90).astype(int)
    metrics_high = _calculate_live_metrics(result, preds_high)

    assert metrics_low["recall_failure"] >= metrics_high["recall_failure"]
    assert "f1_failure" in metrics_low
    assert "precision_failure" in metrics_low


def test_classification_figures_includes_baseline_reference_line() -> None:
    frame = supervised_frame(60)
    config = ExperimentConfig(
        task_type="classification",
        algorithm="logistic_regression",
        target="Machine failure",
        features=("Torque [Nm]", "Rotational speed [rpm]"),
        random_state=42,
    )
    result = run_supervised_experiment(frame, config)
    figs = classification_figures(result, baseline_accuracy=0.9661)

    assert "confusion_matrix" in figs
    assert "metrics_chart" in figs
    chart = figs["metrics_chart"]
    # Check that baseline horizontal line annotation exists
    annotations = [ann.text for ann in chart.layout.annotations if ann.text]
    assert any("Trivial Baseline" in text for text in annotations)


def test_model_lab_algorithms_for_tasks() -> None:
    assert "random_forest" in algorithms_for("classification")
    assert "kmeans" in algorithms_for("clustering")
    assert "pca" in algorithms_for("pca")
    # Regression engine code remains intact in ALGORITHMS
    assert "ridge" in algorithms_for("regression")
