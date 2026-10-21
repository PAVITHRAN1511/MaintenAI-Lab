import numpy as np
import pandas as pd
import pytest

from core.ml_engine import (
    ConfigurationError,
    ExperimentConfig,
    default_feature_columns,
    run_clustering_experiment,
    run_pca_experiment,
    run_supervised_experiment,
)


def supervised_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.RandomState(12)
    torque = rng.normal(45, 6, rows)
    speed = rng.normal(1500, 120, rows)
    failure = ((torque > 47) & (speed < 1510)).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(rows),
            "Product ID": [f"M{i}" for i in range(rows)],
            "Type": np.where(np.arange(rows) % 2, "M", "L"),
            "Torque [Nm]": torque,
            "Rotational speed [rpm]": speed,
            "Tool wear [min]": rng.randint(0, 200, rows),
            "Machine failure": failure,
            "TWF": failure,
            "continuous_target": 0.6 * torque + 0.01 * speed + rng.normal(0, 0.1, rows),
        }
    )


def test_classification_has_real_metrics_and_is_reproducible() -> None:
    frame = supervised_frame()
    config = ExperimentConfig(
        task_type="classification", algorithm="logistic_regression", target="Machine failure",
        validation="stratified_kfold", cv_folds=3, random_state=7,
    )
    first = run_supervised_experiment(frame, config)
    second = run_supervised_experiment(frame, config)
    assert {"accuracy", "f1_macro", "roc_auc"}.issubset(first.metrics)
    assert first.validation_scores == second.validation_scores
    assert np.array_equal(first.predictions, second.predictions)
    assert "TWF" not in first.feature_columns


def test_regression_has_real_metrics() -> None:
    result = run_supervised_experiment(
        supervised_frame(),
        ExperimentConfig(
            task_type="regression", algorithm="ridge", target="continuous_target",
            features=("Torque [Nm]", "Rotational speed [rpm]", "Type"), validation="kfold", cv_folds=3,
        ),
    )
    assert {"mae", "rmse", "r2"}.issubset(result.metrics)
    assert len(result.predictions) == len(result.truth)


def test_clustering_produces_labels_and_metrics() -> None:
    rng = np.random.RandomState(4)
    frame = pd.DataFrame({"x": np.r_[rng.normal(-3, .2, 20), rng.normal(3, .2, 20)], "y": np.r_[rng.normal(-3, .2, 20), rng.normal(3, .2, 20)]})
    result = run_clustering_experiment(frame, ExperimentConfig(task_type="clustering", algorithm="kmeans", hyperparameters={"n_clusters": 2}))
    assert len(result.labels) == len(frame)
    assert result.metrics["clusters"] == 2
    assert "silhouette" in result.metrics


def test_pca_returns_variance_analysis() -> None:
    frame = supervised_frame()[["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"]]
    result = run_pca_experiment(frame, ExperimentConfig(task_type="pca", algorithm="pca", pca_components=2))
    assert result.transformed_data.shape == (len(frame), 2)
    assert len(result.explained_variance_ratio) == 2
    assert 0 < result.metrics["explained_variance_total"] <= 1


def test_machine_failure_defaults_and_manual_leakage_are_rejected() -> None:
    frame = supervised_frame()
    defaults = default_feature_columns(frame, "Machine failure")
    assert "UDI" not in defaults and "Product ID" not in defaults and "TWF" not in defaults
    leaking = ExperimentConfig(task_type="classification", algorithm="decision_tree", target="Machine failure", features=("Torque [Nm]", "TWF"))
    with pytest.raises(ConfigurationError, match="leakage"):
        run_supervised_experiment(frame, leaking)


def test_failure_subtype_target_also_excludes_outcome_labels() -> None:
    frame = supervised_frame()
    defaults = default_feature_columns(frame, "TWF")
    assert "Machine failure" not in defaults and "TWF" not in defaults
    config = ExperimentConfig(task_type="classification", algorithm="decision_tree", target="TWF", features=("Machine failure", "Torque [Nm]"))
    with pytest.raises(ConfigurationError, match="leakage"):
        run_supervised_experiment(frame, config)


def test_non_failure_targets_also_reject_outcome_labels_as_predictors() -> None:
    frame = supervised_frame()
    defaults = default_feature_columns(frame, "continuous_target")
    assert "Machine failure" not in defaults and "TWF" not in defaults
    config = ExperimentConfig(
        task_type="regression", algorithm="ridge", target="continuous_target",
        features=("Torque [Nm]", "Machine failure"),
    )
    with pytest.raises(ConfigurationError, match="leakage"):
        run_supervised_experiment(frame, config)
