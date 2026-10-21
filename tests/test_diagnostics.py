import numpy as np
import pandas as pd

from core.diagnostics import classification_figures, cluster_figure, experiment_summary, pca_projection_figure, pca_variance_figure, regression_figures
from core.explainability import global_feature_importance, shap_status
from core.ml_engine import ExperimentConfig, run_clustering_experiment, run_pca_experiment, run_supervised_experiment
from tests.test_ml_engine import supervised_frame


def test_classification_diagnostics_and_importance_use_fitted_result() -> None:
    frame = supervised_frame()
    result = run_supervised_experiment(frame, ExperimentConfig(task_type="classification", algorithm="decision_tree", target="Machine failure", features=("Torque [Nm]", "Rotational speed [rpm]", "Type")))
    figures = classification_figures(result)
    importance, status = global_feature_importance(result)
    assert {"confusion_matrix", "roc", "precision_recall"}.issubset(figures)
    assert importance is not None and status.available
    assert result.runtime_seconds is not None and result.runtime_seconds >= 0
    assert "Validation" in experiment_summary(result)["Setting"].tolist()


def test_supported_pipeline_importance_uses_transformed_feature_names() -> None:
    frame = supervised_frame()
    result = run_supervised_experiment(
        frame,
        ExperimentConfig(
            task_type="classification", algorithm="logistic_regression", target="Machine failure",
            features=("Torque [Nm]", "Rotational speed [rpm]", "Type"),
        ),
    )
    importance, status = global_feature_importance(result)
    assert status.available and importance is not None
    assert len(importance) == len(result.model.named_steps["preprocessor"].get_feature_names_out())
    assert any("Torque [Nm]" in feature for feature in importance["Feature"])
    assert any("Type_" in feature for feature in importance["Feature"])


def test_regression_diagnostics_have_real_figures() -> None:
    frame = supervised_frame()
    result = run_supervised_experiment(frame, ExperimentConfig(task_type="regression", algorithm="ridge", target="continuous_target", features=("Torque [Nm]", "Rotational speed [rpm]")))
    assert {"actual_vs_predicted", "residuals"}.issubset(regression_figures(result))


def test_clustering_and_pca_diagnostics_and_unsupervised_leakage_rejection() -> None:
    frame = pd.DataFrame({"x": np.r_[np.zeros(10), np.ones(10) * 4], "y": np.r_[np.zeros(10), np.ones(10) * 4], "Machine failure": [0] * 10 + [1] * 10})
    cluster = run_clustering_experiment(frame, ExperimentConfig(task_type="clustering", algorithm="kmeans", features=("x", "y"), hyperparameters={"n_clusters": 2}))
    pca = run_pca_experiment(frame, ExperimentConfig(task_type="pca", algorithm="pca", features=("x", "y"), pca_components=2))
    assert cluster_figure(cluster) is not None
    assert pca_variance_figure(pca) is not None
    assert pca_projection_figure(pca) is not None
    from core.ml_engine import ConfigurationError
    import pytest
    with pytest.raises(ConfigurationError, match="leakage"):
        run_clustering_experiment(frame, ExperimentConfig(task_type="clustering", algorithm="kmeans", features=("x", "Machine failure")))


def test_explainability_reports_shap_fallback_without_dependency() -> None:
    frame = supervised_frame()
    result = run_supervised_experiment(frame, ExperimentConfig(task_type="classification", algorithm="knn", target="Machine failure", features=("Torque [Nm]", "Rotational speed [rpm]")))
    assert not shap_status(result).available
