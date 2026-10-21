import sqlite3

import pandas as pd
import pytest

from core.experiment_storage import ExperimentStorage, StorageError, dataframe_fingerprint
from core.ml_engine import ExperimentConfig, run_supervised_experiment
from tests.test_ml_engine import supervised_frame


def completed_result():
    frame = supervised_frame()
    config = ExperimentConfig(task_type="classification", algorithm="decision_tree", target="Machine failure", features=("Torque [Nm]", "Rotational speed [rpm]", "Type"), random_state=17)
    return frame, config, run_supervised_experiment(frame, config)


def test_storage_persists_across_connections_and_crud(tmp_path) -> None:
    path = tmp_path / "experiments.db"
    frame, _, result = completed_result()
    saved = ExperimentStorage(path).save_result(result, "synthetic-ai4i.csv", frame, experiment_id="run-1")
    reopened = ExperimentStorage(path)
    loaded = reopened.get("run-1")
    assert saved.experiment_id == "run-1"
    assert loaded is not None and loaded.metrics == result.metrics
    assert loaded.dataset_fingerprint == dataframe_fingerprint(frame)
    assert reopened.delete("run-1")
    assert reopened.get("run-1") is None


def test_storage_exports_and_comparison_flags_incompatible_runs(tmp_path) -> None:
    storage = ExperimentStorage(tmp_path / "experiments.db")
    frame, config, result = completed_result()
    first = storage.save_result(result, "dataset.csv", frame, experiment_id="first")
    second_config = ExperimentConfig(**{**config.__dict__, "validation": "kfold"})
    second = storage.save_failure(second_config, "dataset.csv", frame, "Cross-validation setup failed", experiment_id="second")
    assert "first" in storage.export_csv()
    assert '"experiment_id": "first"' in storage.export_json()
    comparison = storage.compare([first.experiment_id, second.experiment_id])
    assert not comparison.compatible
    assert any("validation" in warning for warning in comparison.warnings)
    assert any("Failed" in warning for warning in comparison.warnings)


def test_storage_handles_duplicate_ids_malformed_json_and_invalid_failure(tmp_path) -> None:
    path = tmp_path / "experiments.db"
    storage = ExperimentStorage(path)
    frame, config, result = completed_result()
    storage.save_result(result, "dataset.csv", frame, experiment_id="duplicate")
    with pytest.raises(StorageError, match="already exists"):
        storage.save_result(result, "dataset.csv", frame, experiment_id="duplicate")
    with pytest.raises(StorageError, match="useful error"):
        storage.save_failure(config, "dataset.csv", frame, "")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE experiments SET metrics_json = ? WHERE experiment_id = ?", ("not-json", "duplicate"))
    loaded = storage.get("duplicate")
    assert loaded is not None and loaded.metrics == {}
