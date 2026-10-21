"""SQLite persistence for serializable experiment metadata and real outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd

from core.ml_engine import ExperimentConfig, ExperimentResult


class StorageError(RuntimeError):
    """Raised for safe, user-facing storage failures."""


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    created_at: str
    status: str
    task_type: str
    algorithm: str
    dataset_name: str
    dataset_fingerprint: str | None
    target: str | None
    features: list[str]
    test_size: float | None
    validation: str
    hyperparameters: dict[str, Any]
    preprocessing: dict[str, Any]
    random_state: int | None
    metrics: dict[str, float]
    runtime_seconds: float | None
    error_message: str | None

    def summary(self) -> dict[str, Any]:
        return {
            "Experiment ID": self.experiment_id,
            "Timestamp": self.created_at,
            "Status": self.status,
            "Task": self.task_type,
            "Algorithm": self.algorithm,
            "Dataset": self.dataset_name,
            "Target": self.target or "—",
            "Validation": self.validation,
            "Split": self.test_size,
            "Runtime (s)": self.runtime_seconds,
            **self.metrics,
            "Error": self.error_message,
        }


@dataclass(frozen=True)
class Comparison:
    records: list[ExperimentRecord]
    compatible: bool
    warnings: list[str]


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Stable fingerprint of dataframe values, columns, dtypes, and row order."""
    digest = sha256()
    digest.update("\x1f".join(map(str, df.columns)).encode("utf-8"))
    digest.update("\x1f".join(map(str, df.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()


class ExperimentStorage:
    """A small transactional repository; connections are short-lived and restart-safe."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        """Yield a connection and always close it (important for Windows file locks)."""
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        try:
            with self._connection() as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiments (
                        experiment_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('completed', 'failed')),
                        task_type TEXT NOT NULL,
                        algorithm TEXT NOT NULL,
                        dataset_name TEXT NOT NULL,
                        dataset_fingerprint TEXT,
                        target TEXT,
                        features_json TEXT NOT NULL,
                        test_size REAL,
                        validation TEXT NOT NULL,
                        hyperparameters_json TEXT NOT NULL,
                        preprocessing_json TEXT NOT NULL,
                        random_state INTEGER,
                        metrics_json TEXT NOT NULL,
                        runtime_seconds REAL,
                        error_message TEXT
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at DESC)")
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to initialize experiment database: {exc}") from exc

    def save_result(self, result: ExperimentResult, dataset_name: str, dataset: pd.DataFrame, experiment_id: str | None = None) -> ExperimentRecord:
        if not isinstance(result.metrics, dict):
            raise StorageError("Completed experiment metrics must be a dictionary.")
        return self._save(
            config=result.config,
            dataset_name=dataset_name,
            dataset_fingerprint=dataframe_fingerprint(dataset),
            metrics={key: float(value) for key, value in result.metrics.items()},
            runtime_seconds=result.runtime_seconds,
            status="completed",
            error_message=None,
            experiment_id=experiment_id,
        )

    def save_failure(self, config: ExperimentConfig, dataset_name: str, dataset: pd.DataFrame | None, error_message: str, experiment_id: str | None = None) -> ExperimentRecord:
        if not error_message.strip():
            raise StorageError("A failed experiment requires a useful error message.")
        return self._save(
            config=config,
            dataset_name=dataset_name,
            dataset_fingerprint=dataframe_fingerprint(dataset) if dataset is not None else None,
            metrics={}, runtime_seconds=None, status="failed", error_message=error_message.strip(), experiment_id=experiment_id,
        )

    def _save(self, *, config: ExperimentConfig, dataset_name: str, dataset_fingerprint: str | None, metrics: dict[str, float], runtime_seconds: float | None, status: str, error_message: str | None, experiment_id: str | None) -> ExperimentRecord:
        if status not in {"completed", "failed"}:
            raise StorageError("Experiment status must be completed or failed.")
        if not dataset_name.strip():
            raise StorageError("Dataset name is required for experiment storage.")
        identifier = experiment_id or str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        preprocessing = {
            "numeric_imputation": config.numeric_imputation,
            "categorical_imputation": config.categorical_imputation,
            "scale_numeric": config.scale_numeric,
            "encode_categorical": config.encode_categorical,
            "class_weight": config.class_weight,
            "resampling": config.resampling,
            "pca_components": config.pca_components,
        }
        record = ExperimentRecord(
            identifier, timestamp, status, config.task_type, config.algorithm, dataset_name, dataset_fingerprint,
            config.target, list(config.features or ()), config.test_size, config.validation, dict(config.hyperparameters),
            preprocessing, config.random_state, metrics, runtime_seconds, error_message,
        )
        values = (
            record.experiment_id, record.created_at, record.status, record.task_type, record.algorithm,
            record.dataset_name, record.dataset_fingerprint, record.target, self._json(record.features), record.test_size,
            record.validation, self._json(record.hyperparameters), self._json(record.preprocessing), record.random_state,
            self._json(record.metrics), record.runtime_seconds, record.error_message,
        )
        try:
            with self._connection() as connection, connection:
                connection.execute(
                    """INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", values
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"Experiment ID '{identifier}' already exists.") from exc
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise StorageError(f"Unable to save experiment: {exc}") from exc
        return record

    def list_records(self) -> list[ExperimentRecord]:
        try:
            with self._connection() as connection:
                rows = connection.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to load experiment history: {exc}") from exc
        return [self._row_to_record(row) for row in rows]

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        try:
            with self._connection() as connection:
                row = connection.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to load experiment: {exc}") from exc
        return self._row_to_record(row) if row else None

    def delete(self, experiment_id: str) -> bool:
        try:
            with self._connection() as connection, connection:
                cursor = connection.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise StorageError(f"Unable to delete experiment: {exc}") from exc

    def export_json(self, records: Iterable[ExperimentRecord] | None = None) -> str:
        rows = list(records) if records is not None else self.list_records()
        return json.dumps([record.__dict__ for record in rows], indent=2, ensure_ascii=False, allow_nan=False)

    def export_csv(self, records: Iterable[ExperimentRecord] | None = None) -> str:
        rows = list(records) if records is not None else self.list_records()
        output = StringIO()
        summaries = [record.summary() for record in rows]
        fields = sorted({key for row in summaries for key in row})
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
        return output.getvalue()

    def compare(self, experiment_ids: Iterable[str]) -> Comparison:
        records = [record for identifier in experiment_ids if (record := self.get(identifier)) is not None]
        warnings: list[str] = []
        if len(records) < 2:
            warnings.append("Select at least two saved experiments to compare.")
        if records:
            baseline = records[0]
            checks = {
                "task type": {record.task_type for record in records},
                "dataset fingerprint": {record.dataset_fingerprint for record in records},
                "target": {record.target for record in records},
                "validation": {record.validation for record in records},
                "train/test split": {record.test_size for record in records},
                "metric set": {tuple(sorted(record.metrics)) for record in records},
            }
            for label, values in checks.items():
                if len(values) > 1:
                    warnings.append(f"Different {label} values make direct metric comparison unreliable.")
            if any(record.status != "completed" for record in records):
                warnings.append("Failed experiments do not have metrics and cannot be compared with completed runs.")
        return Comparison(records, compatible=len(records) >= 2 and not warnings, warnings=warnings)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)

    @staticmethod
    def _loads(value: str, expected: type) -> Any:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, expected) else expected()
        except (json.JSONDecodeError, TypeError):
            return expected()

    def _row_to_record(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            row["experiment_id"], row["created_at"], row["status"], row["task_type"], row["algorithm"],
            row["dataset_name"], row["dataset_fingerprint"], row["target"], self._loads(row["features_json"], list),
            row["test_size"], row["validation"], self._loads(row["hyperparameters_json"], dict),
            self._loads(row["preprocessing_json"], dict), row["random_state"], self._loads(row["metrics_json"], dict),
            row["runtime_seconds"], row["error_message"],
        )
