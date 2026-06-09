"""MLflow integration: experiment tracking + model registry.

Wraps MLflow calls so model lifecycle is consistent across:
    - Training pipeline (logs run + registers model)
    - API serving (loads latest Production model)
    - Monitoring pipeline (compares models)

Reference: https://mlflow.org/docs/latest/model-registry.html
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import mlflow
import mlflow.spark
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelInfo:
    """Snapshot of a registered model version."""

    name: str
    version: str
    stage: str
    run_id: str
    uri: str
    creation_time: int

    @classmethod
    def from_mlflow(cls, mv: ModelVersion) -> ModelInfo:
        return cls(
            name=mv.name,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id,
            uri=f"models:/{mv.name}/{mv.version}",
            creation_time=mv.creation_timestamp,
        )


class MLflowRegistry:
    """High-level wrapper around MLflow tracking + registry."""

    def __init__(self, tracking_uri: str | None = None, experiment_name: str | None = None):
        cfg = get_config().get("mlflow", {})
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", cfg.get("tracking_uri", "http://mlflow:5000")
        )
        self.experiment_name = experiment_name or cfg.get("experiment_name", "credit_risk_platform")

        mlflow.set_tracking_uri(self.tracking_uri)
        self._ensure_experiment()
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        log.info("mlflow_registry_initialized", uri=self.tracking_uri, exp=self.experiment_name)

    def _ensure_experiment(self) -> None:
        """Create experiment if not exists."""
        try:
            mlflow.set_experiment(self.experiment_name)
        except MlflowException as e:
            log.warning("mlflow_experiment_creation_failed", error=str(e))

    # ──────────────────────────────────────────────────────────
    # Run lifecycle
    # ──────────────────────────────────────────────────────────

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Iterator[mlflow.ActiveRun]:
        """Context manager for an MLflow run."""
        all_tags = {"project": "credit-risk-platform", **(tags or {})}
        with mlflow.start_run(run_name=run_name, tags=all_tags, nested=nested) as run:
            log.info("mlflow_run_started", run_id=run.info.run_id, name=run_name)
            yield run
            log.info("mlflow_run_ended", run_id=run.info.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters. MLflow truncates string params at 500 chars."""
        clean = {k: _safe_param_value(v) for k, v in params.items()}
        mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics."""
        for key, value in metrics.items():
            if value is not None and not _is_nan(value):
                mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict_as_json(self, data: dict, artifact_file: str) -> None:
        mlflow.log_dict(data, artifact_file)

    # ──────────────────────────────────────────────────────────
    # Model registration
    # ──────────────────────────────────────────────────────────

    def log_spark_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> str:
        """Log a Spark PipelineModel and register it.

        Returns the model URI suitable for `mlflow.spark.load_model`.
        """
        name = registered_model_name or get_config()["mlflow"].get(
            "registered_model_name", "credit_risk_gbt"
        )
        mlflow.spark.log_model(
            spark_model=model,
            artifact_path=artifact_path,
            registered_model_name=name,
        )
        run = mlflow.active_run()
        uri = f"runs:/{run.info.run_id}/{artifact_path}" if run else artifact_path
        log.info("spark_model_logged", uri=uri, registered_as=name)
        return uri

    def promote_model(
        self,
        name: str,
        version: str,
        stage: str = "Production",
        archive_existing: bool = True,
    ) -> None:
        """Promote a model version to a stage (Staging / Production / Archived)."""
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        log.info("model_promoted", name=name, version=version, stage=stage)

    def get_latest_version(self, name: str, stage: str = "Production") -> ModelInfo | None:
        """Return latest model version in given stage, or None if not found."""
        try:
            versions = self.client.get_latest_versions(name, stages=[stage])
            if not versions:
                return None
            return ModelInfo.from_mlflow(versions[0])
        except MlflowException as e:
            log.warning("get_latest_version_failed", name=name, stage=stage, error=str(e))
            return None

    def load_production_model(self, name: str | None = None) -> Any:
        """Load the Production-stage model. Used by the API at startup."""
        name = name or get_config()["mlflow"]["registered_model_name"]
        info = self.get_latest_version(name, "Production")
        if info is None:
            log.warning("no_production_model", name=name, fallback="Staging")
            info = self.get_latest_version(name, "Staging")
        if info is None:
            raise RuntimeError(f"No registered model found for '{name}' in Production or Staging")

        log.info("loading_model", uri=info.uri, version=info.version, stage=info.stage)
        return mlflow.spark.load_model(info.uri), info

    def list_versions(self, name: str) -> list[ModelInfo]:
        """List all versions of a registered model."""
        versions = self.client.search_model_versions(f"name='{name}'")
        return [ModelInfo.from_mlflow(v) for v in versions]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _safe_param_value(value: Any) -> Any:
    """Coerce values to MLflow-loggable types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_nan(value: Any) -> bool:
    try:
        return value != value  # NaN check
    except TypeError:
        return False
