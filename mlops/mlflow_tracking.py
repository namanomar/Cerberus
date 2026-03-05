

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import yaml
from loguru import logger
from mlflow.tracking import MlflowClient


class MLflowTracker:
    """Centralised MLflow tracker for the fraud detection system."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        tracking_uri = cfg["mlflow"]["tracking_uri"]
        self.experiment_name = cfg["mlflow"]["experiment_name"]

        # Fall back to local filesystem if remote server is unreachable
        if tracking_uri.startswith("http"):
            try:
                import urllib.request
                urllib.request.urlopen(tracking_uri, timeout=2)
            except Exception:
                logger.warning(
                    f"MLflow server at {tracking_uri} unreachable — "
                    f"using local filesystem tracking (mlruns/)."
                )
                tracking_uri = "mlruns"

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        logger.info(f"MLflow tracking: {self.tracking_uri} | experiment: {self.experiment_name}")

    @contextmanager
    def start_run(self, run_name: str = None, tags: dict = None):
        """Context manager for a tracked run."""
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            logger.info(f"MLflow run started: {run.info.run_id}")
            yield run
            logger.info(f"MLflow run finished: {run.info.run_id}")

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def log_dict(self, data: dict, filename: str) -> None:
        mlflow.log_dict(data, filename)

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: str = "",
    ) -> str:
        """Register a logged model in the MLflow Model Registry."""
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = result.version

        self.client.update_model_version(
            name=model_name,
            version=version,
            description=description,
        )

        logger.success(f"Registered model '{model_name}' version {version}")
        return version

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,  # "Staging" | "Production" | "Archived"
    ) -> None:
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        logger.success(f"Model '{model_name}' v{version} → {stage}")

    def load_production_model(self, model_name: str):
        """Load the current Production model from the registry."""
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri)

    def get_best_run(
        self,
        metric: str = "pr_auc",
        ascending: bool = False,
    ) -> Optional[dict]:
        """Return the run with the best value for `metric`."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if runs.empty:
            return None

        return runs.iloc[0].to_dict()

    def compare_runs(self, n_runs: int = 10) -> pd.DataFrame:
        """Return a DataFrame comparing the last n runs."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=n_runs,
            order_by=["start_time DESC"],
        )

        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        param_cols = [c for c in runs.columns if c.startswith("params.")]

        display_cols = ["run_id", "start_time", "status"] + metric_cols + param_cols
        return runs[[c for c in display_cols if c in runs.columns]]
