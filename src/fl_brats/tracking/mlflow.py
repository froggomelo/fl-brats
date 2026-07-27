"""MLflow metadata-only tracker."""

import os
from pathlib import Path


class MLflowTracker:
    """Thin MLflow wrapper which never starts an MLflow server."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        import mlflow

        # MLflow 3 requires an explicit opt-in for its local file store.
        # This only enables the configured local backend; no server is started.
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> None:
        self.mlflow.start_run(run_name=run_name)

    def log_parameters(self, parameters: dict[str, object]) -> None:
        self.mlflow.log_params({key: str(value) for key, value in parameters.items()})

    def log_text(self, text: str, name: str) -> None:
        self.mlflow.log_text(text, name)

    def log_artifact(self, path: Path) -> None:
        self.mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.mlflow.set_tags(tags)

    def end_run(self) -> None:
        self.mlflow.end_run()
