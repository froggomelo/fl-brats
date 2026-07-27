from pathlib import Path

import pytest


def test_mlflow_file_tracking(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    from fl_brats.tracking.mlflow import MLflowTracker

    tracker = MLflowTracker(f"file:{tmp_path / 'mlruns'}", "test-infrastructure")
    tracker.start_run("metadata-only")
    tracker.log_parameters({"seed": 1})
    tracker.end_run()
