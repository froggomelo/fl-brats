from typer.testing import CliRunner

from fl_brats.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0 and result.stdout.strip() == "0.1.0"


def test_show_config() -> None:
    result = CliRunner().invoke(app, ["show-config"])
    assert result.exit_code == 0 and "project_name: fl-brats" in result.stdout


def test_dataset_index_path_overrides_experiment_index(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from fl_brats.cli import _experiment_values

    config = tmp_path / "experiment.yaml"
    config.write_text("data: {partition_root: data/partitions/fedavg_uneven}\n", encoding="utf-8")
    monkeypatch.setenv("DATASET_INDEX_PATH", "data/index/brats2021_v1/dataset_index.csv")
    values = _experiment_values(config, SimpleNamespace(args=[]))
    assert values["data"]["index_path"] == "data/index/brats2021_v1/dataset_index.csv"
