from pathlib import Path

from typer.testing import CliRunner

from tests.data.conftest import make_subject

from fl_brats.cli import app
from fl_brats.data.indexing import build_dataset_index


def test_data_commands_are_registered() -> None:
    assert "preprocess-data" in CliRunner().invoke(app, ["--help"]).stdout


def test_preprocess_data_accepts_hydra_style_overrides(raw_root: Path, tmp_path: Path) -> None:
    make_subject(raw_root)
    processed = tmp_path / "processed"
    index_path, _, _ = build_dataset_index(raw_root, processed)

    result = CliRunner().invoke(
        app,
        [
            "preprocess-data",
            "--index-path",
            str(index_path),
            "--raw-root",
            str(raw_root),
            "--output-dir",
            str(processed),
            "preprocessing=brats2021_v1",
            "preprocessing.materialize=false",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert '"mode": "manifest-only"' in result.stdout
