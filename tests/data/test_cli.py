from typer.testing import CliRunner

from fl_brats.cli import app


def test_data_commands_are_registered() -> None:
    assert "preprocess-data" in CliRunner().invoke(app, ["--help"]).stdout
