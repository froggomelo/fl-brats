from typer.testing import CliRunner

from fl_brats.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0 and result.stdout.strip() == "0.1.0"


def test_show_config() -> None:
    result = CliRunner().invoke(app, ["show-config"])
    assert result.exit_code == 0 and "project_name: fl-brats" in result.stdout
