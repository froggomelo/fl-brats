from pathlib import Path

from fl_brats.utils.environment import collect_environment, write_environment_report
from fl_brats.utils.git import get_git_branch


def test_environment_serialization(tmp_path: Path) -> None:
    report = collect_environment()
    for format in ("json", "yaml", "text"):
        assert write_environment_report(report, tmp_path / f"report.{format}", format).exists()


def test_git_fallback(tmp_path: Path) -> None:
    assert get_git_branch(tmp_path) is None
