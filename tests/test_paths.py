from pathlib import Path

from fl_brats.utils.paths import create_run_directory, ensure_writable_directory, resolve_path


def test_paths(tmp_path: Path) -> None:
    assert ensure_writable_directory(tmp_path / "new").is_dir()
    assert create_run_directory(tmp_path, "run").is_dir()
    assert resolve_path(tmp_path).is_absolute()
