from pathlib import Path

from fl_brats.tracking.local import LocalTracker


def test_local_tracking(tmp_path: Path) -> None:
    tracker = LocalTracker(tmp_path)
    tracker.start_run("one")
    tracker.log_parameters({"seed": 1})
    tracker.log_text("ok", "note.txt")
    tracker.set_tags({"scope": "test"})
    tracker.end_run()
    assert (tmp_path / "one" / "parameters.json").exists()
