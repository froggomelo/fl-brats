"""Local filesystem tracker."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from fl_brats.utils.paths import ensure_directory


class LocalTracker:
    """Write metadata and artifacts to a run directory owned by the user."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = ensure_directory(root_dir)
        self.run_dir: Path | None = None

    def start_run(self, run_name: str) -> None:
        self.run_dir = ensure_directory(self.root_dir / run_name)
        (self.run_dir / "metadata.json").write_text(
            json.dumps({"run_name": run_name}, indent=2), encoding="utf-8"
        )

    def _run(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("start_run must be called first")
        return self.run_dir

    def _merge(self, name: str, values: Mapping[str, object]) -> None:
        target = self._run() / name
        existing = json.loads(target.read_text()) if target.exists() else {}
        existing.update(values)
        target.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    def log_parameters(self, parameters: dict[str, object]) -> None:
        self._merge("parameters.json", parameters)

    def log_text(self, text: str, name: str) -> None:
        (self._run() / name).write_text(text, encoding="utf-8")

    def log_artifact(self, path: Path) -> None:
        shutil.copy2(path, ensure_directory(self._run() / "artifacts") / path.name)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._merge("tags.json", tags)

    def end_run(self) -> None:
        pass
