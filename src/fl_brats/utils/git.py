"""Safe Git metadata helpers."""

import subprocess
from pathlib import Path


def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=path, check=False, capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_git_commit(path: Path | None = None) -> str | None:
    """Return the current commit hash, or ``None`` outside a repository."""
    return _git(path or Path.cwd(), "rev-parse", "HEAD")


def get_git_branch(path: Path | None = None) -> str | None:
    """Return the current branch, or ``None`` when unavailable."""
    return _git(path or Path.cwd(), "rev-parse", "--abbrev-ref", "HEAD")


def is_git_dirty(path: Path | None = None) -> bool | None:
    """Return whether tracked state is dirty, or ``None`` when unavailable."""
    result = _git(path or Path.cwd(), "status", "--porcelain")
    return None if result is None else bool(result)


def get_git_root(path: Path | None = None) -> Path | None:
    """Return repository root, or ``None`` outside a Git worktree."""
    root = _git(path or Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(root) if root else None
