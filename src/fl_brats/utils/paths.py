"""Safe, user-space path operations."""

import os
from pathlib import Path

_SYSTEM_PREFIXES = (Path("/usr"), Path("/opt"), Path("/root"), Path("/etc"), Path("/var"))


def resolve_path(path: str | Path) -> Path:
    """Expand variables and home markers, returning an absolute path."""
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def is_user_owned_or_writable(path: str | Path) -> bool:
    """Return whether a path or nearest existing parent is writable by this user."""
    candidate = resolve_path(path)
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return os.access(candidate, os.W_OK)


def _reject_system_path(path: Path) -> None:
    if any(path == prefix or prefix in path.parents for prefix in _SYSTEM_PREFIXES):
        raise PermissionError(f"Refusing to write to system path: {path}")


def ensure_directory(path: str | Path) -> Path:
    """Create a safe directory when it is user-writable."""
    target = resolve_path(path)
    _reject_system_path(target)
    if not is_user_owned_or_writable(target):
        raise PermissionError(f"Directory is not writable: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_writable_directory(path: str | Path) -> Path:
    """Create and verify a writable directory."""
    target = ensure_directory(path)
    probe = target / ".fl_brats_write_probe"
    try:
        probe.touch(exist_ok=True)
    finally:
        probe.unlink(missing_ok=True)
    return target


def ensure_readable_path(path: str | Path) -> Path:
    """Ensure a path exists and can be read."""
    target = resolve_path(path)
    if not target.exists() or not os.access(target, os.R_OK):
        raise FileNotFoundError(f"Path is missing or unreadable: {target}")
    return target


def create_run_directory(output_root: str | Path, run_name: str) -> Path:
    """Create a named run directory below a safe output root."""
    if Path(run_name).name != run_name:
        raise ValueError("run_name must not contain path separators")
    return ensure_directory(resolve_path(output_root) / run_name)


def create_standard_directories(config: object) -> dict[str, Path]:
    """Create infrastructure roots from an object exposing standard path fields."""
    names = ("data_root", "output_root", "log_root", "cache_root", "checkpoint_root")
    return {name: ensure_writable_directory(getattr(config, name)) for name in names}
