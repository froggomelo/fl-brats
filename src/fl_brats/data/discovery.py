"""Discovery of immutable BraTS 2021 subject directories."""

from collections import defaultdict
from pathlib import Path

REQUIRED_SUFFIXES = {"t1": "_t1.nii.gz", "t1ce": "_t1ce.nii.gz", "t2": "_t2.nii.gz", "flair": "_flair.nii.gz", "seg": "_seg.nii.gz"}


def discover_subjects(root: Path) -> dict[str, list[Path]]:
    """Return every candidate subject ID and its directories; retain duplicates."""
    found: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*_seg.nii.gz"):
        subject_id = path.name[: -len(REQUIRED_SUFFIXES["seg"])]
        found[subject_id].append(path.parent)
    return dict(found)


def subject_files(subject_id: str, directory: Path) -> dict[str, Path]:
    """Return canonical expected paths without changing raw data."""
    return {name: directory / f"{subject_id}{suffix}" for name, suffix in REQUIRED_SUFFIXES.items()}
