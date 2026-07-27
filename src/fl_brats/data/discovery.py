"""Recursive, read-only discovery of BraTS subject directories."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_SUFFIXES = {
    "t1": "_t1.nii.gz",
    "t1ce": "_t1ce.nii.gz",
    "t2": "_t2.nii.gz",
    "flair": "_flair.nii.gz",
    "seg": "_seg.nii.gz",
}


@dataclass
class SubjectRecord:
    subject_id: str
    source_collection: str
    subject_directory: Path
    t1_path: Path | None = None
    t1ce_path: Path | None = None
    t2_path: Path | None = None
    flair_path: Path | None = None
    segmentation_path: Path | None = None
    complete: bool = False
    missing_files: list[str] = field(default_factory=list)
    duplicate_files: dict[str, list[str]] = field(default_factory=dict)
    validation_status: str = "unvalidated"

    @property
    def paths(self) -> dict[str, Path | None]:
        return {
            "t1": self.t1_path,
            "t1ce": self.t1ce_path,
            "t2": self.t2_path,
            "flair": self.flair_path,
            "seg": self.segmentation_path,
        }


def _role_candidates(directory: Path, subject_id: str, role: str) -> list[Path]:
    """Find canonical and modality-like NIfTIs, retaining ambiguity for validation."""
    canonical = f"{subject_id}_{role}.nii.gz"
    prefix = f"{subject_id}_{role}".casefold()
    return sorted(
        p
        for p in directory.glob("*.nii.gz")
        if p.name.casefold() == canonical.casefold()
        or (
            p.name.casefold().startswith(prefix)
            and p.name.casefold() != f"{subject_id}_t1ce.nii.gz".casefold()
            if role == "t1"
            else p.name.casefold().startswith(prefix)
        )
    )


def _candidate_ids(directory: Path) -> set[str]:
    result: set[str] = set()
    for path in directory.glob("*.nii.gz"):
        lower = path.name.casefold()
        for suffix in REQUIRED_SUFFIXES.values():
            if lower.endswith(suffix):
                result.add(path.name[: -len(suffix)])
    return result


def discover_records(root: Path, follow_symlinks: bool = False) -> list[SubjectRecord]:
    root = root.resolve()
    directories = sorted(
        p
        for p in root.rglob("*")
        if p.is_dir() and (follow_symlinks or not p.is_symlink()) and _candidate_ids(p)
    )
    records: list[SubjectRecord] = []
    for directory in directories:
        for subject_id in sorted(_candidate_ids(directory)):
            values: dict[str, Path | None] = {}
            ambiguous: dict[str, list[str]] = {}
            for role, suffix in REQUIRED_SUFFIXES.items():
                canonical = directory / f"{subject_id}{suffix}"
                files = _role_candidates(directory, subject_id, role)
                values[role] = canonical if canonical.exists() and len(files) == 1 else None
                if len(files) > 1 or (files and not canonical.exists()):
                    ambiguous[role] = [str(p) for p in files]
            missing = [
                role for role, path in values.items() if path is None and role not in ambiguous
            ]
            records.append(
                SubjectRecord(
                    subject_id,
                    directory.parent.name,
                    directory,
                    values["t1"],
                    values["t1ce"],
                    values["t2"],
                    values["flair"],
                    values["seg"],
                    not missing and not ambiguous,
                    missing,
                    ambiguous,
                )
            )
    return records


def discover_subjects(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for record in discover_records(root):
        result[record.subject_id].append(record.subject_directory)
    return dict(result)


def subject_files(subject_id: str, directory: Path) -> dict[str, Path]:
    return {role: directory / f"{subject_id}{suffix}" for role, suffix in REQUIRED_SUFFIXES.items()}
