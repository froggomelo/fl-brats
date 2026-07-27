"""Accumulating structural and BraTS-label validation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from .discovery import SubjectRecord
from .nifti import NiftiInfo, inspect_nifti, nifti_info_dict


@dataclass
class ValidationResult:
    subject_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: dict[str, dict[str, object]] = field(default_factory=dict)
    segmentation_labels: list[int] = field(default_factory=list)
    unreadable_files: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_subject(
    record: SubjectRecord,
    allow_unexpected_labels: bool = False,
    affine_tolerance: float = 1e-4,
    spacing_tolerance: float = 1e-5,
    duplicate_directories: list[Path] | None = None,
) -> ValidationResult:
    result = ValidationResult(record.subject_id, False)
    if record.missing_files:
        result.errors.append("missing files: " + ", ".join(record.missing_files))
    if record.duplicate_files:
        result.errors.append("ambiguous modality files: " + ", ".join(record.duplicate_files))
    if duplicate_directories and len(duplicate_directories) > 1:
        result.errors.append(
            "duplicate subject ID directories: " + ", ".join(map(str, duplicate_directories))
        )
    infos: dict[str, NiftiInfo] = {}
    for role, path in record.paths.items():
        if path is None:
            continue
        try:
            info = inspect_nifti(path)
            infos[role] = info
            result.files[role] = nifti_info_dict(info)
        except Exception as error:
            message = str(error)
            result.errors.append(f"{role}: {message}")
            result.unreadable_files.append({"role": role, "path": str(path), "reason": message})
    reference = next((infos[x] for x in ("t1", "t1ce", "t2", "flair") if x in infos), None)
    if reference:
        for role, info in infos.items():
            if info.shape != reference.shape:
                result.errors.append(f"{role}: shape differs from modalities")
            if not np.allclose(info.spacing, reference.spacing, atol=spacing_tolerance, rtol=0):
                result.errors.append(f"{role}: spacing differs from modalities")
            if not np.allclose(info.affine, reference.affine, atol=affine_tolerance, rtol=0):
                result.errors.append(f"{role}: affine differs from modalities")
            if info.orientation != reference.orientation:
                result.errors.append(f"{role}: orientation differs from modalities")
    if record.segmentation_path and "seg" in infos:
        try:
            data = np.asanyarray(nib.load(str(record.segmentation_path)).dataobj)  # type: ignore[attr-defined]
            if not np.isfinite(data).all():
                result.errors.append("seg: non-finite values")
            if not np.equal(data, np.rint(data)).all():
                result.errors.append("seg: labels are not integer-valued")
            result.segmentation_labels = sorted(int(v) for v in np.unique(data[np.isfinite(data)]))
            unknown = sorted(set(result.segmentation_labels) - {0, 1, 2, 4})
            if unknown:
                (result.warnings if allow_unexpected_labels else result.errors).append(
                    f"seg: unexpected labels {unknown}"
                )
        except Exception as error:
            result.errors.append(f"seg: {error}")
    result.valid = not result.errors
    return result


def validate_index(index_path: Path, allow_incomplete: bool = False) -> list[str]:
    import csv

    rows = list(csv.DictReader(index_path.open(encoding="utf-8")))
    ids = [row["subject_id"] for row in rows]
    errors = [
        f"duplicate subject ID: {value}" for value in sorted(set(ids)) if ids.count(value) > 1
    ]
    if not allow_incomplete:
        errors += [
            f"incomplete subject: {r['subject_id']}" for r in rows if r.get("complete") != "True"
        ]
    return errors + [
        f"invalid subject: {r['subject_id']}"
        for r in rows
        if r.get("validation_status") == "invalid"
    ]
