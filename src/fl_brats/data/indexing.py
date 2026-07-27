"""Dataset index creation with deliberately explicit validation results."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .discovery import REQUIRED_SUFFIXES, discover_subjects, subject_files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset_index(root: Path, output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Scan BraTS files and write a CSV index plus a complete validation summary."""
    root, output_dir = root.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_subjects(root)
    rows: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    invalid: list[str] = []
    labels: Counter[int] = Counter()
    counts: Counter[str] = Counter()
    for subject_id, directories in sorted(discovered.items()):
        for directory in directories:
            files = subject_files(subject_id, directory)
            existing = {key: value.exists() for key, value in files.items()}
            counts.update(key for key, value in existing.items() if value)
            row: dict[str, Any] = {"subject_id": subject_id, "source_collection": directory.parent.name, "complete": all(existing.values())}
            row.update({f"{key}_path": str(path) for key, path in files.items()})
            row.update({f"{key}_bytes": path.stat().st_size if path.exists() else 0 for key, path in files.items()})
            row.update({f"has_{key}": value for key, value in existing.items()})
            if not row["complete"]:
                missing[f"{subject_id}:{directory}"] = [key for key, value in existing.items() if not value]
            probe = next((path for path in files.values() if path.exists()), None)
            if probe:
                try:
                    image: Any = nib.load(str(probe))
                    row["shape"] = json.dumps(list(image.shape))
                    row["spacing"] = json.dumps([float(v) for v in image.header.get_zooms()[:3]])
                    row["orientation"] = "".join(nib.aff2axcodes(image.affine))  # type: ignore[no-untyped-call]
                    if files["seg"].exists():
                        segmentation: Any = nib.load(str(files["seg"]))
                        values = np.unique(np.asanyarray(segmentation.dataobj)).astype(int)
                        labels.update(values.tolist())
                        row["segmentation_labels"] = json.dumps(values.tolist())
                except Exception as error:  # nibabel errors differ by backend
                    invalid.append(f"{subject_id}: {error}")
            rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    index_path = output_dir / "dataset_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    duplicates = sorted(key for key, paths in discovered.items() if len(paths) > 1)
    observed = sorted(labels)
    summary = {"root": str(root), "total_discovered_subjects": len(rows), "complete_subjects": sum(bool(r["complete"]) for r in rows), "incomplete_subjects": sum(not bool(r["complete"]) for r in rows), "duplicate_subject_ids": duplicates, "missing_files": missing, "counts_per_modality": dict(counts), "segmentation_labels_found": observed, "unexpected_labels": sorted(set(observed) - {0, 1, 2, 4}), "invalid_or_unreadable_nifti": invalid, "dataset_index_sha256": sha256_file(index_path)}
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return index_path, summary_path, summary
