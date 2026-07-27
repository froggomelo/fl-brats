"""Portable dataset index, detailed validation report, and consistent summary."""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from fl_brats import __version__

from .discovery import REQUIRED_SUFFIXES, discover_records
from .hashing import sha256_file
from .validation import validate_subject


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def build_dataset_index(
    root: Path, output_dir: Path, allow_unexpected_labels: bool = False, relative_paths: bool = True
) -> tuple[Path, Path, dict[str, Any]]:
    root, output_dir = root.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_records(root)
    by_id: dict[str, list[Path]] = defaultdict(list)
    for record in records:
        by_id[record.subject_id].append(record.subject_directory)
    duplicate_ids = {key: sorted(map(str, paths)) for key, paths in by_id.items() if len(paths) > 1}
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, object]] = []
    missing = Counter[str]()
    ambiguous = Counter[str]()
    zero = Counter[str]()
    unreadable = Counter[str]()
    unreadable_details = []
    shapes = Counter[str]()
    spacings = Counter[str]()
    orientations = Counter[str]()
    dtypes = Counter[str]()
    labels = Counter[int]()
    total_bytes = valid_bytes = invalid_bytes = 0
    for record in records:
        result = validate_subject(
            record, allow_unexpected_labels, duplicate_directories=by_id[record.subject_id]
        )
        for role in record.missing_files:
            missing[role] += 1
        for role in record.duplicate_files:
            ambiguous[role] += 1
        for item in result.unreadable_files:
            unreadable[item["role"]] += 1
            unreadable_details.append({"subject_id": record.subject_id, **item})
        sizes: list[int] = []
        for role, path in record.paths.items():
            if path and path.exists():
                size = path.stat().st_size
                sizes.append(size)
                if not size:
                    zero[role] += 1
        subject_bytes = sum(sizes)
        total_bytes += subject_bytes
        if result.valid:
            valid_bytes += subject_bytes
        else:
            invalid_bytes += subject_bytes
            invalid.append(result.as_dict())
        first = next(iter(result.files.values()), {})
        shape, spacing, orientation = (
            first.get("shape"),
            first.get("spacing"),
            first.get("orientation"),
        )
        if shape:
            shapes[_json(shape)] += 1
        if spacing:
            spacings[_json(spacing)] += 1
        if orientation:
            orientations[str(orientation)] += 1
        for role in ("t1", "t1ce", "t2", "flair"):
            if role in result.files:
                dtypes[str(result.files[role]["dtype"])] += 1
        labels.update(result.segmentation_labels)

        def fmt(path: Path | None) -> str:
            return (
                "" if path is None else str(path.relative_to(root)) if relative_paths else str(path)
            )

        row = {
            "subject_id": record.subject_id,
            "source_collection": record.source_collection,
            "subject_directory": fmt(record.subject_directory),
            "t1_path": fmt(record.t1_path),
            "t1ce_path": fmt(record.t1ce_path),
            "t2_path": fmt(record.t2_path),
            "flair_path": fmt(record.flair_path),
            "segmentation_path": fmt(record.segmentation_path),
            "seg_path": fmt(record.segmentation_path),
            "complete": record.complete,
            "validation_status": "valid" if result.valid else "invalid",
            "missing_files": _json(record.missing_files),
            "ambiguous_files": _json(record.duplicate_files),
            "image_shape": _json(shape or []),
            "voxel_spacing": _json(spacing or []),
            "orientation": orientation or "",
            "modality_dtype": _json(
                {k: v.get("dtype") for k, v in result.files.items() if k != "seg"}
            ),
            "segmentation_dtype": result.files.get("seg", {}).get("dtype", ""),
            "segmentation_labels": _json(result.segmentation_labels),
            "total_subject_bytes": subject_bytes,
        }
        rows.append(row)
    fields = list(rows[0]) if rows else ["subject_id"]
    csv_path = output_dir / "dataset_index.csv"
    jsonl_path = output_dir / "dataset_index.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    summary: dict[str, Any] = {
        "candidate_subject_directories": len({r.subject_directory for r in records}),
        "discovered_subjects": len(records),
        "unique_subject_ids": len(by_id),
        "duplicate_subject_ids": duplicate_ids,
        "complete_subjects": sum(r.complete for r in records),
        "incomplete_subjects": sum(not r.complete for r in records),
        "valid_subjects": sum(r["validation_status"] == "valid" for r in rows),
        "invalid_subjects": len(invalid),
        "files_expected": len(records) * len(REQUIRED_SUFFIXES),
        "files_discovered": sum(bool(p) for r in records for p in r.paths.values()),
        "missing_files_by_role": dict(missing),
        "ambiguous_files_by_role": dict(ambiguous),
        "zero_byte_files_by_role": dict(zero),
        "zero_byte_file_count": sum(zero.values()),
        "unreadable_files_by_role": dict(unreadable),
        "unreadable_file_count": len(unreadable_details),
        "subjects_with_unreadable_files": sorted({x["subject_id"] for x in unreadable_details}),
        "unreadable_file_details": unreadable_details,
        "invalid_geometry_count": sum(
            any("differs from modalities" in e for e in cast(list[str], x["errors"])) for x in invalid
        ),
        "unexpected_segmentation_labels": sorted(set(labels) - {0, 1, 2, 4}),
        "shape_distribution": dict(shapes),
        "spacing_distribution": dict(spacings),
        "orientation_distribution": dict(orientations),
        "modality_data_types": dict(dtypes),
        "segmentation_label_distribution": {str(k): v for k, v in labels.items()},
        "total_dataset_bytes": total_bytes,
        "valid_dataset_bytes": valid_bytes,
        "invalid_dataset_bytes": invalid_bytes,
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(root),
        "pipeline_version": __version__,
        "relative_paths": relative_paths,
        "index_sha256": sha256_file(csv_path),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "invalid_subjects.json").write_text(
        json.dumps(invalid, indent=2, sort_keys=True), encoding="utf-8"
    )
    return csv_path, output_dir / "dataset_summary.json", summary
