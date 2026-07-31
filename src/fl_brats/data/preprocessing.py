"""Resumable, atomic optional materialization; raw inputs remain immutable."""

import csv
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

from fl_brats import __version__

from .hashing import sha256_file, sha256_payload
from .labels import brats_to_regions
from .normalization import foreground_zscore
from .resampling import resample_image

ROLES = ("t1", "t1ce", "t2", "flair", "segmentation")
GEOMETRY_AFFINE_TOLERANCE = 1e-4
GEOMETRY_SPACING_TOLERANCE = 1e-5


def _triple(
    value: int | list[int] | tuple[int, ...] | None, name: str
) -> tuple[int, int, int] | None:
    if value is None:
        return None
    values = (value,) * 3 if isinstance(value, int) else tuple(value)
    if len(values) != 3 or any(not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError(f"{name} must be a positive integer or three positive integers")
    return int(values[0]), int(values[1]), int(values[2])


def _source_state(paths: list[Path], root: Path, hashes: bool) -> list[dict[str, Any]]:
    return [
        {
            "path": str(p.relative_to(root)),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            **({"sha256": sha256_file(p)} if hashes else {}),
        }
        for p in paths
    ]


def _crop_bounds(
    images: list[np.ndarray], margin: tuple[int, int, int], divisible: tuple[int, int, int] | None
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    mask = np.any(np.stack([x != 0 for x in images]), axis=0)
    shape = images[0].shape
    if not mask.any():
        return (0, 0, 0), (int(shape[0]), int(shape[1]), int(shape[2]))
    coords = np.where(mask)
    start = [max(0, int(coords[i].min()) - margin[i]) for i in range(3)]
    end = [min(shape[i], int(coords[i].max()) + 1 + margin[i]) for i in range(3)]
    if divisible:
        for i, d in enumerate(divisible):
            wanted = ((end[i] - start[i] + d - 1) // d) * d
            extra = wanted - (end[i] - start[i])
            before = min(start[i], extra // 2)
            after = min(shape[i] - end[i], extra - before)
            start[i] -= before
            end[i] += after
    return (start[0], start[1], start[2]), (end[0], end[1], end[2])


def _pad(
    array: np.ndarray, target: tuple[int, int, int], value: float
) -> tuple[np.ndarray, list[list[int]]]:
    if any(target[i] < array.shape[-3 + i] for i in range(3)):
        raise ValueError("padding target_shape cannot be smaller than current shape")
    widths = []
    for i, size in enumerate(array.shape[-3:]):
        total = target[i] - size
        widths.append([total // 2, total - total // 2])
    return np.pad(
        array, [(0, 0)] * (array.ndim - 3) + [tuple(x) for x in widths], constant_values=value
    ), widths


def _validate_aligned_images(images: list[Any]) -> dict[str, Any]:
    """Ensure transformed modalities still share a voxel grid before array operations."""
    reference = images[0]
    reference_shape = tuple(int(value) for value in reference.shape[:3])
    reference_spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    reference_affine = np.asarray(reference.affine, dtype=float)
    reference_orientation = "".join(
        nib.aff2axcodes(reference_affine)  # type: ignore[no-untyped-call]
    )
    for role, image in zip(ROLES[1:], images[1:], strict=True):
        shape = tuple(int(value) for value in image.shape[:3])
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        affine = np.asarray(image.affine, dtype=float)
        orientation = "".join(nib.aff2axcodes(affine))  # type: ignore[no-untyped-call]
        if shape != reference_shape:
            raise ValueError(
                f"post-transform geometry mismatch: {role} shape differs from t1 "
                f"({shape} != {reference_shape})"
            )
        if not np.allclose(
            spacing, reference_spacing, atol=GEOMETRY_SPACING_TOLERANCE, rtol=0
        ):
            raise ValueError(f"post-transform geometry mismatch: {role} spacing differs from t1")
        if not np.allclose(affine, reference_affine, atol=GEOMETRY_AFFINE_TOLERANCE, rtol=0):
            raise ValueError(f"post-transform geometry mismatch: {role} affine differs from t1")
        if orientation != reference_orientation:
            raise ValueError(
                f"post-transform geometry mismatch: {role} orientation differs from t1"
            )
    return {
        "passed": True,
        "reference_role": "t1",
        "shape": list(reference_shape),
        "spacing": list(reference_spacing),
        "orientation": reference_orientation,
        "affine_tolerance": GEOMETRY_AFFINE_TOLERANCE,
        "spacing_tolerance": GEOMETRY_SPACING_TOLERANCE,
    }


def _shift_affine(affine: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
    translation = np.eye(4)
    translation[:3, 3] = offset
    return np.asarray(affine, dtype=float) @ translation


def _outputs_exist(target: Path, fmt: str) -> bool:
    return (
        (target / "complete.json").exists()
        and (target / "metadata.json").exists()
        and (
            (target / "data.pt").exists()
            if fmt == "tensor"
            else (target / "image.nii.gz").exists() and (target / "segmentation.nii.gz").exists()
        )
    )


def preprocess_subject(
    row: dict[str, str],
    raw_root: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> str:
    subject = row["subject_id"]
    target = output_root / "subjects" / subject
    temp = target.with_name(target.name + ".tmp")
    fmt = str(config.get("output_format", "nifti"))
    sources = [raw_root / row[f"{r}_path"] for r in ROLES]
    state = _source_state(
        sources, raw_root, bool(config.get("hashing", {}).get("source_files", False))
    )
    config_hash = sha256_payload(config)
    if resume and _outputs_exist(target, fmt):
        old = json.loads((target / "metadata.json").read_text())
        if (
            old.get("preprocessing_configuration_hash") == config_hash
            and old.get("source_files") == state
        ):
            return "skipped"
    if target.exists() and not overwrite and not resume:
        raise FileExistsError(f"processed subject exists: {target}; use --overwrite or --resume")
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir(parents=True, exist_ok=False)
    try:
        loaded: list[Any] = [nib.load(str(p)) for p in sources]
        original = [
            {
                "role": role,
                "orientation": "".join(nib.aff2axcodes(x.affine)),  # type: ignore[no-untyped-call]
                "affine": np.asarray(x.affine).tolist(),
                "shape": list(x.shape),
                "spacing": list(x.header.get_zooms()[:3]),
            }
            for role, x in zip(ROLES, loaded, strict=True)
        ]
        target_orientation = str(config.get("target_orientation", "RAS"))
        images = [nib.as_closest_canonical(x) if target_orientation == "RAS" else x for x in loaded]  # type: ignore[no-untyped-call]
        if config.get("resample", {}).get("enabled", False):
            spacing = tuple(config["resample"]["target_spacing"])
            images = [resample_image(x, spacing, i == 4) for i, x in enumerate(images)]
        transformed_geometry = _validate_aligned_images(images)
        arrays = [np.asanyarray(x.dataobj) for x in images]
        affine = np.asarray(images[0].affine, dtype=float)
        crop = config.get("crop", {})
        margin = tuple(crop.get("margin", [0, 0, 0]))
        divisible = _triple(crop.get("divisible_by"), "crop.divisible_by")
        start, end = (0, 0, 0), tuple(int(x) for x in arrays[0].shape)
        if crop.get("enabled", True):
            start, end = _crop_bounds(arrays[:4], margin, divisible)
            arrays = [x[start[0] : end[0], start[1] : end[1], start[2] : end[2]] for x in arrays]
            affine = _shift_affine(affine, start)
        norm = config.get("normalization", {})
        normalized, stats = zip(
            *(
                foreground_zscore(x, float(norm.get("epsilon", 1e-8)), norm.get("clip"))
                for x in arrays[:4]
            ),
            strict=True,
        )
        image_data = np.stack(normalized)
        seg = arrays[4]
        padding = config.get("padding", {})
        pad_widths = [[0, 0]] * 3
        before_padding = list(image_data.shape[-3:])
        if padding.get("enabled", False):
            target_shape = _triple(padding.get("target_shape"), "padding.target_shape") or tuple(
                ((s + d - 1) // d) * d
                for s, d in zip(
                    before_padding,
                    _triple(padding.get("divisible_by"), "padding.divisible_by") or (1, 1, 1),
                    strict=True,
                )
            )
            image_data, pad_widths = _pad(
                image_data, target_shape, float(padding.get("image_value", 0.0))
            )
            seg, _ = _pad(seg, target_shape, float(padding.get("segmentation_value", 0)))
            affine = _shift_affine(affine, tuple(-width[0] for width in pad_widths))
        if config.get("convert_labels", False):
            seg = brats_to_regions(seg).numpy()
        if fmt == "tensor":
            import torch

            torch.save(
                {"image": torch.from_numpy(image_data), "segmentation": torch.as_tensor(seg)},
                temp / "data.pt",
            )
        else:
            nib.save(nib.Nifti1Image(image_data, affine), str(temp / "image.nii.gz"))  # type: ignore[no-untyped-call]
            segmentation_image = nib.Nifti1Image(seg.astype(np.uint8), affine)  # type: ignore[no-untyped-call]
            nib.save(segmentation_image, str(temp / "segmentation.nii.gz"))
        final_orientation = "".join(nib.aff2axcodes(affine))  # type: ignore[no-untyped-call]
        metadata = {
            "subject_id": subject,
            "source_files": state,
            "inputs": original,
            "all_inputs_same_orientation": len({x["orientation"] for x in original}) == 1,
            "any_reoriented": any(x["orientation"] != final_orientation for x in original),
            "post_transform_geometry": transformed_geometry,
            "final_orientation": final_orientation,
            "final_affine": np.asarray(affine).tolist(),
            "final_shape": list(image_data.shape[-3:]),
            "final_spacing": list(images[0].header.get_zooms()[:3]),
            "crop_coordinates": {
                "start": start,
                "end": end,
                "requested_divisible_by": divisible,
                "realised_shape": [end[i] - start[i] for i in range(3)],
            },
            "padding": {
                "pre_padding_shape": before_padding,
                "post_padding_shape": list(image_data.shape[-3:]),
                "pad_widths": pad_widths,
                "configuration": padding,
            },
            "normalization_statistics": list(stats),
            "segmentation_labels_present": sorted(int(x) for x in np.unique(arrays[4])),
            "preprocessing_configuration_hash": config_hash,
            "pipeline_version": __version__,
            "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (temp / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
        (temp / "complete.json").write_text(
            json.dumps({"subject_id": subject, "config_hash": config_hash}), encoding="utf-8"
        )
        if target.exists():
            shutil.rmtree(target)
        temp.replace(target)
        return "completed"
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _preprocess_subject_task(
    row: dict[str, str],
    raw_root: Path,
    output_root: Path,
    config: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> str:
    return preprocess_subject(
        row,
        raw_root,
        output_root,
        config,
        resume=resume,
        overwrite=overwrite,
    )


def preprocess_index(
    index_path: Path, raw_root: Path, output_root: Path, config: dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preprocessing_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    rows = [
        r
        for r in csv.DictReader(index_path.open(encoding="utf-8"))
        if r["validation_status"] == "valid"
    ]
    subject, limit = kwargs.get("subject_id"), kwargs.get("limit")
    rows = [r for r in rows if not subject or r["subject_id"] == subject][:limit]
    started = time.monotonic()
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    materialize = bool(config.get("materialize", False))
    workers = int(kwargs.get("workers") or config.get("workers", 0) or 0)
    if materialize:
        resume = bool(kwargs.get("resume", False))
        overwrite = bool(kwargs.get("overwrite", False))
        if workers > 1 and len(rows) > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _preprocess_subject_task,
                        row,
                        raw_root,
                        output_root,
                        config,
                        resume,
                        overwrite,
                    )
                    for row in rows
                ]
                for future in as_completed(futures):
                    try:
                        counts[future.result()] += 1
                    except Exception:
                        counts["failed"] += 1
        else:
            for row in rows:
                try:
                    counts[
                        preprocess_subject(
                            row,
                            raw_root,
                            output_root,
                            config,
                            resume=resume,
                            overwrite=overwrite,
                        )
                    ] += 1
                except Exception:
                    counts["failed"] += 1
    summary = {
        "mode": "materialized" if materialize else "manifest-only",
        "subjects_attempted": len(rows),
        **counts,
        "total_runtime_seconds": time.monotonic() - started,
        "output_format": config.get("output_format", "nifti"),
        "workers": workers if materialize else 0,
        "preprocessing_config_hash": sha256_payload(config),
        "dataset_index_hash": sha256_file(index_path),
    }
    summary["average_runtime_per_subject_seconds"] = (
        summary["total_runtime_seconds"] / len(rows) if rows else 0
    )
    (output_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
