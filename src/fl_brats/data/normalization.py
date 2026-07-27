"""Leakage-free per-subject foreground intensity normalization."""

import numpy as np


def foreground_zscore(
    array: np.ndarray, epsilon: float = 1e-8, clip: float | None = None
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    data = np.asarray(array, dtype=np.float32)
    mask = data != 0
    count = int(mask.sum())
    stats: dict[str, float | int | bool] = {
        "foreground_voxel_count": count,
        "mean": 0.0,
        "std": 0.0,
        "minimum": 0.0,
        "maximum": 0.0,
        "skipped": False,
    }
    if not count:
        stats["skipped"] = True
        return np.zeros_like(data), stats
    values = data[mask]
    mean, std = float(values.mean()), float(values.std())
    stats.update(mean=mean, std=std, minimum=float(values.min()), maximum=float(values.max()))
    if std <= epsilon:
        stats["skipped"] = True
        return np.where(mask, 0.0, data).astype(np.float32), stats
    output = np.zeros_like(data)
    output[mask] = (values - mean) / std
    if clip is not None:
        output[mask] = np.clip(output[mask], -clip, clip)
    return output, stats
