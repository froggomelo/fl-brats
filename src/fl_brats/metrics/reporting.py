"""Run-level research metric summaries and comparison-ready records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import pvariance
from typing import Any

from .segmentation import REGIONS


def per_client_summary(clients: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain client scores and quantify population variance across clients."""
    variances: dict[str, float | None] = {}
    for region in (*REGIONS, "mean"):
        key = f"{region}_dice"
        values = [float(item[key]) for item in clients if isinstance(item.get(key), (int, float))]
        variances[key] = pvariance(values) if values else None
    return {"clients": [dict(item) for item in clients], "dice_variance": variances}


def convergence_rounds(records: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    """Find first rounds reaching 90% and 95% of final validation mean Dice."""
    usable = [
        item for item in records if isinstance(item.get("validation_mean_dice"), (int, float))
    ]
    final = float(usable[-1]["validation_mean_dice"]) if usable else None
    result: dict[str, int | None] = {}
    for percentage in (90, 95):
        threshold = final * percentage / 100 if final is not None else None
        match = next(
            (
                int(item["round"])
                for item in usable
                if threshold is not None and float(item["validation_mean_dice"]) >= threshold
            ),
            None,
        )
        result[f"round_to_{percentage}_percent_final_dice"] = match
    return result


def communication_summary(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize measured application payloads without claiming inaccessible wire timing."""
    points, cumulative = [], 0
    for item in rounds:
        transferred = int(item.get("uploaded_bytes", 0)) + int(item.get("downloaded_bytes", 0))
        cumulative += transferred
        points.append(
            {
                "round": int(item["round"]),
                "uploaded_bytes": int(item.get("uploaded_bytes", 0)),
                "downloaded_bytes": int(item.get("downloaded_bytes", 0)),
                "transferred_bytes": transferred,
                "transferred_megabytes": transferred / 1_000_000,
                "cumulative_bytes": cumulative,
                "cumulative_megabytes": cumulative / 1_000_000,
                "network_transmission_seconds": item.get("network_transmission_seconds"),
            }
        )
    return {
        "rounds": points,
        "total_bytes_transferred": cumulative,
        "cumulative_megabytes_transferred": cumulative / 1_000_000,
        "network_transmission_time_available": False,
        "measurement_note": (
            "Application-level model payload bytes are measured. Flower does not expose "
            "network-only wall-clock transmission duration or transport-header bytes."
        ),
    }


def dice_communication_curve(
    evaluations: Sequence[Mapping[str, Any]], communication: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join validation Dice to cumulative communication for Pareto comparison."""
    costs: dict[int, int] = {}
    cumulative = 0
    for item in communication:
        cumulative += int(item.get("uploaded_bytes", 0)) + int(item.get("downloaded_bytes", 0))
        costs[int(item["round"])] = cumulative
    return [
        {
            "round": int(item["round"]),
            "mean_dice": float(item["validation_mean_dice"]),
            "cumulative_bytes": costs.get(int(item["round"])),
            "cumulative_megabytes": costs.get(int(item["round"]), 0) / 1_000_000,
        }
        for item in evaluations
        if isinstance(item.get("validation_mean_dice"), (int, float))
        and int(item["round"]) in costs
    ]


def pareto_frontier(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return non-dominated high-Dice/low-communication points."""
    return [
        dict(point)
        for point in points
        if not any(
            float(other["mean_dice"]) >= float(point["mean_dice"])
            and int(other["cumulative_bytes"]) <= int(point["cumulative_bytes"])
            and (
                float(other["mean_dice"]) > float(point["mean_dice"])
                or int(other["cumulative_bytes"]) < int(point["cumulative_bytes"])
            )
            for other in points
        )
    ]
