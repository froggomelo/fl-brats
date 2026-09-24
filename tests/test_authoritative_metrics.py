import pytest
import torch

from fl_brats.metrics.reporting import (
    communication_summary,
    convergence_rounds,
    dice_communication_curve,
    pareto_frontier,
    per_client_summary,
)
from fl_brats.metrics.segmentation import segmentation_metrics


def test_segmentation_metrics_cover_authoritative_regions_and_metrics() -> None:
    target = torch.zeros((1, 3, 4, 4, 4))
    target[:, :, 1:3, 1:3, 1:3] = 1
    logits = torch.where(target.bool(), torch.tensor(20.0), torch.tensor(-20.0))
    metrics = segmentation_metrics(logits, target)

    assert set(metrics) == {
        f"{region}_{metric}"
        for region in ("wt", "tc", "et")
        for metric in ("dice", "hd95", "sensitivity")
    } | {"mean_dice", "mean_hd95", "mean_sensitivity"}
    assert metrics["mean_dice"] == 1.0
    assert metrics["mean_hd95"] == 0.0
    assert metrics["mean_sensitivity"] == 1.0


def test_empty_metric_policy_is_finite_and_explicit() -> None:
    target = torch.zeros((1, 3, 2, 2, 2))
    logits = torch.full_like(target, -20)
    metrics = segmentation_metrics(logits, target)
    assert metrics["mean_dice"] == metrics["mean_sensitivity"] == 1.0
    assert metrics["mean_hd95"] == 0.0


def test_dice_only_metrics_skip_expensive_hd95_and_sensitivity() -> None:
    target = torch.zeros((1, 3, 2, 2, 2))
    logits = torch.full_like(target, -20)

    metrics = segmentation_metrics(logits, target, metric_names=("dice",))

    assert set(metrics) == {
        "wt_dice",
        "tc_dice",
        "et_dice",
        "mean_dice",
    }


def test_run_level_reporting_uses_final_dice_and_measured_payloads() -> None:
    evaluations = [
        {"round": 1, "validation_mean_dice": 0.7},
        {"round": 2, "validation_mean_dice": 1.0},
        {"round": 3, "validation_mean_dice": 0.8},
    ]
    communication = [
        {"round": 1, "uploaded_bytes": 10, "downloaded_bytes": 5},
        {"round": 2, "uploaded_bytes": 8, "downloaded_bytes": 2},
        {"round": 3, "uploaded_bytes": 4, "downloaded_bytes": 1},
    ]
    assert convergence_rounds(evaluations) == {
        "round_to_90_percent_final_dice": 2,
        "round_to_95_percent_final_dice": 2,
    }
    summary = communication_summary(communication)
    assert summary["total_bytes_transferred"] == 30
    assert summary["rounds"][-1]["cumulative_megabytes"] == 0.00003
    curve = dice_communication_curve(evaluations, communication)
    assert pareto_frontier(curve) == curve[:2]


def test_per_client_dice_variance_is_reported() -> None:
    clients = [
        {"wt_dice": 0.4, "tc_dice": 0.5, "et_dice": 0.6, "mean_dice": 0.5},
        {"wt_dice": 0.8, "tc_dice": 0.5, "et_dice": 0.4, "mean_dice": 0.6},
    ]
    result = per_client_summary(clients)
    assert result["dice_variance"]["wt_dice"] == pytest.approx(0.04)
    assert len(result["clients"]) == 2
