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


def test_parallel_hd95_matches_serial_evaluation() -> None:
    import torch
    from torch.utils.data import DataLoader

    from fl_brats.training.engine import evaluate

    torch.manual_seed(0)
    records = []
    for _ in range(5):
        label = torch.zeros(3, 12, 12, 12)
        label[:, 3:8, 4:9, 2:7] = 1
        records.append({"image": torch.randn(4, 12, 12, 12), "label": label})
    loader = DataLoader(records, batch_size=1)
    model = torch.nn.Conv3d(4, 3, kernel_size=1)
    loss = torch.nn.BCEWithLogitsLoss()
    serial = evaluate(model, loader, loss, torch.device("cpu"))
    parallel = evaluate(model, loader, loss, torch.device("cpu"), hd95_workers=2)
    assert serial == parallel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU HD95 needs CUDA")
def test_gpu_hd95_matches_monai_exactly() -> None:
    from monai.metrics import compute_hausdorff_distance

    from fl_brats.metrics.segmentation import _hd95_gpu

    generator = torch.Generator().manual_seed(0)
    for case in range(40):
        shape = (40, 36, 30)
        truth = torch.zeros(shape, dtype=torch.bool)
        truth[8:25, 10:24, 5:20] = True
        noise = torch.rand(shape, generator=generator) < (0.002 * (case % 5))
        prediction = torch.roll(truth, shifts=case % 4, dims=case % 3) | noise
        expected = float(
            compute_hausdorff_distance(
                prediction[None, None].float(), truth[None, None].float(),
                include_background=True, percentile=95,
            ).item()
        )
        assert _hd95_gpu(prediction.cuda(), truth.cuda()) == expected
