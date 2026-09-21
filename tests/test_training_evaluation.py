from typing import Any

import torch

import fl_brats.training.evaluation as evaluation


def test_modality_combination_evaluation_streams_masked_batches(monkeypatch: Any) -> None:
    originals = [
        {"image": torch.ones(1, 4, 2, 2, 2), "label": torch.zeros(1, 1, 2, 2, 2)}
        for _ in range(2)
    ]
    evaluated: list[list[dict[str, torch.Tensor]]] = []

    def consume(_model: Any, batches: Any, _loss_fn: Any, _device: Any) -> dict[str, float]:
        assert not isinstance(batches, list)
        consumed = list(batches)
        evaluated.append(consumed)
        return {"loss": 0.0, "subjects": float(len(consumed))}

    monkeypatch.setattr(evaluation, "evaluate", consume)
    results = evaluation.evaluate_modality_combinations(
        torch.nn.Identity(), originals, None, torch.device("cpu")
    )

    assert len(results) == 15
    assert len(evaluated) == 15
    assert all(len(batches) == 2 for batches in evaluated)
    assert all(torch.equal(batch["image"], torch.ones_like(batch["image"])) for batch in originals)
    assert all("modality_mask" not in batch for batch in originals)
