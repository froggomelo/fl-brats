from typing import Any

from monai.transforms import Randomizable

from fl_brats.data.brats import build_dataloader
from monai.data import Dataset


class _Draw(Randomizable):
    def randomize(self, data: Any = None) -> None:
        self.value = self.R.random()

    def __call__(self, data: dict) -> dict:
        self.randomize()
        return {**data, "draw": self.value}


def _draws(workers: int) -> list[float]:
    dataset = Dataset([{"index": i} for i in range(8)], transform=_Draw())
    loader = build_dataloader(dataset, 1, False, workers=workers, seed=7, pin_memory=False)
    return [float(item["draw"][0]) for item in loader]


def test_workers_draw_independent_augmentation() -> None:
    draws = _draws(2)
    # Without per-worker reseeding both workers replay one stream: draws come in
    # identical pairs (worker 0 and worker 1 alternate batches).
    assert len(set(draws)) == len(draws)


def test_worker_draws_are_reproducible() -> None:
    assert _draws(2) == _draws(2)
