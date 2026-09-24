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


def test_workers_are_spawned_not_forked() -> None:
    # Forked workers of a process with live gRPC threads abort.
    dataset = Dataset([{"index": 0}], transform=_Draw())
    loader = build_dataloader(dataset, 1, False, workers=1, seed=7, pin_memory=False)
    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_worker_loader_raises_open_file_limit() -> None:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, hard), hard))
    try:
        build_dataloader(Dataset([{"index": 0}]), 1, False, workers=1, seed=7, pin_memory=False)
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == hard
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
