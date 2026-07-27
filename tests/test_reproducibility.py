import random

import numpy as np

from fl_brats.utils.reproducibility import seed_everything


def test_seed_is_reproducible() -> None:
    seed_everything(7)
    first = (random.random(), float(np.random.rand()))
    seed_everything(7)
    assert first == (random.random(), float(np.random.rand()))
