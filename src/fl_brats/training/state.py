"""Training state records."""

from dataclasses import dataclass


@dataclass
class TrainingState:
    epoch: int = 0
    best_metric: float = float("-inf")
