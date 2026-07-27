"""Simple sample-weighted scalar aggregation."""


def weighted_mean(values: list[tuple[float, int]]) -> float:
    count = sum(n for _, n in values)
    return sum(value * n for value, n in values) / count if count else 0.0
