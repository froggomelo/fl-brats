"""Canonical BraTS modality names, masks, and deterministic missingness policies."""

from __future__ import annotations

import hashlib
import random
from itertools import combinations
from typing import Iterable, Mapping, Protocol

import torch

MODALITIES: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
POLICY_VERSION = "missing-modalities-v1"


def canonicalise_modality_name(name: str) -> str:
    """Return a canonical modality name or raise for unsupported names."""
    value = name.lower().strip()
    if value not in MODALITIES:
        raise ValueError(f"unknown modality {name!r}; expected one of {MODALITIES}")
    return value


def canonical_modality_order(modalities: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Validate and order a modality collection in fixed BraTS channel order."""
    values = tuple(canonicalise_modality_name(item) for item in modalities)
    if len(values) != len(set(values)):
        raise ValueError("duplicate modality names are not permitted")
    return tuple(item for item in MODALITIES if item in values)


def modality_combination_key(modalities: tuple[str, ...] | list[str]) -> str:
    """Serialize a non-empty availability set in canonical order."""
    ordered = canonical_modality_order(modalities)
    if not ordered:
        raise ValueError("a modality combination cannot be empty")
    return "+".join(ordered)


def combination_to_mask(modalities: tuple[str, ...] | list[str]) -> torch.Tensor:
    """Return the canonical bool [4] availability mask."""
    values = set(canonical_modality_order(modalities))
    if not values:
        raise ValueError("a sample must have at least one modality")
    return torch.tensor([item in values for item in MODALITIES], dtype=torch.bool)


def mask_to_combination(mask: torch.Tensor | list[bool]) -> tuple[str, ...]:
    """Decode a canonical availability mask."""
    values = mask.tolist() if isinstance(mask, torch.Tensor) else mask
    if len(values) != len(MODALITIES):
        raise ValueError("modality mask must have shape [4]")
    result = tuple(item for item, present in zip(MODALITIES, values, strict=True) if bool(present))
    if not result:
        raise ValueError("a sample must have at least one modality")
    return result


def validate_modality_mask(mask: torch.Tensor) -> torch.Tensor:
    """Validate canonical non-empty modality masks and return bool [B, 4]."""
    value = mask.bool()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != len(MODALITIES):
        raise ValueError("modality mask must have shape [4] or [B, 4]")
    if not value.any(dim=1).all():
        raise ValueError("each sample requires at least one modality")
    return value


def mask_to_combination_id(mask: torch.Tensor | list[bool]) -> torch.Tensor:
    """Encode canonical availability masks as non-zero four-bit IDs in [1, 15]."""
    value = torch.as_tensor(mask, dtype=torch.bool)
    batched = validate_modality_mask(value)
    weights = torch.tensor([1, 2, 4, 8], dtype=torch.long, device=batched.device)
    ids = (batched.long() * weights).sum(dim=1)
    return ids if value.ndim == 2 else ids.squeeze(0)


def combination_id_to_mask(combination_id: torch.Tensor | int) -> torch.Tensor:
    """Decode non-zero four-bit modality-combination IDs to bool masks."""
    ids = torch.as_tensor(combination_id, dtype=torch.long)
    if ((ids < 1) | (ids > 15)).any():
        raise ValueError("modality combination IDs must be in [1, 15]")
    bits = torch.tensor([1, 2, 4, 8], dtype=torch.long, device=ids.device)
    masks = (ids.unsqueeze(-1) & bits).ne(0)
    return masks if ids.ndim > 0 else masks.squeeze(0)


def modality_counts(masks: torch.Tensor | Iterable[Iterable[bool]]) -> dict[str, int]:
    """Count available samples per canonical modality."""
    batched = validate_modality_mask(torch.as_tensor(masks, dtype=torch.bool))
    sums = batched.long().sum(dim=0).tolist()
    return {modality: int(sums[index]) for index, modality in enumerate(MODALITIES)}


def modality_combination_counts(masks: torch.Tensor | Iterable[Iterable[bool]]) -> dict[int, int]:
    """Count samples per non-empty four-bit modality-combination ID."""
    ids = mask_to_combination_id(torch.as_tensor(masks, dtype=torch.bool)).view(-1)
    return {int(key): int((ids == key).sum()) for key in ids.unique(sorted=True)}


def all_non_empty_modality_combinations() -> tuple[tuple[str, ...], ...]:
    """Return all fifteen non-empty combinations in deterministic order."""
    return tuple(combo for size in range(1, 5) for combo in combinations(MODALITIES, size))


class MissingModalityPolicy(Protocol):
    def available_modalities(self, *, subject_id: str, client_id: str, split: str) -> tuple[str, ...]: ...


def _rng(seed: int, subject_id: str, client_id: str, split: str) -> random.Random:
    payload = f"{POLICY_VERSION}|{seed}|{split}|{client_id}|{subject_id}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


class NoMissingnessPolicy:
    """Policy returning all modalities."""
    def available_modalities(self, *, subject_id: str, client_id: str, split: str) -> tuple[str, ...]:
        return MODALITIES


class CompleteMissingnessPolicy:
    """One deterministic availability set for each client."""
    def __init__(self, seed: int, explicit: Mapping[str, list[str]] | None = None,
                 min_available_modalities: int = 1, max_available_modalities: int = 4) -> None:
        if not 1 <= min_available_modalities <= max_available_modalities <= 4:
            raise ValueError("complete availability bounds must be between 1 and 4")
        self.seed, self.explicit, self.minimum, self.maximum = seed, dict(explicit or {}), min_available_modalities, max_available_modalities
        for client, values in self.explicit.items():
            if not client:
                raise ValueError("client policy references an empty client id")
            canonical_modality_order(values)
            if not values:
                raise ValueError("client policy cannot be empty")

    def available_modalities(self, *, subject_id: str, client_id: str, split: str) -> tuple[str, ...]:
        if client_id in self.explicit:
            return canonical_modality_order(self.explicit[client_id])
        generator = _rng(self.seed, "client-policy", client_id, split)
        count = generator.randint(self.minimum, self.maximum)
        return canonical_modality_order(generator.sample(list(MODALITIES), count))


class ArbitraryMissingnessPolicy:
    """Stable independent or explicit-combination sampling per subject."""
    def __init__(self, seed: int, probabilities: Mapping[str, float], combinations: Mapping[str, float] | None = None,
                 require_at_least_one: bool = True) -> None:
        self.seed, self.probabilities, self.combinations, self.require_at_least_one = seed, dict(probabilities), dict(combinations or {}), require_at_least_one
        if set(self.probabilities) != set(MODALITIES) or any(not 0 <= float(v) <= 1 for v in self.probabilities.values()):
            raise ValueError("per-modality missing probabilities must cover canonical modalities and be in [0, 1]")
        if self.combinations:
            parsed = {modality_combination_key(key.split("+")): float(value) for key, value in self.combinations.items()}
            if len(parsed) != len(self.combinations) or any(not 0 <= v <= 1 for v in parsed.values()) or abs(sum(parsed.values()) - 1) > 1e-8:
                raise ValueError("combination probabilities must be unique, valid, and sum to 1")
            self.combinations = parsed

    def available_modalities(self, *, subject_id: str, client_id: str, split: str) -> tuple[str, ...]:
        generator = _rng(self.seed, subject_id, client_id, split)
        if self.combinations:
            point, running = generator.random(), 0.0
            for key, probability in self.combinations.items():
                running += probability
                if point <= running:
                    return tuple(key.split("+"))
            return tuple(next(reversed(self.combinations)).split("+"))
        available = tuple(item for item in MODALITIES if generator.random() >= self.probabilities[item])
        if available or not self.require_at_least_one:
            return available
        return (MODALITIES[generator.randrange(len(MODALITIES))],)


class MixedMissingnessPolicy:
    """Apply complete policy for explicitly configured clients, arbitrary otherwise."""
    def __init__(self, complete: CompleteMissingnessPolicy, arbitrary: ArbitraryMissingnessPolicy) -> None:
        self.complete, self.arbitrary = complete, arbitrary

    def available_modalities(self, *, subject_id: str, client_id: str, split: str) -> tuple[str, ...]:
        return self.complete.available_modalities(subject_id=subject_id, client_id=client_id, split=split) if client_id in self.complete.explicit else self.arbitrary.available_modalities(subject_id=subject_id, client_id=client_id, split=split)


DIRICHLET_COMBINATION_MODE = "dirichlet_combination"


def dirichlet_combination_assignment(
    client_subjects: list[list[str]], alpha: float, seed: int
) -> tuple[list[dict[str, tuple[str, ...]]], list[list[float]]]:
    """Give every training subject a modality combination, FedAMM style.

    Port of generate_mask_distribution_only_maskid from the FedAMM authors' code
    (github.com/13sky/FedAMM, utils/preprocessing/generate_dir_imb_mr.py). Each
    client draws its own Dirichlet(alpha * ones(15)) over the 15 non-empty
    combinations, gets one subject of every combination, and draws the rest from
    its Dirichlet; the list is shuffled and dealt to the client's subjects in
    sorted order. All clients' proportions come from one seeded draw, as there.
    Returns the per-client assignment and each client's Dirichlet proportions.
    """
    import numpy as np

    options = all_non_empty_modality_combinations()
    if alpha <= 0:
        raise ValueError("dirichlet_combination alpha must be positive")
    generator = np.random.RandomState(seed)
    proportions = generator.dirichlet(np.repeat(float(alpha), len(options)), size=len(client_subjects))
    assignments = []
    for subjects, weights in zip(client_subjects, proportions, strict=True):
        if len(subjects) < len(options):
            raise ValueError(f"each client needs at least {len(options)} subjects, one per combination")
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("the Dirichlet draw degenerated; use a larger alpha")
        extra = generator.choice(len(options), size=len(subjects) - len(options), p=weights / total)
        ids = list(range(len(options))) + extra.tolist()
        generator.shuffle(ids)
        assignments.append({subject: options[i] for subject, i in zip(sorted(subjects), ids, strict=True)})
    return assignments, proportions.tolist()


def pooled_training_distribution(partition_root: object) -> dict[str, int]:
    """Count training subjects per combination over every client of a partition."""
    import json
    from pathlib import Path

    counts: dict[str, int] = {}
    for path in sorted(Path(str(partition_root)).glob("client_*.json")):
        for item in json.loads(path.read_text(encoding="utf-8"))["subjects"]:
            key = modality_combination_key(item.get("available_modalities", list(MODALITIES)))
            counts[key] = counts.get(key, 0) + 1
    return counts


class DirichletCombinationPolicy:
    """FedAMM-style per-client Dirichlet over the 15 combinations.

    Training subjects get their combination from the partition as a whole
    (assign; written into the client manifests). Validation and test subjects
    draw theirs, per subject and seeded, from the pooled training distribution.
    """

    def __init__(self, seed: int, alpha: float, pooled: Mapping[str, float] | None = None) -> None:
        if alpha <= 0:
            raise ValueError("dirichlet_combination.alpha must be positive")
        self.seed, self.alpha = seed, float(alpha)
        self.pooled = {
            modality_combination_key(key.split("+")): float(value)
            for key, value in (pooled or {}).items()
            if float(value) > 0
        }

    def assign(self, client_subjects: list[list[str]]) -> tuple[list[dict[str, tuple[str, ...]]], list[list[float]]]:
        return dirichlet_combination_assignment(client_subjects, self.alpha, self.seed)

    def available_modalities(self, *, subject_id: str, client_id: str, split: str) -> tuple[str, ...]:
        if split == "train":
            raise ValueError("dirichlet_combination training masks come from the partition manifests")
        if not self.pooled:
            raise ValueError("dirichlet_combination held-out masks need the training partition (data.partition_root)")
        keys = [key for key in map(modality_combination_key, all_non_empty_modality_combinations()) if key in self.pooled]
        point = _rng(self.seed, subject_id, client_id, split).random() * sum(self.pooled.values())
        running = 0.0
        for key in keys:
            running += self.pooled[key]
            if point < running:
                return tuple(key.split("+"))
        return tuple(keys[-1].split("+"))


def policy_from_config(config: Mapping[str, object], partition_root: object | None = None) -> MissingModalityPolicy:
    """Build the configured deterministic policy.

    dirichlet_combination needs partition_root for held-out subjects: they follow
    the pooled training distribution of that partition.
    """
    if not config.get("enabled", False) or config.get("mode", "none") == "none": return NoMissingnessPolicy()
    seed = int(config.get("seed", 42))
    if config.get("mode") == DIRICHLET_COMBINATION_MODE:
        settings = dict(config.get(DIRICHLET_COMBINATION_MODE, {}))
        pooled = pooled_training_distribution(partition_root) if partition_root else None
        return DirichletCombinationPolicy(seed, float(settings.get("alpha", 0.0)), pooled)
    complete = dict(config.get("complete", {}))
    arbitrary = dict(config.get("arbitrary", {}))
    cp = CompleteMissingnessPolicy(seed, complete.get("client_available_modalities"), int(complete.get("min_available_modalities", 1)), int(complete.get("max_available_modalities", 4)))
    ap = ArbitraryMissingnessPolicy(seed, arbitrary.get("per_modality_missing_probability", {m: 0.0 for m in MODALITIES}), arbitrary.get("combination_probabilities"), bool(config.get("require_at_least_one", True)))
    mode = config["mode"]
    return cp if mode == "complete" else ap if mode == "arbitrary" else MixedMissingnessPolicy(cp, ap)


def policy_fingerprint(config: Mapping[str, object]) -> str:
    """Fingerprint policy configuration for manifest/run reproducibility."""
    import json
    return hashlib.sha256((POLICY_VERSION + json.dumps(config, sort_keys=True)).encode()).hexdigest()
