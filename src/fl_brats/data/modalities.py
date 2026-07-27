"""Canonical BraTS modality names, masks, and deterministic missingness policies."""

from __future__ import annotations

import hashlib
import random
from itertools import combinations
from typing import Mapping, Protocol

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


def policy_from_config(config: Mapping[str, object]) -> MissingModalityPolicy:
    """Build the configured deterministic policy."""
    if not config.get("enabled", False) or config.get("mode", "none") == "none": return NoMissingnessPolicy()
    complete = dict(config.get("complete", {}))
    arbitrary = dict(config.get("arbitrary", {}))
    seed = int(config.get("seed", 42))
    cp = CompleteMissingnessPolicy(seed, complete.get("client_available_modalities"), int(complete.get("min_available_modalities", 1)), int(complete.get("max_available_modalities", 4)))
    ap = ArbitraryMissingnessPolicy(seed, arbitrary.get("per_modality_missing_probability", {m: 0.0 for m in MODALITIES}), arbitrary.get("combination_probabilities"), bool(config.get("require_at_least_one", True)))
    mode = config["mode"]
    return cp if mode == "complete" else ap if mode == "arbitrary" else MixedMissingnessPolicy(cp, ap)


def policy_fingerprint(config: Mapping[str, object]) -> str:
    """Fingerprint policy configuration for manifest/run reproducibility."""
    import json
    return hashlib.sha256((POLICY_VERSION + json.dumps(config, sort_keys=True)).encode()).hexdigest()
