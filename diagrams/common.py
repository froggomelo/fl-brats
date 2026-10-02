"""Shared style, data loading, and the FedAMM partition procedure for the diagrams.

Colours match the light theme of the results page. The client palette is the
page's validated categorical order, and the modality-count palette is its
ordinal blue ramp.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "partitions.json"
OUT = ROOT / "out"

INK = "#16202A"
MUTED = "#5A6774"
FAINT = "#8A96A2"
RULE = "#D6DDE4"
GRID = "#E6EBF0"
SURFACE = "#FFFFFF"
NOTE = "#F2F4F7"
ACCENT = "#3A6EA5"
HEAT_LO = "#F7F3EE"
HEAT_HI = "#2B4C8C"

# Clients 0 to 7, in fixed order.
CLIENT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Subjects with 1, 2, 3, and 4 modalities, light to dark.
COUNT_COLOURS = {1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 4: "#0d366b"}

MODALITIES = ("t1", "t1ce", "t2", "flair")
MODALITY_LABELS = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}

# The 15 non-empty combinations, ordered by size and then by modality order.
COMBINATIONS = sorted(
    (tuple(m for i, m in enumerate(MODALITIES) if bits >> i & 1) for bits in range(1, 16)),
    key=lambda c: (len(c), [MODALITIES.index(m) for m in c]),
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "text.color": INK,
            "axes.edgecolor": RULE,
            "axes.labelcolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_partitions() -> dict:
    return json.loads(DATA.read_text())


def combination_key(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def combination_label(combo: tuple[str, ...]) -> str:
    return " + ".join(MODALITY_LABELS[m] for m in combo)


def paper_style_partition(sizes: list[int], alpha: float, seed: int = 1024) -> list[Counter]:
    """Assign modality combinations the way the FedAMM release does.

    Mirrors generate_mask_distribution_only_maskid in the FedAMM code
    (utils/preprocessing/generate_dir_imb_mr.py). Every client draws its own
    Dirichlet(alpha) proportions over the 15 combinations, receives one subject
    of each combination, and draws the rest of its subjects from its
    proportions. The release uses four even clients and seed 1024. Here the
    client sizes are this study's.
    """
    rng = np.random.RandomState(seed)
    proportions = rng.dirichlet(np.repeat(alpha, len(COMBINATIONS)), size=len(sizes))
    clients = []
    for size, p in zip(sizes, proportions):
        ids = list(range(len(COMBINATIONS)))
        ids += list(rng.choice(len(COMBINATIONS), size=size - len(COMBINATIONS), p=p / p.sum()))
        clients.append(Counter(combination_key(COMBINATIONS[i]) for i in ids))
    return clients


def save(fig, name: str) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f"{name}.png", OUT / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths
