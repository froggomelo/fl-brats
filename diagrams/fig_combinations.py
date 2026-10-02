"""Show each client's mix of the 15 modality combinations under both missingness schemes.

Identical rows mean every client sees the same mix. Rows dominated by one cell
mean each client concentrates on its own combination, the heterogeneity FedAMM
was designed for.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import (
    COMBINATIONS, FAINT, HEAT_HI, HEAT_LO, INK, MODALITIES, MODALITY_LABELS, MUTED, RULE, SURFACE,
    apply_style, combination_key, load_partitions, paper_style_partition, save,
)

PANELS = [
    ("current", "missing_uneven/alpha_0_1", "Current, q = 0.1"),
    ("current", "missing_uneven/alpha_0_2", "Current, q = 0.2"),
    ("paper", 1.0, "FedAMM procedure, α = 1"),
    ("paper", 0.1, "FedAMM procedure, α = 0.1"),
    ("paper", 0.001, "FedAMM procedure, α = 0.001"),
]
MIN_LABELLED = 0.05  # share; smaller cells are left to the colour scale


def shares(clients: list[tuple[int, dict]]) -> np.ndarray:
    grid = np.zeros((len(clients), len(COMBINATIONS)))
    for row, (n, combinations) in enumerate(clients):
        for col, combo in enumerate(COMBINATIONS):
            grid[row, col] = combinations.get(combination_key(combo), 0) / n
    return grid


def main() -> None:
    apply_style()
    partitions = load_partitions()
    sizes = [c["n"] for c in partitions["fedavg_uneven"]["clients"]]
    cmap = LinearSegmentedColormap.from_list("heat", [HEAT_LO, HEAT_HI])

    heights = [4] * len(PANELS) + [3.2]
    fig, axes = plt.subplots(len(PANELS) + 1, 1, figsize=(10, 11), sharex=True,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.5})
    for ax, (kind, source, title) in zip(axes, PANELS):
        if kind == "current":
            clients = [(c["n"], c["combinations"]) for c in partitions[source]["clients"]]
        else:
            clients = list(zip(sizes, paper_style_partition(sizes, source)))
        grid = shares(clients)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=0.7, aspect="auto")
        for (row, col), value in np.ndenumerate(grid):
            if value >= MIN_LABELLED:
                ax.text(col, row, f"{value:.0%}", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if value >= 0.3 else INK)
        ax.set_xticks(np.arange(len(COMBINATIONS) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(clients) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_yticks(range(len(clients)), [f"Client {k}" for k in range(len(clients))])
        for spine in ax.spines.values():
            spine.set_visible(False)
        complete = sum(n * g[-1] for n, g in zip(sizes, grid)) / sum(sizes)
        ax.set_title(f"{title}   ·   {complete:.0%} of subjects complete", fontsize=9.5)

    # Dot matrix naming the modalities in each column.
    key = axes[-1]
    for row, modality in enumerate(MODALITIES):
        for col, combo in enumerate(COMBINATIONS):
            present = modality in combo
            key.scatter(col, row, s=34, color=INK if present else SURFACE,
                        edgecolors=INK if present else RULE, linewidths=1, zorder=3)
    key.set_ylim(len(MODALITIES) - 0.5, -0.5)
    key.set_yticks(range(len(MODALITIES)), [MODALITY_LABELS[m] for m in MODALITIES])
    key.set_xticks(range(len(COMBINATIONS)), [str(len(c)) for c in COMBINATIONS], fontsize=8, color=FAINT)
    key.set_xlabel("Modalities in the combination", color=MUTED)
    key.set_xlim(-0.5, len(COMBINATIONS) - 0.5)
    for spine in key.spines.values():
        spine.set_visible(False)

    key.tick_params(which="minor", length=0)
    fig.subplots_adjust(top=0.885, bottom=0.05)
    fig.text(0.02, 0.955, "Share of each client's subjects in each modality combination", fontsize=11,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.937, "Current partitions give every client the same mix. The FedAMM procedure gives each "
             "client its own mix, more so as α falls.", fontsize=8.5, color=MUTED)
    fig.text(0.02, 0.922, "FedAMM rows are one simulated draw (seed 1024) on this study's client sizes. "
             "Colour runs from 0 to 70 %.", fontsize=8.5, color=MUTED)
    save(fig, "modality_combinations")
    plt.close(fig)


if __name__ == "__main__":
    main()
