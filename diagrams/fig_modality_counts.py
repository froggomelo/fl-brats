"""Compare how many modalities each subject keeps under the two missingness schemes."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import COUNT_COLOURS, INK, MUTED, SURFACE, apply_style, load_partitions, paper_style_partition, save

CURRENT = [("missing_uneven/alpha_0_001", "q = 0.001"), ("missing_uneven/alpha_0_1", "q = 0.1"),
           ("missing_uneven/alpha_0_2", "q = 0.2")]
PAPER = [(1.0, "α = 1"), (0.1, "α = 0.1"), (0.001, "α = 0.001")]
MIN_LABELLED = 0.12  # share; narrower segments are left to the legend


def count_shares(combinations: dict, n: int) -> dict[int, float]:
    shares = {k: 0.0 for k in COUNT_COLOURS}
    for combo, count in combinations.items():
        shares[len(combo.split("+"))] += count / n
    return shares


def draw(ax, clients: list[tuple[int, dict]], title: str) -> None:
    for row, (n, combinations) in enumerate(clients):
        left = 0.0
        for size in (4, 3, 2, 1):
            share = count_shares(combinations, n)[size]
            ax.barh(row, share, left=left, height=0.56, color=COUNT_COLOURS[size], edgecolor=SURFACE, linewidth=1.2)
            if share >= MIN_LABELLED:
                ax.text(left + share / 2, row, f"{share:.0%}", ha="center", va="center", fontsize=8,
                        color=SURFACE if size >= 2 else INK)
            left += share
    ax.set_yticks(range(len(clients)), [f"Client {k}" for k in range(len(clients))])
    ax.set_ylim(len(clients) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0 %", "50 %", "100 %"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=9.5)


def main() -> None:
    apply_style()
    partitions = load_partitions()
    sizes = [c["n"] for c in partitions["fedavg_uneven"]["clients"]]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharey=True)
    for ax, (name, label) in zip(axes[0], CURRENT):
        draw(ax, [(c["n"], c["combinations"]) for c in partitions[name]["clients"]], label)
    for ax, (alpha, label) in zip(axes[1], PAPER):
        draw(ax, list(zip(sizes, paper_style_partition(sizes, alpha))), label)
    fig.text(0.01, 0.985, "Modalities kept per subject, by client", fontsize=11, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.935, "Current partitions (q, each modality dropped independently)", fontsize=9, color=MUTED, va="top")
    fig.text(0.01, 0.48, "FedAMM procedure (α, per-client Dirichlet over 15 combinations), one simulated draw "
             "on the same client sizes", fontsize=9, color=MUTED, va="top")
    handles = [Patch(color=COUNT_COLOURS[k], label=f"{k} modalit{'y' if k == 1 else 'ies'}") for k in (4, 3, 2, 1)]
    fig.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0.01, -0.02), fontsize=8.5,
               handlelength=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), h_pad=3.2)
    save(fig, "modality_counts")
    plt.close(fig)


if __name__ == "__main__":
    main()
