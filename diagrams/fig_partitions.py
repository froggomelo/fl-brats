"""Draw client sizes (one column) and the modalities kept per client (two columns) as separate report figures, from the real partitions."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import CLIENT_COLOURS, COUNT_COLOURS, SURFACE, apply_style, load_partitions, save

BLACK = "#000000"


def tint(colour: str, amount: float = 0.4) -> str:
    """Mix a colour with white so black text on it stays readable."""
    rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


# Lighter versions of the shared palettes, so every number can be black.
CLIENT_FILLS = [tint(c) for c in CLIENT_COLOURS]
# Steps of the robustness heatmap's YlGnBu scale, dark enough to tell apart and light enough for black text.
COUNT_FILLS = {1: "#edf8b1", 2: "#a1dab4", 3: "#41b6c4", 4: "#2c7fb8"}

# Sizes of the K = 4-shaped client sweeps (docs/data.md).
LORENZ = {"K = 3": [511, 233, 131], "K = 5": [351, 218, 140, 96, 70],
          "K = 8": [223, 218, 109, 109, 65, 65, 43, 43]}
PANELS = [("missing_uneven/alpha_0_1", "$q$ = 0.1"), ("missing_uneven/alpha_0_2", "$q$ = 0.2"),
          ("missing_dirichlet/alpha_1", "$\\alpha$ = 1"), ("missing_dirichlet/alpha_0_1", "$\\alpha$ = 0.1"),
          ("missing_dirichlet/alpha_0_001", "$\\alpha$ = 0.001")]
MIN_SUBJECTS = 40
MIN_SHARE = 0.15


def draw_sizes(ax, partitions) -> None:
    rows = [("Even", [c["n"] for c in partitions["fedavg_even"]["clients"]]),
            ("Uneven", [c["n"] for c in partitions["fedavg_uneven"]["clients"]])] + list(LORENZ.items())
    for row, (_, sizes) in enumerate(rows):
        left = 0
        for k, n in enumerate(sizes):
            ax.barh(row, n, left=left, height=0.62, color=CLIENT_FILLS[k], edgecolor=SURFACE, linewidth=0.6)
            if n >= MIN_SUBJECTS:
                ax.text(left + n / 2, row, str(n), ha="center", va="center", fontsize=6.0,
                        color=BLACK)
            left += n
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=7.0, color=BLACK)
    ax.invert_yaxis()
    ax.set_xlim(0, 875)
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def draw_mix(ax, clients, title: str, first: bool) -> None:
    for row, client in enumerate(clients):
        shares = {size: 0.0 for size in COUNT_COLOURS}
        for combo, count in client["combinations"].items():
            shares[len(combo.split("+"))] += count / client["n"]
        left = 0.0
        for size in (4, 3, 2, 1):
            ax.barh(row, shares[size], left=left, height=0.62, color=COUNT_FILLS[size], edgecolor=SURFACE,
                    linewidth=0.5)
            if shares[size] >= MIN_SHARE:
                ax.text(left + shares[size] / 2, row, f"{shares[size]:.0%}", ha="center", va="center", fontsize=6.0,
                        color=BLACK)
            left += shares[size]
        # Outline the whole bar so the palest segment still shows where the bar ends.
        ax.barh(row, 1.0, height=0.62, fill=False, edgecolor=BLACK, linewidth=0.4, zorder=3)
    ax.set_yticks(range(len(clients)), [f"C{k}" for k in range(len(clients))] if first else [], fontsize=7.0,
                  color=BLACK)
    ax.set_ylim(len(clients) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=7.5, color=BLACK, pad=4)


def main() -> None:
    apply_style()
    partitions = load_partitions()

    # Client sizes, one column wide.
    fig, ax = plt.subplots(figsize=(3.45, 1.45))
    draw_sizes(ax, partitions)
    fig.tight_layout(pad=0.2)
    save(fig, "partition_sizes")
    plt.close(fig)

    # Modalities kept per client, two columns wide.
    fig = plt.figure(figsize=(7.16, 1.55))
    grid = fig.add_gridspec(1, len(PANELS), wspace=0.12, left=0.035, right=0.995, top=0.87, bottom=0.2)
    for index, (name, label) in enumerate(PANELS):
        draw_mix(fig.add_subplot(grid[0, index]), partitions[name]["clients"], label, index == 0)
    handles = [Patch(facecolor=COUNT_FILLS[k], edgecolor=BLACK, linewidth=0.4, label=f"{k} scan{'' if k == 1 else 's'}") for k in (4, 3, 2, 1)]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.515, -0.02), fontsize=7.0,
               handlelength=1.0, columnspacing=1.5, labelcolor=BLACK)
    save(fig, "partition_modalities")
    plt.close(fig)


if __name__ == "__main__":
    main()
