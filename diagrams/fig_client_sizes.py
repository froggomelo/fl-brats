"""Draw how the 875 training subjects are divided between clients in each partition."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import CLIENT_COLOURS, GRID, INK, MUTED, SURFACE, apply_style, load_partitions, save

ROWS = [
    "fedavg_even",
    "fedavg_uneven",
    "fedavg_train_configs_uneven/k3",
    "fedavg_train_configs_uneven/k5",
    "fedavg_train_configs_uneven/k8",
]
MIN_LABELLED = 45  # subjects; narrower segments are left to the legend


def main() -> None:
    apply_style()
    partitions = load_partitions()
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    most_clients = 0
    for row, name in enumerate(ROWS):
        left = 0
        clients = partitions[name]["clients"]
        most_clients = max(most_clients, len(clients))
        for k, client in enumerate(clients):
            n = client["n"]
            ax.barh(row, n, left=left, height=0.52, color=CLIENT_COLOURS[k], edgecolor=SURFACE, linewidth=1.5)
            if n >= MIN_LABELLED:
                ax.text(left + n / 2, row, str(n), ha="center", va="center", fontsize=8.5,
                        color=SURFACE if k in (0, 6) else INK)
            left += n
    ax.set_yticks(range(len(ROWS)), [partitions[n]["label"] for n in ROWS])
    ax.invert_yaxis()
    ax.set_xlim(0, 880)
    ax.set_xticks(range(0, 876, 125))
    ax.set_xlabel("Training subjects")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Training subjects per client", pad=26)
    ax.text(0, 1.04, "Each partition divides the same 875 subjects. Uneven splits give each client a smaller share "
            "than the one before (50 / 25 / 15 / 10 % at K = 4).", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    handles = [Patch(color=CLIENT_COLOURS[k], label=f"Client {k}") for k in range(most_clients)]
    ax.legend(handles=handles, ncol=most_clients, loc="upper left", bbox_to_anchor=(0, -0.2),
              fontsize=8.5, handlelength=1.0, columnspacing=1.2)
    save(fig, "client_sizes")
    plt.close(fig)


if __name__ == "__main__":
    main()
