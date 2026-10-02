"""Draw the data pipeline and federated training loop shared by every run."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import ACCENT, INK, MUTED, NOTE, RULE, SURFACE, apply_style, load_partitions, save

# Cohort sizes of the seeded 70/15/15 subject-level split (docs/dataset_splits.md).
TOTAL, VALIDATION, TEST = 1251, 187, 189


LINE = 3.1


def box(ax, x, top, w, title, lines=(), *, fill=SURFACE, edge=RULE, lw=1.0):
    """Draw a box hanging from `top`, sized to its text, and return its bottom edge."""
    h = 6.5 + LINE * len(lines)
    y = top - h
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=fill, edgecolor=edge, linewidth=lw,
        )
    )
    ax.text(x + 1.4, top - 2.0, title, fontsize=9.5, fontweight="bold", color=INK, va="top")
    for i, line in enumerate(lines):
        ax.text(x + 1.4, top - 5.6 - LINE * i, line, fontsize=8.2, color=MUTED, va="top")
    return y


def arrow(ax, start, end, *, colour=INK, both=False, lw=1.0, connection="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=9,
            color=colour, linewidth=lw, shrinkA=0, shrinkB=0, connectionstyle=connection,
        )
    )


def main() -> None:
    apply_style()
    sizes = [c["n"] for c in load_partitions()["fedavg_uneven"]["clients"]]
    train = sum(sizes)

    fig, ax = plt.subplots(figsize=(16.3, 7.2))
    ax.set_xlim(0, 163)
    ax.set_ylim(0, 72)
    ax.set_aspect("equal")
    ax.axis("off")

    # Data and split.
    bottom = box(ax, 1, 66, 25, "BraTS 2021", [f"{TOTAL:,} subjects", "T1, T1ce, T2, FLAIR, labels"])
    arrow(ax, (13.5, bottom), (13.5, 49))
    bottom = box(ax, 1, 49, 25, "Subject-level split", ["Seeded 70 / 15 / 15", f"Training {train}",
                                                        f"Validation {VALIDATION}", f"Test {TEST}"])
    arrow(ax, (13.5, bottom), (13.5, 26))
    box(ax, 1, 26, 25, "Held-out cohorts", [f"Validation {VALIDATION}, every round",
                                            f"Test {TEST}, once", "Never seen by a client"], fill=NOTE)

    # Preprocessing.
    arrow(ax, (26, 40), (30, 40))
    box(ax, 30, 66, 29, "Preprocessing, all cohorts", [
        "Resample to 1 mm voxels",
        "Skull-stripped volumes",
        "Z-score per modality",
        "Crop to the foreground",
        "Labels to WT, TC, ET",
        "",
        "Deterministic steps cached",
        "on each node's local SSD",
    ])

    # Partitioning.
    arrow(ax, (59, 50), (63, 50))
    shares = " / ".join(f"{round(100 * n / train)}" for n in sizes)
    box(ax, 63, 66, 32, f"Partition {train} training subjects", [
        f"Client shares {shares} %",
        ", ".join(str(n) for n in sizes) + " subjects",
        "Sweeps at K = 3, 5, and 8",
        "",
        "Missing modalities, per subject",
        "q   each modality dropped",
        "     independently, same mix",
        "     at every client (current)",
        "α   per-client Dirichlet over",
        "     15 combinations (FedAMM)",
    ])

    # Clients.
    mids = []
    for k, (n, top) in enumerate(zip(sizes, [66, 54, 42, 30])):
        bottom = box(ax, 100, top, 24, f"Client {k}, {n} subjects", ["RTX 3060, E epochs, batch B"])
        mids.append((top + bottom) / 2)
        arrow(ax, (95, 47), (100, mids[-1]))
    ax.text(100, bottom - 1.6, "Random crops and flips run here", fontsize=8.2, color=MUTED, va="top")

    # Exchange with the server.
    bus_x = 128
    for y in mids:
        arrow(ax, (124, y), (bus_x, y), colour=ACCENT, lw=1.6, both=True)
    ax.plot([bus_x, bus_x], [mids[-1], mids[0]], color=ACCENT, linewidth=1.6, solid_capstyle="butt")
    arrow(ax, (bus_x, 58), (133, 58), colour=ACCENT, lw=1.6)
    ax.text(bus_x, 68.5, "updates up, model down, every round", fontsize=8.2, color=ACCENT, ha="center")

    bottom = box(ax, 133, 66, 29, "Server (Flower)", ["FedAvg or FedAMM aggregation",
                                                      "Upload compression, optional",
                                                      "U-Net, FedAMM net, SAM 1 or 2"], edge=INK, lw=1.3)
    arrow(ax, (147.5, bottom), (147.5, 47))
    bottom = box(ax, 133, 47, 29, "Validate the global model", [f"{VALIDATION} subjects, every round"])
    arrow(ax, (147.5, bottom), (147.5, 34))
    bottom = box(ax, 133, 34, 29, "Keep the best checkpoint", ["At least 100 rounds, then stop",
                                                               "after 50 without improvement"])
    arrow(ax, (147.5, bottom), (147.5, 18))
    box(ax, 133, 18, 29, f"Test once, {TEST} subjects", ["Natural missing modalities",
                                                         "All 15 modality combinations",
                                                         "Dice and HD95 for WT, TC, ET"])

    # Held-out subjects reach evaluation without passing through a client.
    ax.add_patch(
        FancyArrowPatch(
            (26, 11), (133, 11), arrowstyle="-|>", mutation_scale=9, color=MUTED,
            linewidth=1.0, linestyle=(0, (4, 3)), shrinkA=0, shrinkB=0,
        )
    )
    ax.text(79.5, 12.2, "held-out subjects, preprocessed identically", fontsize=8.2, color=MUTED, ha="center")

    save(fig, "pipeline")
    plt.close(fig)


if __name__ == "__main__":
    main()
