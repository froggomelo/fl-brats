"""Draw the Results figures of the report from the reported runs in data/results.json.

results.json holds the batch (RTX 3060) runs of the results page, one record per run
with its test metrics, traffic and forced-combination scores, plus the decoded-on-demand
qualitative slices. Three figures are drawn:

- tradeoff: test Dice against total traffic per run, all-modality U-Net settings;
- robustness: test Dice for each of the 15 modality combinations, per method;
- qualitative: one axial slice per test subject with the ground truth and several predictions.
"""

import base64
import json
import statistics as st
import zlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from common import (COMBINATIONS, INK, MODALITY_LABELS, MUTED, ROOT, RULE, SURFACE,
                    apply_style, combination_key, save)

RESULTS = ROOT / "data" / "results.json"

WT, TC, ET = "#1F9E89", "#D08A2E", "#C2334D"
GROUP_COLOURS = {"fedavg": "#5F6F7E", "config": "#2a78d6", "compress": "#eb6834", "broadcast": "#1baf7a"}


def load() -> dict:
    return json.loads(RESULTS.read_text())


def runs_of(data, family, variant):
    return [r for r in data["runs"] if r["family"] == family and r["variant"] == variant]


def mean_dice(data, family, variant) -> float:
    return st.mean(r["test"]["mean_dice"] for r in runs_of(data, family, variant))


def traffic_gb(data, family, variant) -> float:
    return st.mean((r["traffic"]["up"] + r["traffic"]["down"]) / 1e9 for r in runs_of(data, family, variant))


# label, family, variant, group, label position in data coordinates (traffic GB, Dice), alignment
TRADEOFF = [
    ("FedAvg", "fedavg_uneven", "baseline", "fedavg", (29, 0.8655), "left"),
    ("Even split", "fedavg_even", "baseline", "fedavg", (29, 0.857), "left"),
    ("$E$ = 5", "fedavg_train_configs_uneven", "e5", "config", (29, 0.8905), "left"),
    ("$E$ = 10", "fedavg_train_configs_uneven", "e10", "config", (13.5, 0.8915), "right"),
    ("$B$ = 4", "fedavg_train_configs_uneven", "b4", "config", (29, 0.848), "left"),
    ("$B$ = 8", "fedavg_train_configs_uneven", "b8", "config", (29, 0.8115), "left"),
    ("4-bit", "fedmpq_uneven", "homogeneous_4bit", "compress", (8.6, 0.8635), "right"),
    ("2/4/8-bit", "fedmpq_uneven", "heterogeneous_2_4_8bit", "compress", (8.6, 0.8555), "right"),
    ("$p$ = 0.01", "selective_transmission_uneven", "p_0_01", "compress", (8.6, 0.8435), "right"),
    ("$p$ = 0.1", "selective_transmission_uneven", "p_0_1", "compress", (15.5, 0.8465), "left"),
    ("$p$ = 0.25", "selective_transmission_uneven", "p_0_25", "compress", (17.5, 0.8525), "left"),
    ("$p$ = 0.25 + 2/4/8", "comm_eff_uneven", "best", "compress", (15.0, 0.8795), "left"),
    ("$p$ = 0.1 + 2/4/8", "comm_eff_uneven", "best_p_0_1", "compress", (12.0, 0.8725), "right"),
    ("+ 8-bit broadcast", "comm_eff_uneven", "best_p_0_1_downlink_8bit", "broadcast", (4.6, 0.8675), "left"),
    ("+ 4-bit", "comm_eff_uneven", "best_p_0_1_downlink_4bit", "broadcast", (3.1, 0.853), "left"),
    ("+ 2-bit", "comm_eff_uneven", "best_p_0_1_downlink_2bit", "broadcast", (2.0, 0.8665), "center"),
]
GROUP_LABELS = {"fedavg": "FedAvg U-Net", "config": "Training configuration", "compress": "Upload compression",
                "broadcast": "Quantised broadcast"}


def draw_tradeoff(data) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    for label, family, variant, group, (tx, ty), align in TRADEOFF:
        x, y = traffic_gb(data, family, variant), mean_dice(data, family, variant)
        seeds = len(runs_of(data, family, variant))
        ax.scatter(x, y, s=16, color=GROUP_COLOURS[group], zorder=3,
                   marker="o" if seeds > 1 else "D", edgecolor=SURFACE, linewidth=0.4)
        ax.annotate(label, (x, y), xytext=(tx, ty), textcoords="data", fontsize=5.2, color="black",
                    ha=align, va="center", bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"},
                    arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.4, "shrinkA": 1, "shrinkB": 2.5})
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
    ax.set_xlim(1.7, 60)
    ax.set_ylim(0.80, 0.895)
    ax.set_xlabel("Total traffic per run (GB, log scale)", fontsize=6.5, color="black")
    ax.set_ylabel("Test mean Dice", fontsize=6.5, color="black")
    ax.tick_params(which="both", labelsize=6, colors="black")
    ax.grid(color="black", linewidth=0.3, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("black")
    handles = [Patch(color=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLOURS.items()]
    legend = ax.legend(handles=handles, fontsize=5.2, loc="lower left", handlelength=0.9, borderaxespad=0.2,
                       frameon=True, framealpha=1, edgecolor="none", fancybox=False, labelcolor="black")
    legend.get_frame().set_linewidth(0)
    fig.tight_layout(pad=0.3)
    save(fig, "tradeoff")
    plt.close(fig)


ROBUST_PANELS = [
    ("Per-modality dropout, $q$ = 0.1", [
        ("Zero-filled U-Net", "fedavg_missing_modalities_uneven", "unet_zero_fill_alpha_0_1"),
        ("FedAvg-MM", "fedavg_missing_modalities_uneven", "alpha_0_1"),
        ("FedAMM-core", "fedamm_uneven", "alpha_0_1"),
        ("FedAvg-MM + compression", "fedavg_mm_comm_eff_uneven", "best_p_0_1"),
        ("FedAMM-core + compression", "fedamm_comm_eff_uneven", "best_p_0_1"),
    ]),
    ("Dirichlet, $\\alpha$ = 0.1", [
        ("Zero-filled U-Net", "fedavg_missing_modalities_uneven", "unet_zero_fill_dirichlet_alpha_0_1"),
        ("FedAvg-MM", "fedavg_missing_modalities_uneven", "dirichlet_alpha_0_1"),
        ("FedAMM-core", "fedamm_uneven", "dirichlet_alpha_0_1"),
        ("FedAvg-MM + compression", "fedavg_mm_comm_eff_uneven", "dirichlet_alpha_0_1"),
        ("FedAMM-core + compression", "fedamm_comm_eff_uneven", "dirichlet_alpha_0_1"),
    ]),
]


def combination_dice(data, family, variant) -> list[float]:
    runs = [r for r in runs_of(data, family, variant) if r.get("test_combinations")]
    return [st.mean(r["test_combinations"][combination_key(c)][0] for r in runs) for c in COMBINATIONS]


def short_combo(combo) -> str:
    return "\n".join(MODALITY_LABELS[m] for m in combo)


def draw_robustness(data) -> None:
    # A multi-hue ramp over the observed range, so nearby Dice values still differ in colour.
    cmap = plt.get_cmap("YlGnBu")
    vmin, vmax = 0.1, 0.9
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.3), gridspec_kw={"hspace": 0.55})
    for ax, (title, methods) in zip(axes, ROBUST_PANELS):
        grid = np.array([combination_dice(data, f, v) for _, f, v in methods])
        ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                # Black or white, whichever contrasts more with the cell (WCAG relative luminance).
                lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                       for c in cmap((value - vmin) / (vmax - vmin))[:3]]
                lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
                light = (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05)
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.6,
                        color="black" if light else SURFACE)
        ax.set_yticks(range(len(methods)), [m for m, _, _ in methods], fontsize=6.2, color="black")
        if ax is axes[-1]:
            ax.set_xticks(range(len(COMBINATIONS)), [short_combo(c) for c in COMBINATIONS], fontsize=5.2, color="black")
        else:
            ax.set_xticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        means = grid.mean(axis=1)
        for i, value in enumerate(means):
            ax.text(len(COMBINATIONS) - 0.4, i, f"{value:.3f}", ha="left", va="center", fontsize=5.8, color="black")
        ax.text(len(COMBINATIONS) - 0.35, -0.85, "Mean", ha="left", va="center", fontsize=5.8, color="black")
        ax.set_title(title, fontsize=7, pad=8, color="black")
    fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.17)
    save(fig, "robustness")
    plt.close(fig)


QUAL_METHODS = [
    ("FedAvg", "fedavg_uneven/baseline"),
    ("Compressed", "comm_eff_uneven/best"),
    ("FedAMM-core, $q$ = 0.1", "fedamm_uneven/alpha_0_1"),
]
QUAL_EXAMPLES = ["typical", "hard", "missing"]


def decode(text: str, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(text)), np.uint8).reshape(shape)


def label_image(mask: np.ndarray) -> np.ndarray:
    """Map the bit-packed regions (bit 7 WT, bit 6 TC, bit 5 ET) to 0 background, 1 oedema, 2 non-enhancing core, 3 enhancing."""
    out = np.zeros(mask.shape, np.uint8)
    out[(mask & 128) > 0] = 1
    out[(mask & 64) > 0] = 2
    out[(mask & 32) > 0] = 3
    return out


def tight_box(examples) -> list[tuple[slice, slice]]:
    boxes = []
    for ex in examples:
        truth = decode(ex["truth"], ex["shape"]) > 0
        ys, xs = np.nonzero(truth)
        cy, cx, half = int(ys.mean()), int(xs.mean()), 46
        y0 = int(np.clip(cy - half, 0, ex["shape"][0] - 2 * half))
        x0 = int(np.clip(cx - half, 0, ex["shape"][1] - 2 * half))
        boxes.append((slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half)))
    return boxes


def draw_qualitative(data) -> None:
    q = data["qualitative"]
    examples = [next(e for e in q["examples"] if e["role"] == role) for role in QUAL_EXAMPLES]
    overlay = ListedColormap([(0, 0, 0, 0), WT, TC, ET])
    columns = [("Ground truth", "truth")] + QUAL_METHODS
    fig, axes = plt.subplots(len(examples), len(columns), figsize=(3.5, 3.15),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.22})
    for row, (ex, box) in enumerate(zip(examples, tight_box(examples))):
        image = decode(ex["images"]["flair"], ex["shape"])[box]
        for col, (title, key) in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(image, cmap="gray", interpolation="nearest")
            if key == "truth":
                ax.imshow(label_image(decode(ex["truth"], ex["shape"])[box]), cmap=overlay, vmin=0, vmax=3,
                          interpolation="nearest", alpha=0.75)
            elif key is not None:
                pred = label_image(decode(ex["predictions"][key], ex["shape"])[box])
                ax.imshow(pred, cmap=overlay, vmin=0, vmax=3, interpolation="nearest", alpha=0.75)
                dice = q["methods"][key]["dice"][ex["subject"]]
                ax.set_xlabel(f"{st.mean(dice):.2f}", fontsize=5.6, labelpad=1.5, color=INK)
                if 0 in ex["masks"][key]:
                    absent = [MODALITY_LABELS[m] for m, a in zip(("t1", "t1ce", "t2", "flair"), ex["masks"][key])
                              if not a]
                    ax.text(0.03, 0.97, "no " + ", ".join(absent), transform=ax.transAxes, fontsize=4.8,
                            color=SURFACE, va="top")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(RULE)
                side.set_linewidth(0.4)
            if row == 0:
                ax.set_title(title, fontsize=5.8, fontweight="normal", loc="center", pad=3)
        axes[row, 0].set_ylabel(ex["role"].capitalize(), fontsize=6, color=MUTED)
    handles = [Patch(color=WT, label="Oedema"), Patch(color=TC, label="Non-enhancing core"),
               Patch(color=ET, label="Enhancing tumour")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.94, bottom=0.08)
    save(fig, "qualitative")
    plt.close(fig)


def main() -> None:
    apply_style()
    data = load()
    draw_tradeoff(data)
    draw_robustness(data)
    draw_qualitative(data)


if __name__ == "__main__":
    main()
