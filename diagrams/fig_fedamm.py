"""Draw the modality-specific network in black, outlining what FedAMM adds to FedAvg-MM in the accent colour."""

import matplotlib.pyplot as plt

from common import ACCENT, apply_style, draw_arrow, draw_box, report_canvas, save

BLACK = "#000000"
BOX = dict(edge=BLACK, title_colour=BLACK, line_colour=BLACK)


def main() -> None:
    apply_style()
    fig, ax = report_canvas(3.5, 2.55)

    # Inputs and encoders; FLAIR is absent for this subject.
    rows = [("T1", 25.0), ("T1ce", 20.0), ("T2", 15.0), ("FLAIR", 10.0)]
    for name, top in rows:
        absent = name == "FLAIR"
        ax.text(0.3, top - 1.5, name, fontsize=6.6, color=BLACK, va="center")
        draw_arrow(ax, (4.1, top - 1.5), (5.5, top - 1.5), colour=BLACK, dashed=absent)
        draw_box(ax, 5.5, top, 6.6, f"$E_{{\\mathrm{{{name}}}}}$", h=3.0, title_size=6.6, **BOX)
        if absent:
            ax.patches[-1].set_linestyle((0, (3, 2)))
        # Head at the midpoint, so the four lines do not crowd their heads where they meet the box.
        start, end = (12.1, top - 1.5), (15.0, 15.0)
        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.plot([start[0], end[0]], [start[1], end[1]], color=BLACK, linewidth=0.7,
                linestyle=(0, (3, 2)) if absent else "-", solid_capstyle="butt")
        back = (middle[0] - (end[0] - start[0]) * 0.05, middle[1] - (end[1] - start[1]) * 0.05)
        draw_arrow(ax, back, middle, colour=BLACK)
    ax.text(5.5, 5.6, "absent: skipped", fontsize=5.8, color=BLACK, va="top")

    # Fusion, decoder and output.
    draw_box(ax, 15.0, 18.0, 5.8, "Masked\nmean", h=5.6, title_size=6.4, **BOX)
    draw_arrow(ax, (20.8, 15.2), (22.4, 15.2), colour=BLACK)
    draw_box(ax, 22.4, 18.0, 6.2, "Shared\ndecoder", h=5.6, title_size=6.4, **BOX)
    draw_arrow(ax, (28.6, 15.2), (30.2, 15.2), colour=BLACK)
    ax.text(30.3, 16.3, "WT", fontsize=6.2, color=BLACK, va="center")
    ax.text(30.3, 15.2, "TC", fontsize=6.2, color=BLACK, va="center")
    ax.text(30.3, 14.1, "ET", fontsize=6.2, color=BLACK, va="center")

    # What FedAMM adds: blue outlines, arrows and losses, with black text in the boxes.
    draw_box(ax, 13.0, 25.0, 21.5, "Prototypes per region", ["fused (teacher), unimodal (student)"], h=3.9,
             edge=ACCENT, title_colour=BLACK, line_colour=BLACK, title_size=6.4, line_size=5.9)
    draw_arrow(ax, (17.9, 18.0), (17.9, 21.1), colour=ACCENT)
    ax.text(18.5, 19.6, "$\\mathcal{L}_{\\mathrm{dist}}$", fontsize=6.6, color=ACCENT, va="center")
    draw_box(ax, 15.0, 8.0, 19.5, "Server", ["prototype library per combination",
                                             "encoders weighted by modality count"], h=4.6,
             edge=ACCENT, title_colour=BLACK, line_colour=BLACK, title_size=6.4, line_size=5.9)
    draw_arrow(ax, (17.9, 12.4), (17.9, 8.0), colour=ACCENT, both=True)
    ax.text(18.5, 10.3, "$\\mathcal{L}_{\\mathrm{align}}$", fontsize=6.6, color=ACCENT, va="center")

    # Trim the empty canvas below the server box, keeping tenths of an inch per unit.
    low = 2.6
    fig.set_size_inches(3.5, 2.55 - low / 10)
    ax.set_ylim(low, 25.5)

    save(fig, "fedamm")
    plt.close(fig)


if __name__ == "__main__":
    main()
