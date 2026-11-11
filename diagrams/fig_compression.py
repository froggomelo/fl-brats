"""Draw how a client update is compressed for upload and how the server restores it, in black."""

import matplotlib.pyplot as plt

from common import apply_style, draw_arrow, draw_box, report_canvas, save

BLACK = "#000000"
BOX = dict(edge=BLACK, title_colour=BLACK, title_size=6.3)


def main() -> None:
    apply_style()
    fig, ax = report_canvas(3.5, 1.5)
    top, h = 11.5, 3.2
    # Each step is one bold two-line name.
    steps = [("Add\nresidual", 0.3, 5.6), ("Top-$\\boldsymbol{p}$\nselect", 7.1, 5.4),
             ("Quantise\n$\\boldsymbol{b}$ bits", 13.7, 6.4), ("Pack\n+ indices", 21.3, 6.2)]
    for title, x, w in steps:
        draw_box(ax, x, top, w, title, h=h, **BOX)
    for start, end in ((5.9, 7.1), (12.5, 13.7), (20.1, 21.3)):
        draw_arrow(ax, (start, top - h / 2), (end, top - h / 2), colour=BLACK)
    draw_box(ax, 28.7, top, 6.0, "Server\naverage", h=h, lw=0.9, **BOX)
    draw_arrow(ax, (27.5, top - h / 2), (28.7, top - h / 2), colour=BLACK)
    ax.text(0.3, 13.3, "Client $k$: update $\\Delta_k$ after local training", fontsize=6.3, color=BLACK, va="center")

    # Error feedback keeps what was not sent, on the client.
    bottom = top - h
    ax.plot([16.9, 16.9], [bottom, 5.2], color=BLACK, linewidth=0.7)
    ax.plot([16.9, 3.1], [5.2, 5.2], color=BLACK, linewidth=0.7)
    draw_arrow(ax, (3.1, 5.2), (3.1, bottom), colour=BLACK)
    ax.text(10.0, 4.3, "error feedback: what was not sent", fontsize=5.9, color=BLACK, va="top", ha="center")

    # The broadcast returns to the clients.
    ax.text(34.7, 4.3, "quantised broadcast\nto every client", fontsize=5.9, color=BLACK, va="top", ha="right")
    draw_arrow(ax, (31.7, bottom), (31.7, 4.5), colour=BLACK)

    # Trim the empty canvas above and below, keeping tenths of an inch per unit.
    low, high = 1.6, 14.2
    fig.set_size_inches(3.5, (high - low) / 10)
    ax.set_ylim(low, high)

    save(fig, "compression")
    plt.close(fig)


if __name__ == "__main__":
    main()
