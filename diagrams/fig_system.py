"""Draw the experimental system at one-column report width, in black, with the client-server exchange in red."""

import matplotlib.pyplot as plt

from common import apply_style, draw_arrow, draw_box, load_partitions, report_canvas, save

TOTAL, VALIDATION, TEST = 1251, 187, 189
BLACK = "#000000"
RED = "#C00000"
BOX = dict(edge=BLACK, lw=0.6, line_colour=BLACK, title_colour=BLACK)
WIDTH = 3.5  # inches, one IEEE column
LANE = 1.0   # x of the held-out line, left of the clients and the server row
LEFT = 2.0   # left edge of the clients and the server row
RIGHT = 34.9


def row(widths: list[float]) -> list[float]:
    """Return the left edges of boxes spread evenly between LEFT and RIGHT."""
    gap = (RIGHT - LEFT - sum(widths)) / (len(widths) - 1)
    xs, x = [], LEFT
    for w in widths:
        xs.append(x)
        x += w + gap
    return xs


def main() -> None:
    apply_style()
    sizes = [c["n"] for c in load_partitions()["fedavg_uneven"]["clients"]]
    train = sum(sizes)
    fig, ax = report_canvas(WIDTH, 2.6)
    top = 25.6

    # Data, split, preprocessing, and partitioning across the top.
    data_w, prep_w, part_w = 9.6, 10.6, 11.8
    data_x, prep_x = 0.3, 0.3 + data_w + 1.25
    part_x = RIGHT - part_w
    data_bottom = draw_box(ax, data_x, top, data_w, "BraTS 2021",
                           [f"{TOTAL:,} subjects", f"Training {train}", f"Validation {VALIDATION}", f"Test {TEST}"],
                           **BOX)
    draw_box(ax, prep_x, top, prep_w, "Preprocessing",
             ["Resample to 1 mm", "Crop to foreground", "Z-score per modality", "Labels to WT, TC, ET"], **BOX)
    part_bottom = draw_box(ax, part_x, top, part_w, "Partitioning",
                           [f"{train} training subjects", "K clients, uneven shares", "Modalities removed by:",
                            "q, per-modality dropout", "α, Dirichlet per client", "or one scan per client"], **BOX)
    arrow_y = top - 3.2
    draw_arrow(ax, (data_x + data_w, arrow_y), (prep_x, arrow_y), colour=BLACK)
    draw_arrow(ax, (prep_x + prep_w, arrow_y), (part_x, arrow_y), colour=BLACK)

    # Clients in one row, each fed from the partitioning box.
    client_w = 7.5
    client_top = part_bottom - 2.2
    client_xs = row([client_w] * len(sizes))
    fan = (part_x + part_w / 2, part_bottom)
    for k, (x, n) in enumerate(zip(client_xs, sizes)):
        client_bottom = draw_box(ax, x, client_top, client_w, f"Client {k}", [f"{n} subjects"], h=3.4, **BOX)
        draw_arrow(ax, fan, (x + client_w / 2, client_top), colour=BLACK)

    # Exchange with the server, once per round, over a shared bus.
    widths = [6.4, 7.2, 9.9, 6.6]
    sx, vx, bx, tx = row(widths)
    server_top = client_bottom - 3.4
    bus = client_bottom - 1.5
    centres = [x + client_w / 2 for x in client_xs]
    for c in centres:
        draw_arrow(ax, (c, bus), (c, client_bottom), colour=RED, both=True, lw=0.9)
    server_mid = sx + widths[0] / 2
    ax.plot([min(centres[0], server_mid), centres[-1]], [bus, bus], color=RED, linewidth=0.9, solid_capstyle="butt")
    draw_arrow(ax, (server_mid, bus), (server_mid, server_top), colour=RED, both=True, lw=0.9)
    label_y = (bus + server_top) / 2
    ax.text(server_mid + 0.8, label_y, "↓ local updates     ↑ global model", fontsize=6.2, color=RED,
            ha="left", va="center")

    # Server, validation, checkpoint selection, and testing along the bottom.
    h = 4.2
    draw_box(ax, sx, server_top, widths[0], "Server", ["Flower"], h=h, **BOX)
    draw_box(ax, vx, server_top, widths[1], "Validate", [f"{VALIDATION} subjects", "every round"], h=h, **BOX)
    draw_box(ax, bx, server_top, widths[2], "Best checkpoint", ["At least", "100 rounds"], h=h, **BOX)
    test_bottom = draw_box(ax, tx, server_top, widths[3], "Test once", [f"{TEST} subjects"], h=h, **BOX)
    mid_y = server_top - h / 2
    for (x0, w0), x1 in zip(zip([sx, vx, bx], widths), [vx, bx, tx]):
        draw_arrow(ax, (x0 + w0, mid_y), (x1, mid_y), colour=BLACK)

    # Held-out subjects reach evaluation without passing through a client.
    held_y = test_bottom - 1.2
    test_mid = tx + widths[3] / 2
    dashes = (0, (3, 2))
    ax.plot([LANE, LANE], [data_bottom, held_y], color=BLACK, linewidth=0.6, linestyle=dashes)
    ax.plot([LANE, test_mid], [held_y, held_y], color=BLACK, linewidth=0.6, linestyle=dashes)
    draw_arrow(ax, (test_mid, held_y), (test_mid, test_bottom), colour=BLACK, dashed=True)
    ax.text(LEFT, held_y - 0.3, "held-out subjects, never seen by a client", fontsize=6.2, color=BLACK,
            ha="left", va="top")

    # Trim the empty canvas, keeping tenths of an inch per unit.
    low = held_y - 1.6
    fig.set_size_inches(WIDTH, (top + 0.3 - low) / 10)
    ax.set_ylim(low, top + 0.3)

    save(fig, "system")
    plt.close(fig)


if __name__ == "__main__":
    main()
