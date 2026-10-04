"""One-line diagram of k2a_course. Connections come from the model data."""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from tops.ps_models import k2a_course as model_data

# Manual positions make the electrical topology easy to read.
# These are drawing coordinates, not geographical coordinates.
POSITION = {
    "B5": (0, 0), "B6": (1, 0), "B7": (2, 0), "B8": (3, 0),
    "B9": (4, 0), "B10": (5, 0), "B11": (6, 0),
    "B1": (0, 1.8), "B2": (1, 1.8), "B4": (5, 1.8), "B3": (6, 1.8),
}
OUTAGE_LINE = "L7-8-1"  # Highlight only. The diagram shows the intact network.


def rows(table):
    """Turn each data row into a dictionary using the table header."""
    return [dict(zip(table[0], row)) for row in table[1:]]


def main():
    data = model_data.load()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    color = "#b33b27"
    # Shaded boxes identify the two areas. B8 is the corridor midpoint.
    for left, title in ((-0.45, "Area 1"), (3.55, "Area 2")):
        ax.add_patch(Rectangle((left, -1.45), 2.9, 4.3,
                              facecolor="#eef3f7", edgecolor="none", zorder=0))
        ax.text(left + 1.45, 3.12, title, ha="center", weight="bold")

    lines = rows(data["lines"])
    for line in lines:
        a, b = line["from_bus"], line["to_bus"]
        x1, y1 = POSITION[a]
        x2, y2 = POSITION[b]
        parallel = [m for m in lines if {m["from_bus"], m["to_bus"]} == {a, b}]
        offset = (parallel.index(line) - (len(parallel)-1)/2) * 0.42
        y = y1 + offset
        highlighted = line["name"] == OUTAGE_LINE
        ax.plot([x1, x2], [y, y], color=color if highlighted else "black", lw=1.5)
        ax.text((x1+x2)/2, y + (0.12 if offset >= 0 else -0.12),
                line["name"], ha="center", va="bottom" if offset >= 0 else "top",
                fontsize=8, color=color if highlighted else "black")

    for trafo in rows(data["transformers"]):
        x, top = POSITION[trafo["from_bus"]]
        _, bottom = POSITION[trafo["to_bus"]]
        ax.plot([x, x], [bottom, top], color="black", lw=1.2, zorder=1)
        for y in (0.80, 1.03):
            ax.add_patch(Circle((x, y), 0.17, facecolor="white", edgecolor="black", zorder=2))
        ax.text(x+0.23, 0.91, trafo["name"], fontsize=8, va="center")

    for bus in rows(data["buses"]):
        x, y = POSITION[bus["name"]]
        ax.plot([x, x], [y-0.29, y+0.29], color="black", lw=3, zorder=3)
        ax.text(x, y+0.34, bus["name"], ha="center", fontsize=9)

    for gen in rows(data["generators"]["GEN"]):
        x, y = POSITION[gen["bus"]]
        ax.plot([x, x], [y+0.29, y+0.57], color="black", lw=1.2)
        ax.add_patch(Circle((x, y+0.74), 0.17, facecolor="white", edgecolor="black"))
        ax.text(x, y+0.74, "~", ha="center", va="center", fontsize=14)
        label = gen["name"] + (" (PF slack)" if gen["bus"] == data["slack_bus"] else "")
        ax.text(x, y+1.00, label, ha="center", fontsize=8)

    for load in rows(data["loads"]):
        x, y = POSITION[load["bus"]]
        ax.plot([x, x-0.18], [y, y], color="black", lw=1.2)
        ax.annotate("", (x-0.18, y-0.85), (x-0.18, y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
        ax.text(x-0.18, y-1.03, f"Load at {load['bus']}", ha="center", fontsize=8)
        if load["bus"] == "B9":
            ax.text(x-0.18, y-1.25, "RMS load step here", ha="center", fontsize=8, color=color)

    # Shunts are drawn as capacitor banks, as in the supplied Kundur data.
    for shunt in rows(data.get("shunts", [["name", "bus"]])):
        x, y = POSITION[shunt["bus"]]
        ax.plot([x, x+0.23], [y, y], color="black", lw=1.2)
        x += 0.23
        ax.plot([x, x], [y, y-0.50], color="black", lw=1.2)
        for level in (y-0.50, y-0.60):
            ax.plot([x-0.12, x+0.12], [level, level], color="black", lw=1.2)
        ax.plot([x, x], [y-0.60, y-0.82], color="black", lw=1.2)
        ax.plot([x-0.10, x+0.10], [y-0.82, y-0.82], color="black")
        ax.text(x+0.15, y-0.62, shunt["name"], fontsize=8)

    ax.text(3, 0.55, "Tie-line corridor", ha="center", fontsize=9)
    ax.text(3, -1.75, "Red line: L7-8-1 removed in the N-1 study. All lines shown connected.",
            ha="center", fontsize=9, color=color)
    ax.set(xlim=(-0.6, 6.6), ylim=(-1.95, 3.4), aspect="equal")
    ax.axis("off")
    ax.set_title("Kundur two-area system", fontsize=12)
    fig.tight_layout()
    # Show the diagram. Use the plot window's Save button for the report.
    plt.show()


if __name__ == "__main__":
    main()
