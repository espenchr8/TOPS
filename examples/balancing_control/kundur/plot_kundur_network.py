"""One-line diagram of the Kundur two-area system for one IEEE column.

Buses, lines, transformers, generators, loads and capacitor banks are read
from the model data, including line lengths, load MW and capacitor MVAr.
Only the drawing positions are set by hand.
Saves kundur_sld.pdf (vector, for LaTeX) and kundur_sld.png next to the script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from tops.ps_models import k2a as model_data

# Drawing coordinates, not geographical coordinates.
HV_X = {"B5": 0, "B6": 1, "B7": 2, "B8": 3, "B9": 4, "B10": 5, "B11": 6}
GEN_Y = 0.8          # height of the generator buses B1-B4
OUTAGE_LINE = "L7-8-1"   # Highlighted only. All lines remain connected.
LOAD_STEP_BUS = "B9"
LOAD_STEP_MW = 100
RED = "#b03a2e"
LW = 0.9

plt.rcParams.update({"font.size": 6, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif"})


def rows(table):
    """Turn each data row into a dictionary using the table header."""
    return [dict(zip(table[0], row)) for row in table[1:]]


def hline(ax, x0, x1, y, color="k", lw=LW):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")


def vline(ax, x, y0, y1, color="k", lw=LW):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="butt")


def main():
    data = model_data.load()
    fig, ax = plt.subplots(figsize=(3.5, 1.85))

    # Shaded boxes identify the two areas, names in the bottom corners.
    for left in (-0.5, 3.65):
        ax.add_patch(Rectangle((left, -1.27), 2.85, 2.69,
                               color="#e8eef5", zorder=0))
    ax.text(-0.42, -1.22, "Area 1", ha="left", va="bottom", weight="bold")
    ax.text(6.42, -1.22, "Area 2", ha="right", va="bottom", weight="bold")

    # Lines. Parallel circuits are drawn side by side.
    lines = rows(data["lines"])
    corridor_km = None
    for line in lines:
        a, b = line["from_bus"], line["to_bus"]
        parallel = [m for m in lines if {m["from_bus"], m["to_bus"]} == {a, b}]
        y = (parallel.index(line) - (len(parallel) - 1) / 2) * 0.26
        out = line["name"] == OUTAGE_LINE
        hline(ax, HV_X[a], HV_X[b], y, RED if out else "k",
              LW * 1.6 if out else LW)
        mid = (HV_X[a] + HV_X[b]) / 2
        if len(parallel) == 1:
            ax.text(mid, 0.06, f"{line['length']:g} km", ha="center")
        else:
            corridor_km = line["length"]
            if OUTAGE_LINE in [m["name"] for m in parallel]:  # name this pair
                ax.text(mid, y + (0.06 if y > 0 else -0.05), line["name"],
                        ha="center", va="bottom" if y > 0 else "top",
                        fontsize=5.5, color=RED if out else "k")
    if corridor_km is not None:
        ax.text(3.0, 0.78, f"tie corridor\n{corridor_km:g} km per line",
                ha="center", va="center")

    # High-voltage bus bars (B7, B8, B9 labelled above, the rest below).
    for name, x in HV_X.items():
        vline(ax, x, -0.24, 0.24, lw=2.4)
        up = name in ("B7", "B8", "B9")
        ax.text(x, 0.30 if up else -0.30, name, ha="center",
                va="bottom" if up else "top")

    # Step-up transformers between generator bus and high-voltage bus.
    trafo_of = {t["from_bus"]: t["to_bus"] for t in rows(data["transformers"])}

    # Generators on their own bus, drawn above the transformer.
    for gen in rows(data["generators"]["GEN"]):
        lv = gen["bus"]
        x = HV_X[trafo_of[lv]]
        vline(ax, x, 0.24, 0.40)
        ax.add_patch(Circle((x, 0.47), 0.09, fill=False, lw=LW))
        ax.add_patch(Circle((x, 0.59), 0.09, fill=False, lw=LW))
        vline(ax, x, 0.68, 1.0)
        hline(ax, x - 0.17, x + 0.17, GEN_Y, lw=2.0)
        ax.text(x - 0.2, GEN_Y, lv, ha="right", va="center")
        ax.add_patch(Circle((x, 1.1), 0.11, fill=False, lw=LW))
        ax.text(x, 1.1, "~", ha="center", va="center", fontsize=7)
        slack = " (slack)" if lv == data["slack_bus"] else ""
        ax.text(x, 1.23, gen["name"] + slack, ha="center", va="bottom")

    # Loads: own connection, out from the bus bar at 90 degrees, then down.
    # Area 1 loads go left, area 2 loads go right, away from the corridor.
    for load in rows(data["loads"]):
        x = HV_X[load["bus"]]
        side = -1 if x < 3 else 1
        xl = x + side * 0.28
        hline(ax, x, xl, -0.21)
        ax.annotate("", xy=(xl, -0.75), xytext=(xl, -0.21),
                    arrowprops=dict(arrowstyle="-|>", lw=LW, color="k",
                                    mutation_scale=6))
        ax.text(xl, -0.79, f"{load['P']:g} MW", ha="center", va="top")
        if load["bus"] == LOAD_STEP_BUS:
            ax.text(xl, -1.0, f"+{LOAD_STEP_MW:g} MW step", ha="center",
                    va="top", color=RED, fontsize=5.5)

    # Capacitor banks: own connection straight down from the bus bar.
    for shunt in rows(data.get("shunts", [["name", "bus"]])):
        x = HV_X[shunt["bus"]]
        side = -1 if x < 3 else 1     # label towards the corridor
        vline(ax, x, -0.24, -0.50)
        hline(ax, x - 0.08, x + 0.08, -0.50)
        hline(ax, x - 0.08, x + 0.08, -0.56)
        vline(ax, x, -0.56, -0.66)
        hline(ax, x - 0.05, x + 0.05, -0.66)
        ax.text(x - side * 0.12, -0.58, f"{shunt['Q']:g}\nMVAr",
                ha="left" if side < 0 else "right", va="center", fontsize=5.5,
                linespacing=0.9)

    ax.set(xlim=(-0.55, 6.55), ylim=(-1.3, 1.45), aspect="equal")
    ax.axis("off")

    out = Path(__file__).resolve().parent
    fig.savefig(out / "kundur_sld.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out / "kundur_sld.png", dpi=300, bbox_inches="tight",
                pad_inches=0.01)
    plt.show()


if __name__ == "__main__":
    main()
