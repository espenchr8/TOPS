"""
Schematic overview of the Nordic frequency reserves after a large loss of
generation. Qualitative only: the curves are drawn, not simulated, and the
time axis is not to scale. Time marks follow the Nordic requirements: FCR-D >= 86 % after 7.5 s,
aFRR response within 30 s and full activation within 5 min (Norway),
mFRR full activation within 12.5 min. Layout based on figures by Statnett and
the Nordic Analysis Group (ENTSO-E).

Output: reserve_overview.pdf (for Overleaf) and reserve_overview.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9})

x = np.linspace(0, 10, 2000)          # schematic time, not to scale


def s(x, a, b):
    """Smooth step from 0 at x=a to 1 at x=b."""
    u = np.clip((x - a) / (b - a), 0, 1)
    return u * u * (3 - 2 * u)


# Power of each reserve (normalised, 1 = lost generation)
# Positions of the time marks on the schematic axis
T75, T30, T5, T125 = 1.9, 3.4, 5.8, 8.1   # 7.5 s, 30 s, 5 min, 12.5 min
p_ffr = 0.5 * s(x, 0.1, 0.3) * (1 - s(x, 0.6, 1.2))
p_fcr = s(x, 0.25, 2.3) * (1 - s(x, 3.6, T5))
p_afrr = s(x, T30, T5) * (1 - s(x, 6.6, T125))
p_mfrr = s(x, 6.6, T125)

# Frequency deviation (drawn from control points)
pts = np.array([[0, 0], [0.1, 0.36], [0.45, 0.62], [0.8, 0.5],
                [1.2, 0.47], [T75, 0.48], [2.7, 0.46], [3.6, 0.4],
                [4.6, 0.18], [5.4, 0.03], [T5, 0.0], [10, 0.0]])
dev = PchipInterpolator(pts[:, 0], pts[:, 1])(x)
f = 50 - 1.0 * dev

C = {"FFR": "#d62728", "FCR": "#ff7f0e", "aFRR": "#1f77b4", "mFRR": "#2ca02c"}
reserves = [("FFR", p_ffr), ("FCR", p_fcr), ("aFRR", p_afrr), ("mFRR", p_mfrr)]
marks = [(T75, "7.5 s"), (T30, "30 s"), (T5, "5 min"), (T125, "12.5 min")]

fig, (ax2, ax1, ax3) = plt.subplots(
    3, 1, figsize=(5.8, 4.1), sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 0.62], "hspace": 0.12})

# --- Power ---------------------------------------------------------------
for name, p in reserves:
    ax1.plot(x, p, color=C[name], lw=1.8)
ax1.axhline(1, color="0.4", ls=":", lw=0.8)
for name, xl in [("FFR", 0.4), ("FCR", 2.65), ("aFRR", 6.4), ("mFRR", 9.1)]:
    label = "FCR-D" if name == "FCR" else name
    ax1.text(xl, 1.12, label, color=C[name], ha="center", va="bottom",
             fontweight="bold")
ax1.set_ylim(0, 1.32)
ax1.set_yticks([1])
ax1.set_yticklabels([r"$\Delta P_L$"])
ax1.set_ylabel("Activated power")

# --- Frequency -------------------------------------------------------------
ax2.axhspan(49.9, 50.1, color="0.92", zorder=0)
ax2.axhline(50.0, color="k", lw=0.8)
ax2.axhline(49.9, color="0.4", ls="--", lw=0.8)
ax2.plot(x, f, color="k", lw=1.6)
ax2.text(9.95, 49.87, "Standard frequency range (49.9\u201350.1 Hz)", ha="right",
         va="top", fontsize=7.5, color="0.35")
ax2.axhline(49.0, color="0.4", ls=":", lw=0.9)
ax2.text(9.95, 49.03, "Minimum instantaneous frequency", ha="right",
         va="bottom", fontsize=7.5, color="0.35")
ax2.axhline(49.5, color="0.6", ls=":", lw=0.7)
ax2.set_yticks([49.0, 49.5, 50.0])
ax2.set_yticklabels(["49.0 Hz", "49.5 Hz", "50.0 Hz"])
ax2.set_ylim(48.9, 50.15)
ax2.set_ylabel("Frequency")

# --- Time marks -----------------------------------------------------------
for ax in (ax1, ax2):
    for xm, _ in marks:
        ax.axvline(xm, color="0.5", ls="--", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
ax1.set_xticks([xm for xm, _ in marks])
ax1.set_xticklabels([lab for _, lab in marks])
ax1.tick_params(axis="x", labelbottom=True)
ax1.set_xlim(0, 10)
ax2.tick_params(axis="x", labelbottom=False)

# --- Roles of the reserves --------------------------------------------------
ax3.axis("off")
ax3.set_ylim(0, 1)
roles = [("FFR", 0.0, T75, "Slows the\nfrequency drop\n(low inertia only)"),
         ("FCR", T75, T30 + 0.7, "Stops the drop\nand stabilises\nthe frequency"),
         ("aFRR", T30 + 0.7, T125, "Brings the frequency\nback to 50 Hz"),
         ("mFRR", T125, 10.0, "Releases aFRR and\nmaintains the balance")]
for name, a, b, txt in roles:
    xc = 0.5 * (a + b)
    box = "FCR-D" if name == "FCR" else name
    ax3.text(xc, 0.62, box, ha="center", va="center", color="white",
             fontweight="bold", fontsize=8.5,
             bbox=dict(boxstyle="square,pad=0.3", fc=C[name], ec="none"))
    ax3.text(xc, 0.4, txt, ha="center", va="top", fontsize=7.5)
ax3.set_xlabel("")
fig.text(0.53, 0.02, "Time after disturbance (not to scale)", ha="center",
         fontsize=9)

fig.align_ylabels([ax2, ax1])
fig.savefig("reserve_overview.pdf", bbox_inches="tight")
fig.savefig("reserve_overview.png", dpi=200, bbox_inches="tight")
