"""
Illustrative frequency response of the Nordic system after a reference
incident, showing how inertia, FFR, FCR-N, FCR-D, aFRR and mFRR take over
from each other.

The model is the aggregated swing equation from the theory chapter,

    2 E_k / f0 * d(Delta f)/dt = sum of reserves + load damping - Delta P_L,

with simple first-order / ramp models of the reserves. The parameters are
chosen to resemble the Nordic system (E_k = 150 GWs, reference incident
1450 MW, FCR-N 600 MW, FCR-D 1450 MW) but are NOT calibrated. The figure is
meant as a qualitative illustration only.

Output: reserve_timeline.pdf (for Overleaf) and reserve_timeline.png
"""

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Parameters
# =====================================================================
f0 = 50.0            # nominal frequency [Hz]
Ek = 150e3           # kinetic energy [MWs] (NAG design level for FCR-D)
dP_L = 1450.0        # reference incident [MW]
D = 400.0            # load frequency dependence [MW/Hz]

# FCR-N: 600 MW, linear in +-0.1 Hz, first-order dynamics
P_fcrn_max = 600.0
T_fcrn = 30.0

# FCR-D upwards: 1450 MW, linear 49.9 -> 49.5 Hz, first-order dynamics
P_fcrd_max = 1450.0
T_fcrd = 3.0

# FFR: triggered at 49.7 Hz, full activation 1.3 s, long support 30 s
P_ffr_max = 250.0
f_ffr = 49.7
T_ffr_act = 1.3
T_ffr_hold = 30.0
T_ffr_deact = 60.0

# aFRR: integral on B*Delta f, capacity and ramp limited (FAT 5 min)
B = 6000.0           # frequency bias [MW/Hz]
T_i = 100.0          # integral time [s]
P_afrr_max = 600.0
rate_afrr = P_afrr_max / 120.0   # [MW/s] (full activation in 2 min)

# mFRR: direct activation, ordered at 60 s, 2.5 min preparation,
# linear ramp to full volume over 10 min (FAT 12.5 min)
t_mfrr_order = 60.0
t_mfrr_start = t_mfrr_order + 150.0
t_mfrr_full = t_mfrr_start + 600.0
P_mfrr_max = dP_L

# Simulation
dt = 0.01
t_end = 1500.0

# =====================================================================
# Simulation (explicit Euler)
# =====================================================================
n = int(t_end / dt) + 1
t = np.arange(n) * dt

df = np.zeros(n)
P_fcrn = np.zeros(n)
P_fcrd = np.zeros(n)
P_ffr = np.zeros(n)
P_afrr = np.zeros(n)
P_mfrr = np.zeros(n)
P_load = np.zeros(n)
P_kin = np.zeros(n)

t_trig = None

for k in range(n - 1):
    x = df[k]

    # --- FCR-N and FCR-D steady-state targets ---------------------
    target_n = P_fcrn_max * np.clip(-x / 0.1, -1.0, 1.0)
    target_d = P_fcrd_max * np.clip((-x - 0.1) / 0.4, 0.0, 1.0)
    P_fcrn[k + 1] = P_fcrn[k] + dt * (target_n - P_fcrn[k]) / T_fcrn
    P_fcrd[k + 1] = P_fcrd[k] + dt * (target_d - P_fcrd[k]) / T_fcrd

    # --- FFR: one activation cycle --------------------------------
    if t_trig is None and f0 + x <= f_ffr:
        t_trig = t[k]
    if t_trig is not None:
        tau = t[k] - t_trig
        if tau < T_ffr_act:
            P_ffr[k + 1] = P_ffr_max * tau / T_ffr_act
        elif tau < T_ffr_act + T_ffr_hold:
            P_ffr[k + 1] = P_ffr_max
        else:
            frac = (tau - T_ffr_act - T_ffr_hold) / T_ffr_deact
            P_ffr[k + 1] = P_ffr_max * max(0.0, 1.0 - frac)

    # --- aFRR: rate-limited integral controller -------------------
    dP_req = -(B / T_i) * x * dt
    dP_req = np.clip(dP_req, -rate_afrr * dt, rate_afrr * dt)
    P_afrr[k + 1] = np.clip(P_afrr[k] + dP_req, -P_afrr_max, P_afrr_max)

    # --- mFRR: linear ramp ----------------------------------------
    if t[k] >= t_mfrr_start:
        frac = (t[k] - t_mfrr_start) / (t_mfrr_full - t_mfrr_start)
        P_mfrr[k + 1] = P_mfrr_max * min(1.0, frac)

    # --- Frequency (aggregated swing equation) --------------------
    P_load[k] = -D * x
    P_res = P_fcrn[k] + P_fcrd[k] + P_ffr[k] + P_afrr[k] + P_mfrr[k]
    dP = P_res + P_load[k] - dP_L
    P_kin[k] = -dP                      # power from rotating masses
    df[k + 1] = x + dt * f0 / (2 * Ek) * dP

P_load[-1] = -D * df[-1]
P_kin[-1] = P_kin[-2]

f = f0 + df
k_nadir = np.argmin(f)
print(f"Nadir: {f[k_nadir]:.3f} Hz at t = {t[k_nadir]:.1f} s")
print(f"Initial RoCoF: {f0 * -dP_L / (2 * Ek):.3f} Hz/s")
print(f"Frequency at t = 120 s: {f[int(120 / dt)]:.3f} Hz")
print(f"Frequency at end: {f[-1]:.4f} Hz")

# =====================================================================
# Plot
# =====================================================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(6.3, 5.2), sharex=True,
    gridspec_kw={"height_ratios": [1, 1.25]},
)

t_min = 0.1
# Log-spaced sample points keep the PDF small (the simulation has 150 000 steps)
mask = np.unique(np.round(np.logspace(np.log10(t_min / dt),
                                      np.log10(n - 1), 4000)).astype(int))

# --- Frequency ------------------------------------------------------
ax1.axhspan(49.9, 50.1, color="0.92", zorder=0)
ax1.text(0.12, 50.06, "Standard frequency range", fontsize=7,
         va="center", color="0.35")
ax1.axhline(49.0, color="0.4", ls="--", lw=0.8)
ax1.text(0.12, 49.04, "Minimum instantaneous frequency (49.0 Hz)",
         fontsize=7, color="0.35", va="bottom")
ax1.plot(t[mask], f[mask], color="k", lw=1.4)

ax1.annotate("Nadir", xy=(t[k_nadir], f[k_nadir]),
             xytext=(t[k_nadir] * 0.35, f[k_nadir] - 0.12),
             fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7),
             va="center")
ax1.annotate("RoCoF set by inertia", xy=(0.6, f[int(0.6 / dt)]),
             xytext=(0.13, 49.55), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.7))
k_qss = int(60 / dt)
ax1.annotate("Stabilised by FCR", xy=(60, f[k_qss]),
             xytext=(60, 49.4), fontsize=8, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.7))
ax1.annotate("Restored by FRR", xy=(900, f[int(900 / dt)]),
             xytext=(500, 49.55), fontsize=8, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.7))

ax1.set_ylabel("Frequency [Hz]")
ax1.set_ylim(48.9, 50.15)

# --- Power contributions (stacked) ------------------------------------
layers = [
    ("Inertia", P_kin, "#bdbdbd"),
    ("Load damping", P_load, "#d9d9d9"),
    ("FFR", P_ffr, "#e41a1c"),
    ("FCR-D", P_fcrd, "#ff7f00"),
    ("FCR-N", P_fcrn, "#ffd92f"),
    ("aFRR", P_afrr, "#377eb8"),
    ("mFRR", P_mfrr, "#4daf4a"),
]
# Positive contributions are stacked upwards, negative contributions
# (power absorbed, e.g. when the frequency is above 50 Hz) downwards.
ys_pos = [np.clip(p[mask], 0, None) for _, p, _ in layers]
ys_neg = [np.clip(p[mask], None, 0) for _, p, _ in layers]
ax2.stackplot(t[mask], *ys_pos, labels=[l for l, _, _ in layers],
              colors=[c for _, _, c in layers], lw=0)
ax2.stackplot(t[mask], *ys_neg, colors=[c for _, _, c in layers], lw=0)
ax2.axhline(0, color="k", lw=0.6)
ax2.axhline(dP_L, color="k", lw=0.8, ls=":")
ax2.text(0.12, dP_L + 30, "Lost generation (1450 MW)", fontsize=7,
         va="bottom")

ax2.set_ylabel("Power [MW]")
ax2.set_ylim(-400, 1650)
ax2.legend(loc="lower center", ncol=7, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)

# --- Time axis --------------------------------------------------------
ax2.set_xscale("log")
ax2.set_xlim(t_min, t_end)
ticks = [0.1, 1, 10, 60, 300, 900]
labels = ["0.1 s", "1 s", "10 s", "1 min", "5 min", "15 min"]
ax2.set_xticks(ticks)
ax2.set_xticklabels(labels)
ax2.minorticks_off()
ax2.set_xlabel("Time after disturbance (logarithmic scale)")

for ax in (ax1, ax2):
    ax.grid(True, which="major", color="0.85", lw=0.5)
    ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("reserve_timeline.pdf")
fig.savefig("reserve_timeline.png", dpi=200)

