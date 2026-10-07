"""Part 1 dynamic verification with a voltage-independent disturbance.

Run from the TOPS folder with k2a_course.py installed.

Event: the governor setpoint of G1 is reduced by DP_MW at T_EVENT. This is a
clean active-power step (like losing DP_MW of generation at G1) that does not
depend on bus voltages, unlike a constant-impedance load step.

Checks against hand calculations
1. Final frequency : sum of turbine changes = -DP - K * dw   (droop, TGOV1)
                     K = sum S_i (1/R_i + D_t,i)
2. AVR (SEXS)      : V_t - V_ref = v_pss - (E_f - E_f0) / K_A (P-controller)
3. Time step       : rerun with DT/2 and compare nadir and final frequency.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import k2a_course as model_data

T_END = 90.0
DT = 0.005
T_EVENT = 1.0
DP_MW = 200.0        # Setpoint reduction on G1 (MW)
GEN_STEP = 0         # Index of G1


def simulate(dt):
    data = model_data.load()
    ps = dps.PowerSystemModel(model=data)
    ps.pf_max_it = 30
    ps.init_dyn_sim()
    gen, gov, avr = ps.gen["GEN"], ps.gov["TGOV1"], ps.avr["SEXS"]
    pss = ps.pss["STAB1"]
    f0 = float(data["f"])
    s_n = gen.par["S_n"] * gen.par["N_par"]

    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic,
                                   0, ps.x0, T_END, max_step=dt)
    out = {k: [] for k in ("t", "w", "pm", "vt", "ef", "vpss")}
    n_steps = round(T_END / dt)
    event_step = round(T_EVENT / dt)
    for step in range(n_steps + 1):
        if step == event_step:
            # Steady state of TGOV1: P_m = (bias - w)/R - D_t w  (gen pu).
            # A bias change of -DP/S_n*R lowers P_m by DP at nominal speed.
            r = gov.par["R"][GEN_STEP]
            gov.int_par["bias"][GEN_STEP] -= DP_MW / s_n[GEN_STEP] * r
        x, v = sol.y, sol.v
        out["t"].append(sol.t)
        out["w"].append(gen.speed(x, v).copy())
        out["pm"].append(gen.P_m(x, v) * s_n)
        out["vt"].append(gen.v_t_abs(x, v).copy())
        out["ef"].append(avr.output(x, v).copy())
        out["vpss"].append(pss.output(x, v).copy())
        if step < n_steps:
            sol.step()
    res = {k: np.asarray(val) for k, val in out.items()}
    res.update(f0=f0, s_n=s_n, gen=gen, gov=gov, avr=avr, ps=ps)
    return res


def analyse(res, label):
    t, w, f0, s_n = res["t"], res["w"], res["f0"], res["s_n"]
    gen, gov, avr = res["gen"], res["gov"], res["avr"]
    H = gen.par["H"]
    coi_w = np.average(w, axis=1, weights=H * s_n)
    coi_f = f0 * (1 + coi_w)
    after = t >= T_EVENT
    # First dip: minimum in the first 5 s (electromechanical + governor).
    # The global minimum comes later, from the slow voltage/load tail.
    early = after & (t <= T_EVENT + 5)
    i_nadir = np.flatnonzero(early)[np.argmin(coi_f[early])]
    df_ss = coi_f[-1] - f0

    # Largest COI RoCoF over a 100 ms window (result, not a check)
    k = round(0.1 / (t[1] - t[0]))
    rocof_max = np.min((coi_f[k:] - coi_f[:-k]) / (t[k:] - t[:-k]))

    # 1. Droop: total turbine change = -DP - K * dw (all in MW, dw in pu).
    K = np.sum(s_n * (1 / gov.par["R"] + gov.par["D_t"]))
    dpm_total = np.sum(res["pm"][-1] - res["pm"][0])
    dw_pred = -(DP_MW + dpm_total) / K
    df_pred = dw_pred * f0
    df_ideal = -DP_MW / K * f0          # if demand were constant

    # Settling: last time |f - f_ss| exceeds 5 % of |df_ss|
    band = 0.05 * abs(df_ss)
    outside = np.flatnonzero(abs(coi_f - coi_f[-1]) > band)
    t_settle = t[outside[-1]] - T_EVENT if len(outside) else 0.0

    # 2. AVR steady-state error. In steady state the lead-lag has gain 1, so
    #    E_f = K (V_ref - V_t + v_pss + bias) and E_f0 = K bias.
    v_ref = gen.par["V"]
    vt_err = res["vt"][-1] - v_ref
    vt_pred = res["vpss"][-1] - (res["ef"][-1] - res["ef"][0]) / avr.par["K"]

    print(f"\n=== {label} ===")
    print(f"Event: G1 governor setpoint -{DP_MW:.0f} MW at t = {T_EVENT:g} s")
    print(f"1. Frequency deviation: simulated {df_ss*1e3:+.3f} mHz, "
          f"droop prediction {df_pred*1e3:+.3f} mHz "
          f"(error {abs(df_ss-df_pred)/abs(df_pred)*100:.2f} %)")
    print(f"   Constant-demand estimate -DP/K*f0 = {df_ideal*1e3:+.3f} mHz")
    print(f"   Turbines G2-G4 picked up {np.sum(res['pm'][-1][1:] - res['pm'][0][1:]):+.1f} MW, "
          f"G1 changed {res['pm'][-1][0]-res['pm'][0][0]:+.1f} MW")
    print(f"   First dip {coi_f[i_nadir]:.4f} Hz at t = {t[i_nadir]:.2f} s, "
          f"final {coi_f[-1]:.4f} Hz, "
          f"settling (5 % band) {t_settle:.1f} s after the event")
    print(f"   Largest COI RoCoF (100 ms window): {rocof_max:.4f} Hz/s")
    print("2. AVR: terminal voltage error V_t - V_ref [pu]")
    for name, e, p in zip(gen.par["name"], vt_err, vt_pred):
        print(f"   {name}: simulated {e:+.2e}, v_pss - dE_f/K {p:+.2e}")
    return dict(coi_f=coi_f, nadir=coi_f[i_nadir], t_nadir=t[i_nadir],
                f_end=coi_f[-1], df_pred=df_pred)


def main():
    print("Kundur RMS model in TOPS, Modified Euler DAE solver.")
    res = simulate(DT)
    a = analyse(res, f"Time step {DT} s")
    res2 = simulate(DT / 2)
    b = analyse(res2, f"Time step {DT/2} s")
    print("\n3. Time step check (DT vs DT/2):")
    print(f"   First-dip difference      {abs(a['nadir']-b['nadir'])*1e3:.3f} mHz")
    print(f"   Final freq difference {abs(a['f_end']-b['f_end'])*1e3:.4f} mHz")

    t, f0 = res["t"], res["f0"]
    names = res["gen"].par["name"]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7.5))
    ax = axes[0]
    for i, n in enumerate(names):
        ax.plot(t, f0 * (1 + res["w"][:, i]), lw=0.8, alpha=0.6, label=n)
    ax.plot(t, a["coi_f"], "k", lw=1.5, label="COI")
    ax.axhline(f0 + a["df_pred"], color="tab:red", ls="--", lw=1,
               label=f"Droop prediction {f0 + a['df_pred']:.3f} Hz")
    ax.plot(a["t_nadir"], a["nadir"], "rx", ms=7, mew=2,
            label=f"First dip {a['nadir']:.3f} Hz")
    ax.set_ylabel("Frequency [Hz]")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_title(f"Frequency after G1 setpoint step of −{DP_MW:.0f} MW")

    ax = axes[1]
    dpm = res["pm"] - res["pm"][0]
    for i, n in enumerate(names):
        ax.plot(t, dpm[:, i], label=n)
    ax.plot(t, dpm.sum(axis=1), "k", lw=1.5, label="Sum")
    ax.set_ylabel("ΔP$_m$ [MW]")
    ax.set_title("Turbine power: G1 reduced, G2–G4 share by droop")

    ax = axes[2]
    for i, n in enumerate(names):
        ax.plot(t, res["vt"][:, i] - res["gen"].par["V"][i], label=n)
    ax.set_ylabel("V$_t$ − V$_{ref}$ [pu]")
    ax.set_title("AVR control objective: terminal voltage error")
    ax.set_xlabel("Time [s]")
    for ax in axes:
        ax.axvline(T_EVENT, color="gray", ls=":")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="best")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "dynamic_verification.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved: {out.name}")


if __name__ == "__main__":
    main()
