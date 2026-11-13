"""Part 2: the Part 1 load step with and without secondary control (AGC).

Built on dynamic_kundur_course.py: same model, power flow, solver, time step
and 100 MW load step at B9. The only addition is AGC.

AGC (Kundur Sec. 11.1.5): one integrator per area moves the governor
references of the two units in that area until the area control error is zero.

    ACE_a = dP_tie,a + B_a * w_a          (MW)
    dz_a/dt = -K_I * ACE_a                (z_a: extra set point for area a, MW)

dP_tie,a  change in export from area a on the B7-B8 lines
w_a       speed deviation of area a (pu, inertia-weighted)
B_a       frequency bias = sum S_n (1/R + D_t) of the area, same terms as K
          in the Part 1 droop equation (half of K = 72 072 MW/pu)

Run from the TOPS folder, next to dynamic_kundur_course.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import k2a_course as model_data

from dynamic_kundur_course import DT, LOAD_STEP_MW, T_EVENT, check_balance

T_END = 120.0   # Longer than Part 1, since AGC is slower than droop.
K_I = 0.05      # AGC integral gain [1/s]
AREAS = {"Area 1": ["G1", "G2"], "Area 2": ["G3", "G4"]}
AREA_LOAD_BUS = {"Area 1": "B7", "Area 2": "B9"}


def tie_flow(ps, x, v):
    """Corridor transfer B7->B8 in MW, as in dynamic_kundur_course.py."""
    names = list(ps.lines["Line"].par["name"])
    sf = ps.lines["Line"].s_from(x, v) * ps.s_n
    return sum(sf[names.index(n)].real for n in ("L7-8-1", "L7-8-2"))


def simulate(k_i):
    """Same as simulate(disturbed=True) in Part 1, with AGC when k_i > 0."""
    data = model_data.load()
    ps = dps.PowerSystemModel(model=data)
    ps.perform_kron_reduction = False
    ps.pf_max_it = 30
    ps.init_dyn_sim()
    if not ps.power_flow_ready:
        raise RuntimeError("Initial power flow did not converge")

    gen = ps.gen["GEN"]
    gov = ps.gov["TGOV1"]
    load = ps.loads["Load"]
    load_idx = np.flatnonzero(load.par["bus"] == "B9")[0]
    b9 = load.bus_idx_red["terminal"][load_idx]
    weights = gen.par["H"] * gen.par["S_n"] * gen.par["N_par"]
    p_base = gen.par["S_n"] * gen.par["PF_n"] * gen.par["N_par"]
    names = list(gen.par["name"])

    # --- AGC data -----------------------------------------------------------
    area_idx = [[names.index(g) for g in units] for units in AREAS.values()]
    gov_idx = [list(gov.par["gen"]).index(g) for g in names]  # gov rows in gen order
    R = gov.par["R"][gov_idx]
    S_n = gen.par["S_n"]
    unit_k = S_n * (1 / R + gov.par["D_t"][gov_idx])           # MW per pu speed
    B = np.array([unit_k[i].sum() for i in area_idx])           # bias per area
    bias0 = gov.int_par["bias"].copy()   # TGOV1 reference from initialization
    z = np.zeros(len(AREAS))             # AGC output per area [MW]

    def area_speed(speed):
        return np.array([np.average(speed[i], weights=gen.par["H"][i])
                         for i in area_idx])

    def area_power(x, v, extra_y):
        """Generation, load and export per area [MW], for the end check."""
        pe = gen.p_e(x, v) * S_n
        p_load = load.P(x, v)
        bus = list(load.par["bus"])
        added = abs(v[b9])**2 * extra_y.real * ps.s_n
        tie = tie_flow(ps, x, v)
        return np.array([
            [pe[area_idx[0]].sum(), p_load[bus.index("B7")], tie],
            [pe[area_idx[1]].sum(), p_load[bus.index("B9")] + added, -tie]])

    v0 = ps.solve_algebraic(0, ps.x0)
    tie0 = tie_flow(ps, ps.x0, v0)
    pm_gen0 = gen.P_m(ps.x0, v0) * p_base

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives, ps.solve_algebraic, 0, ps.x0, T_END, max_step=DT)
    extra_y = 0j
    initial_check = check_balance(ps, sol.t, sol.y, extra_y, b9)
    initial_area = area_power(sol.y, sol.v, extra_y)
    times, speeds, ties, mech = [], [], [], []
    event_step = round(T_EVENT / DT)
    for step in range(round(T_END / DT) + 1):
        if step == event_step:
            # Same load step as dynamic_kundur_course.py.
            pre_event_p = abs(sol.v[b9])**2 * load.y_load[load_idx].real * ps.s_n
            extra_y = (LOAD_STEP_MW / pre_event_p) * load.y_load[load_idx]
            ps.y_bus_red_mod[b9, b9] += extra_y
            sol.v[:] = ps.solve_algebraic(sol.t, sol.y)

        speed = gen.speed(sol.y, sol.v).copy()
        d_tie = tie_flow(ps, sol.y, sol.v) - tie0
        times.append(sol.t)
        speeds.append(speed)
        ties.append(d_tie)
        mech.append(gen.P_m(sol.y, sol.v) * p_base - pm_gen0)
        if step < round(T_END / DT):
            sol.step()
            # --- AGC: integrate ACE and move the governor references --------
            ace = np.array([d_tie, -d_tie]) + B * area_speed(speed)
            z += -k_i * ace * DT
            dp = np.zeros(len(names))          # set point change per unit [MW]
            for j, i in enumerate(area_idx):
                dp[i] = z[j] / len(i)          # shared equally in the area
            # TGOV1 droop input is (-speed + bias)/R, so +dp MW needs +R*dp/S_n.
            bias = bias0.copy()
            bias[gov_idx] += R * dp / S_n
            gov.int_par["bias"][:] = bias

    t = np.asarray(times)
    freq = 50 * (1 + np.asarray(speeds))
    coi = np.average(freq, axis=1, weights=weights)
    ties = np.asarray(ties)
    mech = np.asarray(mech)
    dpm_area = np.column_stack([mech[:, i].sum(axis=1) for i in area_idx])

    label = f"AGC, K_I = {k_i:g} 1/s" if k_i else "No AGC (Part 1 load step)"
    print(f"\n=== {label} ===")
    if k_i:
        print("Frequency bias B: " + ", ".join(
            f"{a} {b:.0f} MW/pu" for a, b in zip(AREAS, B)))
    i60 = np.argmin(abs(t - 60))
    print(f"COI frequency at 60 s: {coi[i60]:.6f} Hz. At {T_END:.0f} s: {coi[-1]:.6f} Hz")
    print(f"Lowest COI frequency after the step: {coi[t >= T_EVENT].min():.4f} Hz")
    outside = np.flatnonzero(abs(coi - 50) > 0.005)
    if k_i and outside.size:
        print(f"Within 5 mHz of 50 Hz from {t[outside[-1]] - T_EVENT:.0f} s after the step")
    print(f"Corridor transfer B7->B8: {tie0:.1f} MW -> {tie0 + ties[-1]:.1f} MW")
    for name, d in zip(names, mech[-1]):
        print(f"  {name}: mechanical power change {d:+.2f} MW")
    if k_i:
        print("AGC set point change: " + ", ".join(
            f"{a} {v:+.2f} MW" for a, v in zip(AREAS, z)))
        # Check: each area should cover its own change in load.
        change = area_power(sol.y, sol.v, extra_y) - initial_area
        print("Change per area [MW]   Generation   Load   Export   Losses")
        for a, (g, l, e) in zip(AREAS, change):
            print(f"  {a:18s} {g:+10.2f} {l:+7.2f} {e:+8.2f} {g - l - e:+8.2f}")
    final_check = check_balance(ps, sol.t, sol.y, extra_y, b9)
    print(f"Network balance error before / at end: "
          f"{initial_check[3]:+.1e} / {final_check[3]:+.1e} MW")
    return t, freq, coi, ties, dpm_area


def plot(no_agc, with_agc):
    """Same layout and style as the Part 1 figure."""
    t0, _, coi0, tie0, dpm0 = no_agc
    t, freq, coi, tie, dpm = with_agc
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 4.2))
    for i, name in enumerate(("G1", "G2", "G3", "G4")):
        axes[0].plot(t, freq[:, i], lw=0.8, alpha=0.6, label=name)
    axes[0].plot(t, coi, color="black", lw=1.5, label="COI with AGC")
    axes[0].plot(t0, coi0, "--", color="gray", label="COI without AGC")
    axes[0].plot(t[-1], coi[-1], "rx", ms=7, mew=2,
                 label=f"Final value: {coi[-1]:.4f} Hz")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].ticklabel_format(axis="y", useOffset=False)
    axes[0].set_title(f"(a) Generator frequencies and system average, "
                      f"AGC with $K_I$ = {K_I:g} s$^{{-1}}$")
    for j, area in enumerate(AREAS):
        axes[1].plot(t, dpm[:, j], color=f"C{j}", label=f"{area} with AGC")
    axes[1].plot(t0, dpm0[:, 0], "--", color="gray", label="Each area without AGC")
    axes[1].set_ylabel("$\\Delta P_m$ [MW]")
    axes[1].set_title("(b) Turbine response per area")
    axes[2].plot(t, tie, label="With AGC")
    axes[2].plot(t0, tie0, "--", color="gray", label="Without AGC")
    axes[2].set_ylabel("$\\Delta P_{tie}$ [MW]")
    axes[2].set_title("(c) Change in corridor transfer B7-B8")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.axvline(T_EVENT, color="gray", ls=":", label="B9 load added")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()

    # Comment out fig.savefig(...) to disable saving.
    output_file = Path(__file__).resolve().parent / "agc_response.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    # Vector version for LaTeX: sharp text at any zoom.
    fig.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=0.02)
    plt.show()


def main():
    print("Kundur data from k2a_course.py. Same setup and load step as Part 1.")
    print(f"Time solver: Modified Euler DAE. Time step: {DT:g} s")
    print("Without AGC the result at 60 s should equal Part 1 "
          "(49.970689 Hz, 438.9 MW).")
    no_agc = simulate(k_i=0.0)
    with_agc = simulate(k_i=K_I)
    plot(no_agc, with_agc)


if __name__ == "__main__":
    main()
