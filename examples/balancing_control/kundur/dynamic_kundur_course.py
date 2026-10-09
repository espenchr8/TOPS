"""Part 1 dynamic RMS study: an undisturbed run and a small load increase."""

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import k2a_course as model_data

from pathlib import Path



T_END = 60.0
DT = 0.005
T_EVENT = 5.0
LOAD_STEP_MW = 100.0  # Added active demand at the pre-event B9 voltage.
# 100 MW matches k2a_loadstep in the master project (about 3.7 % of load).
# The added load keeps B9's initial Q/P ratio and is a fixed admittance.
# Its actual MW therefore changes when the voltage changes.


def power_summary(ps, x, v, extra_y, b9):
    """Actual active demand and network losses at the current voltage, in MW."""
    original = sum(np.sum(m.P(x, v)) for m in ps.loads.values())
    # The added load is in the network matrix, not in ps.loads.
    added = abs(v[b9])**2 * extra_y.real * ps.s_n
    losses = sum(np.sum((m.s_from(x, v) + m.s_to(x, v)).real) * ps.s_n
                 for group in (ps.lines, ps.trafos) for m in group.values())
    return np.array([original, added, losses])


def check_balance(ps, t, x, extra_y, b9):
    """Check the network balance and TOPS GEN speed equation separately."""
    v = ps.solve_algebraic(t, x)
    dx = ps.state_derivatives(t, x, v)  # Refresh model outputs at this state.
    gen = ps.gen["GEN"]
    p = gen.par
    base = p["S_n"] * p["N_par"]
    speed = gen.speed(x, v)
    pe = gen.p_e(x, v)  # pu on each generator's MVA base.
    pm = gen.P_m(x, v) * p["PF_n"]  # Same base as pe.
    demand = power_summary(ps, x, v, extra_y, b9)
    electrical = np.sum(pe * base)
    mechanical = np.sum(pm * base)

    # Exact form used by TOPS GEN, expressed on the generator MVA base:
    # 2 H d(speed)/dt = PF_n P_m/(1+speed) - p_e - PF_n D speed.
    acceleration = 2 * p["H"] * gen.local_view(dx)["speed"]
    damping = p["PF_n"] * p["D"] * speed
    driving = pm / (1 + speed) - pe - damping
    swing_error = np.max(abs((driving - acceleration) * base))
    return demand, mechanical, electrical, electrical - np.sum(demand), swing_error


def simulate(disturbed=False):
    """Start from the same power flow as the static base case."""
    data = model_data.load()
    ps = dps.PowerSystemModel(model=data)
    ps.perform_kron_reduction = False  # Keep all 11 bus voltages available.
    ps.pf_max_it = 30
    ps.init_dyn_sim()  # Solve power flow, then initialize machines and controls.
    if not ps.power_flow_ready:
        raise RuntimeError("Initial power flow did not converge")

    gen = ps.gen["GEN"]
    load = ps.loads["Load"]
    b8 = np.flatnonzero(ps.buses["name"] == "B8")[0]
    load_idx = np.flatnonzero(load.par["bus"] == "B9")[0]
    b9 = load.bus_idx_red["terminal"][load_idx]
    f0 = float(data["f"])
    weights = gen.par["H"] * gen.par["S_n"] * gen.par["N_par"]
    # P_m uses the generator active-power base. Convert each unit to MW.
    p_base = gen.par["S_n"] * gen.par["PF_n"] * gen.par["N_par"]

    v0 = ps.solve_algebraic(0, ps.x0)
    dx0 = ps.state_derivatives(0, ps.x0, v0)
    label = f"B9 load step: {LOAD_STEP_MW:.2f} MW at pre-event voltage" if disturbed else "No disturbance"
    print(f"\n=== {label} ===")
    print(f"Power flow ready. Dynamic states: {ps.n_states}")
    print(f"Largest initial state derivative: {max(abs(dx0)):.3e}")
    print("Initial derivatives should be close to zero.")
    print(f"RMS voltage versus power-flow voltage: "
          f"{max(abs(v0 - ps.v_0)):.3e} pu difference")

    # Check: per-generator turbine response and corridor transfer B7->B8.
    line_names = list(ps.lines["Line"].par["name"])

    def tie_flow(x, v):
        sf = ps.lines["Line"].s_from(x, v) * ps.s_n
        return sum(sf[line_names.index(n)].real for n in ("L7-8-1", "L7-8-2"))

    pm_gen0 = gen.P_m(ps.x0, v0) * p_base
    tie0 = tie_flow(ps.x0, v0)

    # Same solver as the master-project baseline.
    # It advances machine/control states and solves network voltages each step.
    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives, ps.solve_algebraic, 0, ps.x0, T_END, max_step=DT)
    times, speeds, mechanical, voltages, terminal = [], [], [], [], []
    v_ref = np.asarray(gen.par["V"], dtype=float)  # AVR voltage setpoints
    extra_y = 0j
    initial_check = check_balance(ps, sol.t, sol.y, extra_y, b9)
    initial_power = initial_check[0]
    event_step = round(T_EVENT / DT)
    for step in range(round(T_END / DT) + 1):
        if disturbed and step == event_step:
            # TOPS turns this load into a fixed admittance after initialization.
            # Add a small parallel load through the network modification matrix.
            # Scale the existing admittance to the requested MW at this voltage.
            pre_event_p = abs(sol.v[b9])**2 * load.y_load[load_idx].real * ps.s_n
            extra_y = (LOAD_STEP_MW / pre_event_p) * load.y_load[load_idx]
            ps.y_bus_red_mod[b9, b9] += extra_y
            sol.v[:] = ps.solve_algebraic(sol.t, sol.y)
            extra_p = abs(sol.v[b9])**2 * extra_y.real * ps.s_n
            print(f"At {sol.t:.2f} s: add a B9 load rated {LOAD_STEP_MW:.2f} MW at the pre-event voltage.")
            print(f"Added demand just after the change: {extra_p:.2f} MW")
            print("Demand then varies with the square of the B9 voltage.")

        times.append(sol.t)
        speeds.append(gen.speed(sol.y, sol.v).copy())
        mechanical.append(np.sum(gen.P_m(sol.y, sol.v) * p_base))
        voltages.append(abs(sol.v[[b8, b9]]))
        # AVR control objective: generator terminal voltage minus setpoint.
        terminal.append(gen.v_t_abs(sol.y, sol.v) - v_ref)
        if step < round(T_END / DT):
            sol.step()

    t = np.asarray(times)
    frequency = f0 * (1 + np.asarray(speeds))
    coi = np.average(frequency, axis=1, weights=weights)
    pm = np.asarray(mechanical)
    voltage = np.asarray(voltages)
    vt_error = np.asarray(terminal)
    v8, v9 = voltage[:, 0], voltage[:, 1]
    tail = t >= T_END - 5
    print(f"Largest generator frequency deviation: "
          f"{np.max(abs(frequency-f0)):.6e} Hz")
    # Nadir is the lowest COI frequency in the simulated post-event interval.
    nadir_idx = np.flatnonzero(t >= T_EVENT)[np.argmin(coi[t >= T_EVENT])]
    print(f"Lowest COI frequency in this run: {coi[nadir_idx]:.6f} Hz "
          f"at {t[nadir_idx]:.2f} s")
    if disturbed and nadir_idx == len(t) - 1:
        print("The minimum is at the end of the run, not an earlier local dip.")
    print(f"Final COI frequency: {coi[-1]:.6f} Hz")
    print(f"COI frequency range in the last 5 s: {np.ptp(coi[tail]):.6e} Hz")
    print(f"Change in total mechanical power: {pm[-1]-pm[0]:+.3f} MW")
    dpm = gen.P_m(sol.y, sol.v) * p_base - pm_gen0
    for name, d in zip(gen.par["name"], dpm):
        print(f"  {name}: mechanical power change {d:+.2f} MW")
    print(f"Corridor transfer B7->B8: {tie0:.1f} MW -> "
          f"{tie_flow(sol.y, sol.v):.1f} MW")
    print(f"B8 voltage: initial {v8[0]:.4f} pu, final {v8[-1]:.4f} pu")
    print(f"B9 voltage: initial {v9[0]:.4f} pu, final {v9[-1]:.4f} pu")
    print("Generator terminal voltage minus AVR setpoint [pu]:")
    for name, peak, end in zip(gen.par["name"], np.max(abs(vt_error), axis=0),
                               vt_error[-1]):
        print(f"  {name}: largest {peak:.4f}, final {end:+.5f}")
    if not disturbed:
        print(f"Largest B8 voltage drift: {max(abs(v8-v8[0])):.3e} pu")
        print("Frequency, voltage and mechanical power should stay almost constant.")
    else:
        print("Check the curves for a frequency dip and increased mechanical power.")
        print("Droop control can leave a frequency offset after the response settles.")
    if disturbed:
        final_check = check_balance(ps, sol.t, sol.y, extra_y, b9)
        final_power, final_pm, final_pe, network_error, swing_error = final_check
        print("\n=== ACTIVE POWER CHECK [MW] ===")
        print("Quantity                  Before       At end       Change")
        before = np.r_[initial_power, initial_check[1], initial_check[2]]
        after = np.r_[final_power, final_pm, final_pe]
        labels = ("Original loads", "Added B9 load", "Network losses",
                  "Mechanical power", "Electrical generation")
        for label, a, b in zip(labels, before, after):
            print(f"{label:24s} {a:10.3f} {b:12.3f} {b-a:+12.3f}")
        print(f"Net change in demand plus losses: {np.sum(final_power-initial_power):+.3f} MW")
        print("\nNetwork check: electrical generation = all loads + network losses")
        print(f"Balance error before event: {initial_check[3]:+.3e} MW")
        print(f"Balance error at end:       {network_error:+.3e} MW")
        print("These errors should be close to zero, including during a transient.")
        print("\nSwing equation check: turbine input, electrical output and rotor response")
        print(f"Largest generator equation residual at end: {swing_error:.3e} MW equivalent")
        print("This residual should be close to zero. It checks the model equation.")
        print(f"Mechanical minus electrical power at end: {final_pm-final_pe:+.6f} MW")
        print("That difference is not the network balance error.")
        print("TOPS includes rotor speed in its mechanical-input term.")
        print("A small equation residual alone does not verify time-step accuracy.")
    return t, frequency, coi, pm, voltage, gen.par["name"], vt_error


def main():
    print("Kundur data from k2a_course.py. Dynamic RMS simulation in TOPS.")
    print(f"Time solver: Modified Euler DAE. Time step: {DT:g} s")
    print("Both runs start from the original static base case.")
    baseline = simulate()
    response = simulate(disturbed=True)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 5))
    t, freq, coi, pm, voltage, names, vt_error = response
    for i, name in enumerate(names):
        axes[0].plot(t, freq[:, i], lw=0.8, alpha=0.6, label=name)
    axes[0].plot(t, coi, color="black", lw=1.5, label="COI (inertia-weighted average)")
    axes[0].plot(baseline[0], baseline[2], "--", color="gray", label="No disturbance")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].ticklabel_format(axis="y", useOffset=False)
    axes[0].plot(t[-1], coi[-1], "rx", ms=7, mew=2,
                 label=f"Final value: {coi[-1]:.4f} Hz")
    axes[0].set_title("Generator frequencies and system average")
    axes[1].plot(t, pm-pm[0], label="With B9 load step")
    axes[1].plot(baseline[0], baseline[3]-baseline[3][0], "--", label="No disturbance")
    axes[1].set_ylabel("Change in total\nmechanical power [MW]")
    axes[1].set_title("Combined turbine response of G1-G4")
    # AVR control objective. B8/B9 voltages are printed in the terminal.
    for i, name in enumerate(names):
        axes[2].plot(t, vt_error[:, i], color=f"C{i}", label=name)
    axes[2].axhline(0, color="gray", ls="--", lw=0.8, label="AVR setpoint")
    axes[2].set_ylabel("$V_t - V_{ref}$ [pu]")
    axes[2].set_title("AVR control objective: generator terminal voltage")
    axes[2].set_xlabel("Time [s]")
    for ax in axes:
        ax.axvline(T_EVENT, color="gray", ls=":", label="B9 load added")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=3)
    #fig.suptitle(f"Kundur RMS response: B9 load step at {T_EVENT:g} s\n"
                 #f"{LOAD_STEP_MW:g} MW at pre-event voltage, voltage-dependent load")
    fig.tight_layout()

    # Comment out fig.savefig(...) to disable PNG saving.
    output_file = Path(__file__).resolve().parent / "dynamic_response.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
