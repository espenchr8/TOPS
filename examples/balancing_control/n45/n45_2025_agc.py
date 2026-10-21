"""N45 disturbance with conventional FCR and a central frequency-based AGC (aFRR).

Who regulates in this simulation (the whole N45 network is simulated):

    Unit                     FCR (droop)   In bias B   AGC (aFRR)
    HYGOV hydro (36)         yes           yes         yes (default: all)
    TGOV1 thermal (11)       yes           yes         only if listed in THERMAL_AGC_UNITS
    HVDC VSC (4)             no            no          only if HVDC_SHARE > 0
    Wind VSC (29)            no            no          no (constant power)
    Loads                    no            no          -

Disturbances: a load step or a reduced HVDC import on one link.
The AGC model and its connections are in ../common/agc.py.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import n45_2025 as model_data

# AGC: make examples/balancing_control/common importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from agc import (
    add_agc_to_model,
    connect_agc,
    regulating_strength_mw_per_hz,
    user_model_library,
)


# =============================================================================
# STUDY SETTINGS
# =============================================================================

EVENT_TIME = 1.0
T_END = 300.0
MAX_STEP = 5e-3
FINAL_AVERAGING_WINDOW_S = 5.0
RESTORATION_BAND_HZ = 0.01

# Disturbance: "load_step" or "hvdc_reduction"
DISTURBANCE = "load_step"

LOAD_NAME = "L5240-1"           # load_step
LOAD_STEP_MW = 1000.0


HVDC_EVENT_LINK = "NO_2-DE"     # hvdc_reduction
HVDC_EVENT_NEW_MW = 0.0


# =============================================================================
# AGC SETTINGS
# =============================================================================

AGC_ENABLED = True
T_F = 2.0                       # Low-pass filter time constant (s)
T_I = 100.0                     # Integral time (s); K_i = B / T_i
K_P = 0.0                       # Proportional gain (MW/Hz); 0 = pure integral
HYDRO_AGC_UNITS = None          # None = all HYGOV units
THERMAL_AGC_UNITS = []          # e.g. ["G3000-1"]; empty = no thermal units
HVDC_SHARE = 0.0                # Share of the AGC command sent via HVDC
HVDC_LINK_SPLIT = {"NO_2-GB": 1.0}


def run_simulation():

    # =========================================================================
    # MODEL LOADING AND INITIALIZATION
    # =========================================================================

    model = model_data.load()
    system_base_mva = float(model["base_mva"])
    nominal_frequency = float(model["f"])

    # Frequency bias B = regulating strength (MW/Hz), from the governors.
    # The AGC must be added before PowerSystemModel so that its states become
    # part of the system state vector.
    strength = regulating_strength_mw_per_hz(model)
    bias_B = sum(strength.values())
    K_I = bias_B / T_I if AGC_ENABLED else 0.0
    add_agc_to_model(model, T_f=T_F, K_p=K_P if AGC_ENABLED else 0.0, K_i=K_I)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_model_library())
    ps.init_dyn_sim()

    # Connect frequency measurement and unit set-points.
    agc = ps.agc["AGC"]
    participation_gen, participation_hvdc = connect_agc(
        agc, ps,
        hydro_units=HYDRO_AGC_UNITS,
        thermal_units=THERMAL_AGC_UNITS,
        hvdc_share=HVDC_SHARE,
        hvdc_link_split=HVDC_LINK_SPLIT,
    )

    gen = ps.gen["GEN"]
    load = ps.loads["Load"]
    vsc = ps.vsc["VSC_SI"]

    print("\nAGC settings")
    print("------------")
    print("AGC enabled:", AGC_ENABLED)
    for gov_type, value in strength.items():
        print(f"  Regulating strength {gov_type}: {value:.0f} MW/Hz")
    print(f"  Frequency bias B: {bias_B:.0f} MW/Hz")
    print(f"  T_f = {T_F:.1f} s, T_i = {T_I:.0f} s, K_p = {K_P:.0f} MW/Hz, K_i = {K_I:.1f} MW/(Hz s)")
    print(f"  Generators in AGC: {len(participation_gen)}, HVDC: {participation_hvdc}")

    # =========================================================================
    # INITIAL EQUILIBRIUM CHECK
    # =========================================================================

    v_initial = ps.solve_algebraic(0.0, ps.x0)
    dx_initial = ps.state_derivatives(0.0, ps.x0, v_initial)
    max_initial_derivative = float(np.max(np.abs(dx_initial)))
    if not ps.power_flow_ready or max_initial_derivative > 1e-5:
        raise RuntimeError(
            "Invalid initial operating point: "
            f"power_flow_ready={ps.power_flow_ready}, "
            f"max |dx/dt|={max_initial_derivative:.3e}."
        )

    # =========================================================================
    # GENERATOR DATA
    # =========================================================================

    # Used for COI frequency and mechanical-power conversion to MW.
    rating_mva = (
        np.asarray(gen.par["S_n"], dtype=float)
        * np.asarray(gen.par["N_par"], dtype=float)
    )
    inertia_weights = np.asarray(gen.par["H"], dtype=float) * rating_mva
    mechanical_base_mw = rating_mva * np.asarray(gen.par["PF_n"], dtype=float)

    vsc_names = np.asarray(vsc.par["name"], dtype=str)
    vsc_s_n = np.asarray(vsc.par["S_n"], dtype=float)
    hvdc_idx = np.where(~np.char.startswith(vsc_names, "WG"))[0]

    # =========================================================================
    # DISTURBANCE
    # =========================================================================

    if DISTURBANCE == "load_step":
        matches = np.where(np.asarray(load.par["name"], dtype=str) == LOAD_NAME)[0]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one load named {LOAD_NAME}.")
        load_idx = int(matches[0])
        load_bus_idx = int(load.bus_idx_red["terminal"][load_idx])
        initial_load_mw = float(load.p(ps.x0, v_initial)[load_idx]) * system_base_mva

        # The load model is constant impedance. The requested P step at the
        # pre-event voltage gives the admittance step delta_Y = delta_P / |V_0|^2.
        delta_y = (LOAD_STEP_MW / system_base_mva) / abs(v_initial[load_bus_idx]) ** 2

        nominal_disturbance_mw = LOAD_STEP_MW
        event_text = f"+{LOAD_STEP_MW:.0f} MW load step at {LOAD_NAME}"

    elif DISTURBANCE == "hvdc_reduction":
        event_link_idx = int(np.where(vsc_names == HVDC_EVENT_LINK)[0][0])
        initial_link_mw = float(vsc._input_values["p_ref"][event_link_idx]) * vsc_s_n[event_link_idx]
        nominal_disturbance_mw = initial_link_mw - HVDC_EVENT_NEW_MW
        event_text = (
            f"HVDC import {HVDC_EVENT_LINK} reduced "
            f"{initial_link_mw:.0f} -> {HVDC_EVENT_NEW_MW:.0f} MW"
        )
    else:
        raise ValueError(f"Unknown disturbance type: {DISTURBANCE}")

    print("Maximum initial derivative:", f"{max_initial_derivative:.6e}")
    print(f"Applying {event_text}, t={EVENT_TIME:.3f} s.")

    # =========================================================================
    # NUMERICAL SOLVER AND RESULT STORAGE
    # =========================================================================

    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0,
        T_END,
        max_step=MAX_STEP,
    )

    def store(x, v):
        time_values.append(float(solver.t))
        speed_values.append(np.asarray(gen.speed(x, v)).copy())
        governor_values.append(np.asarray(gen.P_m(x, v), dtype=float).copy())
        hvdc_values.append(np.asarray(vsc.p_e(x, v), dtype=float)[hvdc_idx] * vsc_s_n[hvdc_idx])
        agc_values.append(float(agc.output(x, v)[0]))

    time_values, speed_values, governor_values, hvdc_values, agc_values = [], [], [], [], []
    store(ps.x0, v_initial)

    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    event_applied = False
    next_progress = 10

    while solver.t < T_END:
        # Apply the disturbance once, at the selected event time.
        if solver.t >= EVENT_TIME and not event_applied:
            if DISTURBANCE == "load_step":
                load.y_load[load_idx] += delta_y
                ps.y_bus_red_mod[load_bus_idx, load_bus_idx] += delta_y
            else:
                # Reduced HVDC import (scheduled set-point change)
                vsc.set_input(
                    "p_ref", HVDC_EVENT_NEW_MW / vsc_s_n[event_link_idx], idx=event_link_idx
                )
            solver.v[:] = ps.solve_algebraic(solver.t, solver.y)
            event_applied = True

            if DISTURBANCE == "load_step":
                actual_step = (
                    float(load.p(solver.y, solver.v)[load_idx]) * system_base_mva
                    - initial_load_mw
                )
                print("Actual immediate load increase:", f"{actual_step:.3f} MW")

        solver.step()
        store(solver.y, solver.v)

        # Terminal progress indicator; this does not affect the simulation.
        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    # =========================================================================
    # RESULT PROCESSING
    # =========================================================================

    t = np.asarray(time_values)
    speed = np.asarray(speed_values)
    governor_p_m = np.asarray(governor_values)
    hvdc_power_mw = np.asarray(hvdc_values)

    coi_frequency = nominal_frequency * (
        1.0 + np.average(speed, axis=1, weights=inertia_weights)
    )

    # Incremental mechanical power of all governors (FCR + aFRR), MW.
    total_governor_response = np.sum(
        (governor_p_m - governor_p_m[0]) * mechanical_base_mw, axis=1
    )

    # Total AGC command, split to generators and HVDC, and HVDC power change.
    agc_command = np.asarray(agc_values)
    afrr_to_generators = (1.0 - HVDC_SHARE) * agc_command
    afrr_to_hvdc = HVDC_SHARE * agc_command
    hvdc_change = hvdc_power_mw - hvdc_power_mw[0]

    # =========================================================================
    # RESPONSE METRICS AND TERMINAL SUMMARY
    # =========================================================================

    post_event = t >= EVENT_TIME
    post_idx = np.where(post_event)[0]
    nadir_idx = int(post_idx[np.argmin(coi_frequency[post_event])])
    final_window = t >= T_END - FINAL_AVERAGING_WINDOW_S

    print("COI-frequency nadir:", f"{coi_frequency[nadir_idx]:.5f} Hz")
    print("Time to nadir:", f"{t[nadir_idx] - EVENT_TIME:.3f} s")
    for t_check in (60.0, 120.0):
        k = int(np.searchsorted(t, EVENT_TIME + t_check))
        if k < len(t):
            print(f"COI frequency {t_check:.0f} s after event: {coi_frequency[k]:.4f} Hz")
    print(
        "Mean COI frequency in final 5 s:",
        f"{np.mean(coi_frequency[final_window]):.5f} Hz",
    )

    outside_band = post_event & (
        np.abs(coi_frequency - nominal_frequency) > RESTORATION_BAND_HZ
    )
    restoration_time = (
        float(t[np.where(outside_band)[0][-1]] - EVENT_TIME)
        if np.any(outside_band) else 0.0
    )
    print(
        f"Last time outside +-{RESTORATION_BAND_HZ * 1e3:.0f} mHz:",
        f"{restoration_time:.1f} s after the event",
    )
    print(
        "Mean total governor response in final 5 s:",
        f"{np.mean(total_governor_response[final_window]):.1f} MW",
    )
    print("Mean AGC command in final 5 s:", f"{np.mean(agc_command[final_window]):.1f} MW")
    for k, name in enumerate(vsc_names[hvdc_idx]):
        print(f"  {name}: final change {np.mean(hvdc_change[final_window, k]):+.1f} MW")

    return {
        "time": t,
        "nominal_frequency": nominal_frequency,
        "coi_frequency": coi_frequency,
        "total_governor_response": total_governor_response,
        "nadir": float(coi_frequency[nadir_idx]),
        "nadir_time": float(t[nadir_idx]),
        "agc_command": agc_command,
        "afrr_to_generators": afrr_to_generators,
        "afrr_to_hvdc": afrr_to_hvdc,
        "event_text": event_text,
        "nominal_disturbance_mw": nominal_disturbance_mw,
    }


def plot_reserve_overview(results):
    """Overview in the style of Statnett's reserve figure.

    Top:    power per reserve type (FCR and aFRR), MW.
    Bottom: system frequency with the 49.9 Hz normal-band limit.

    FCR is not measured directly. It is estimated as the total governor
    response minus the aFRR set-point sent to the generators. FFR and mFRR
    are not modelled and are therefore not shown.
    """

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================

    t = results["time"]
    f_n = results["nominal_frequency"]

    fcr = results["total_governor_response"] - results["afrr_to_generators"]
    total_response = results["total_governor_response"] + results["afrr_to_hvdc"]

    fig, (ax_p, ax_f) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8), gridspec_kw={"hspace": 0.12}
    )
    fig.suptitle(
        "Reserve activation after a disturbance – Nordic 45 (2025)\n"
        f"{results['event_text']}, T_i = {T_I:.0f} s, HVDC share = {HVDC_SHARE:.0%}"
    )

    # =========================================================================
    # POWER PER RESERVE TYPE
    # =========================================================================

    ax_p.plot(t, fcr, color="tab:blue", lw=2.5, label="FCR (governor droop)")
    ax_p.plot(
        t, results["agc_command"], color="tab:green", lw=2.5,
        label="aFRR (AGC)",
    )
    if HVDC_SHARE > 0:
        ax_p.plot(
            t, results["afrr_to_hvdc"], color="tab:purple", lw=1.5, ls="--",
            label="  of which via HVDC",
        )
    ax_p.plot(t, total_response, color="black", lw=1.0, label="Total response")
    ax_p.axhline(
        results["nominal_disturbance_mw"], color="gray", ls="--", lw=1,
        label="Disturbance",
    )
    ax_p.axhline(0.0, color="gray", lw=0.8)
    ax_p.set_ylabel("Power (MW)")
    ax_p.legend(loc="center right")

    # =========================================================================
    # SYSTEM FREQUENCY
    # =========================================================================

    ax_f.plot(t, results["coi_frequency"], color="tab:orange", lw=2.5, label="System frequency (COI)")
    ax_f.axhline(f_n, color="black", lw=1)
    ax_f.axhline(f_n - 0.1, color="gray", ls="--", lw=1)
    ax_f.text(t[-1], f_n - 0.1, " 49.9 Hz", va="center", ha="left", color="gray")
    ax_f.plot(
        results["nadir_time"], results["nadir"], "ko",
        label=f"Nadir {results['nadir']:.3f} Hz",
    )
    ax_f.set_ylabel("Frequency (Hz)")
    ax_f.set_xlabel("Time (s)")
    ax_f.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_f.legend(loc="lower right")

    # =========================================================================
    # PHASE MARKERS (30 s, 2 min, 15 min after the event)
    # =========================================================================

    for delay, label in ((30.0, "30 s"), (120.0, "2 min"), (900.0, "15 min")):
        t_mark = EVENT_TIME + delay
        if t_mark < t[-1]:
            for ax in (ax_p, ax_f):
                ax.axvline(t_mark, color="tab:red", ls="--", lw=1)
            ax_p.text(t_mark, 1.02, label, transform=ax_p.get_xaxis_transform(),
                      ha="center", va="bottom", color="tab:red")

    phase_labels = [("FCR", EVENT_TIME, EVENT_TIME + 120.0, "tab:blue"),
                    ("aFRR", EVENT_TIME + 120.0, min(EVENT_TIME + 900.0, t[-1]), "tab:green")]
    for text, start, end, color in phase_labels:
        if start < t[-1]:
            ax_f.text((start + end) / 2, 0.04, text, transform=ax_f.get_xaxis_transform(),
                      ha="center", va="bottom", color="white", fontweight="bold",
                      bbox=dict(boxstyle="square,pad=0.3", fc=color, ec="none"))

    for ax in (ax_p, ax_f):
        ax.axvline(EVENT_TIME, color="black", ls=":", lw=1)
        ax.set_xlim(0.0, t[-1])
        ax.grid(True, alpha=0.4)

    fig.align_ylabels((ax_p, ax_f))
    fig.subplots_adjust(left=0.08, right=0.93, top=0.88, bottom=0.08)


# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    results = run_simulation()
    plot_reserve_overview(results)
    plt.show()
