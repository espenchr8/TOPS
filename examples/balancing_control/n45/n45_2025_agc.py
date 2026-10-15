"""N45 disturbance with conventional FCR and a central frequency-based AGC (aFRR)."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import n45_2025 as model_data

# Make examples/balancing_control/common importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from agc import (
    add_agc_to_model,
    user_model_library,
    connect_coi_frequency,
    connect_hygov,
    connect_tgov1,
    connect_hvdc,
)


# =============================================================================
# STUDY SETTINGS
# =============================================================================

EVENT_TIME = 1.0
T_END = 300.0
MAX_STEP = 5e-3
FINAL_AVERAGING_WINDOW_S = 5.0
RESTORATION_BAND_HZ = 0.01          # Frequency counted as restored within +-10 mHz

# Disturbance: "load_step" or "hvdc_reduction"
DISTURBANCE = "load_step"

# Load step (constant-impedance load, defined at the pre-event voltage)
LOAD_NAME = "L5240-1"
LOAD_STEP_MW = 1000.0
LOAD_STEP_MVAR = 0.0

# HVDC import reduction (new import set-point in MW)
HVDC_EVENT_LINK = "NO_2-DE"
HVDC_EVENT_NEW_MW = 0.0

REGION_AREA_CODES = {
    "Norway": {11, 12, 13, 14, 15},
    "Sweden": {21, 22, 23, 24},
    "Finland": {31},
}


# =============================================================================
# AGC SETTINGS
# =============================================================================

AGC_ENABLED = True
T_F = 2.0          # Low-pass filter time constant (s)
T_I = 60.0         # Integral time (s). K_i = B / T_i
K_P = 0.0          # Proportional gain (MW/Hz). 0 = pure integral control

# Share of the AGC command delivered via HVDC (0 = hydro only, 1 = HVDC only)
HVDC_SHARE = 0.0

# Split of the HVDC share between the links (must sum to 1)
HVDC_LINK_SPLIT = {"NO_2-GB": 1.0}

# Hydro units taking part in the AGC (None = all HYGOV units).
# The hydro share is split proportionally to generator rating.
HYDRO_AGC_UNITS = None

# Thermal units taking part in the AGC (empty = none)
THERMAL_AGC_UNITS = []


# =============================================================================
# REGULATING STRENGTH FROM MODEL DATA
# =============================================================================

def regulating_strength_mw_per_hz(model):
    """Sum of droop contributions P_n / (R f0) over all governors."""
    f0 = float(model["f"])
    gen_table = model["generators"]["GEN"]
    header = list(gen_table[0])
    gen_rating = {
        row[header.index("name")]: float(row[header.index("S_n")])
        for row in gen_table[1:]
    }

    strength = {}
    for gov_type, table in model["gov"].items():
        gov_header = list(table[0])
        strength[gov_type] = sum(
            gen_rating[row[gov_header.index("gen")]]
            / (float(row[gov_header.index("R")]) * f0)
            for row in table[1:]
        )
    return strength


def run_simulation():

    # =========================================================================
    # MODEL LOADING, AGC AND INITIALIZATION
    # =========================================================================

    model = model_data.load()
    system_base_mva = float(model["base_mva"])
    nominal_frequency = float(model["f"])

    # Frequency bias B = regulating strength of the whole system (MW/Hz).
    strength = regulating_strength_mw_per_hz(model)
    bias_B = sum(strength.values())
    K_I = bias_B / T_I if AGC_ENABLED else 0.0

    # The AGC must be in the model before PowerSystemModel is created, so that
    # its states become part of the system state vector.
    add_agc_to_model(model, T_f=T_F, K_p=K_P if AGC_ENABLED else 0.0, K_i=K_I)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_model_library())
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]
    load = ps.loads["Load"]
    hygov = ps.gov["HYGOV"]
    tgov1 = ps.gov["TGOV1"]
    vsc = ps.vsc["VSC_SI"]
    agc = ps.agc["AGC"]


    # =========================================================================
    # AGC CONNECTIONS
    # =========================================================================

    gen_names = np.asarray(gen.par["name"], dtype=str)
    rating_mva = (
        np.asarray(gen.par["S_n"], dtype=float)
        * np.asarray(gen.par["N_par"], dtype=float)
    )
    gen_rating = dict(zip(gen_names, rating_mva))

    # Hydro participation, proportional to rating.
    hydro_units = (
        list(np.asarray(hygov.par["gen"], dtype=str))
        if HYDRO_AGC_UNITS is None else list(HYDRO_AGC_UNITS)
    )
    thermal_units = list(THERMAL_AGC_UNITS)
    synchronous_units = hydro_units + thermal_units
    synchronous_rating = sum(gen_rating[g] for g in synchronous_units)

    participation_sync = {
        g: (1.0 - HVDC_SHARE) * gen_rating[g] / synchronous_rating
        for g in synchronous_units
    }
    participation_hvdc = {
        link: HVDC_SHARE * split for link, split in HVDC_LINK_SPLIT.items()
    }

    connect_coi_frequency(agc, ps)
    connect_hygov(agc, ps, participation_sync)
    connect_tgov1(agc, ps, participation_sync)
    connect_hvdc(agc, ps, participation_hvdc)


    # =========================================================================
    # INITIALIZED MODEL OVERVIEW
    # =========================================================================

    vsc_names = np.asarray(vsc.par["name"], dtype=str)
    is_wind_vsc = np.char.startswith(vsc_names, "WG")
    hvdc_idx = np.where(~is_wind_vsc)[0]
    hvdc_names = vsc_names[hvdc_idx]
    vsc_s_n = np.asarray(vsc.par["S_n"], dtype=float)

    print("\nInitialized N45 model overview")
    print("------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Synchronous generators:", gen.n_units)
    print("HYGOV hydro units:", hygov.n_units)
    print("TGOV1 thermal units:", tgov1.n_units)
    print("Wind-power VSC units:", int(np.sum(is_wind_vsc)))
    print("HVDC VSC units:", len(hvdc_idx))
    print("Dynamic states:", ps.n_states)

    print("\nAGC settings")
    print("------------")
    print("AGC enabled:", AGC_ENABLED)
    for gov_type, value in strength.items():
        print(f"  Regulating strength {gov_type}: {value:.0f} MW/Hz")
    print(f"  Frequency bias B: {bias_B:.0f} MW/Hz")
    print(f"  T_f = {T_F:.1f} s, T_i = {T_I:.0f} s, K_p = {K_P:.0f} MW/Hz, K_i = {K_I:.1f} MW/(Hz s)")
    print(f"  Units in AGC: {len(hydro_units)} hydro, {len(thermal_units)} thermal")
    print(f"  HVDC share: {HVDC_SHARE:.2f} {participation_hvdc}")


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
    print("Maximum initial derivative:", f"{max_initial_derivative:.6e}")


    # =========================================================================
    # GENERATOR AND REGIONAL DATA
    # =========================================================================

    inertia_weights = np.asarray(gen.par["H"], dtype=float) * rating_mva
    mechanical_base_mw = rating_mva * np.asarray(gen.par["PF_n"], dtype=float)
    gen_index = {name: i for i, name in enumerate(gen_names)}
    governed_indices = np.asarray(
        [gen_index[str(g)] for gov in (hygov, tgov1) for g in gov.par["gen"]],
        dtype=int,
    )

    bus_table = model["buses"]
    bus_header = list(bus_table[0])
    bus_area = {
        str(row[bus_header.index("name")]): int(row[bus_header.index("Area")])
        for row in bus_table[1:]
    }
    gen_area = np.asarray(
        [bus_area[bus] for bus in np.asarray(gen.par["bus"], dtype=str)], dtype=int
    )
    region_indices = {
        region: np.where(np.isin(gen_area, list(codes)))[0]
        for region, codes in REGION_AREA_CODES.items()
    }


    # =========================================================================
    # DISTURBANCE
    # =========================================================================

    if DISTURBANCE == "load_step":
        load_names = np.asarray(load.par["name"], dtype=str)
        load_idx = int(np.where(load_names == LOAD_NAME)[0][0])
        load_bus_idx = int(load.bus_idx_red["terminal"][load_idx])

        # Constant-impedance load: delta_Y = conj(delta_S) / |V_0|^2
        delta_s_pu = (LOAD_STEP_MW + 1j * LOAD_STEP_MVAR) / system_base_mva
        delta_y = np.conj(delta_s_pu) / abs(v_initial[load_bus_idx]) ** 2
        event_text = f"+{LOAD_STEP_MW:.0f} MW load step at {LOAD_NAME}"

    elif DISTURBANCE == "hvdc_reduction":
        event_link_idx = int(np.where(vsc_names == HVDC_EVENT_LINK)[0][0])
        initial_link_mw = (
            float(vsc._input_values["p_ref"][event_link_idx]) * vsc_s_n[event_link_idx]
        )
        event_text = (
            f"HVDC import {HVDC_EVENT_LINK} reduced "
            f"{initial_link_mw:.0f} -> {HVDC_EVENT_NEW_MW:.0f} MW"
        )

    else:
        raise ValueError(f"Unknown disturbance type: {DISTURBANCE}")

    print(f"Disturbance at t={EVENT_TIME:.1f} s: {event_text}")


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
        speed_values.append(np.asarray(gen.speed(x, v), dtype=float).copy())
        p_m_values.append(np.asarray(gen.P_m(x, v), dtype=float).copy())
        agc_values.append(float(agc.output(x, v)[0]))
        hvdc_values.append(
            np.asarray(vsc.p_e(x, v), dtype=float)[hvdc_idx] * vsc_s_n[hvdc_idx]
        )

    time_values = [0.0]
    speed_values, p_m_values, agc_values, hvdc_values = [], [], [], []
    store(ps.x0, v_initial)


    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    event_applied = False
    next_progress = 10
    t_wall_start = time.perf_counter()

    while solver.t < T_END:
        if solver.t >= EVENT_TIME and not event_applied:
            if DISTURBANCE == "load_step":
                load.y_load[load_idx] += delta_y
                ps.y_bus_red_mod[load_bus_idx, load_bus_idx] += delta_y
            else:
                vsc.set_input(
                    "p_ref",
                    HVDC_EVENT_NEW_MW / vsc_s_n[event_link_idx],
                    idx=event_link_idx,
                )
            solver.v[:] = ps.solve_algebraic(solver.t, solver.y)
            event_applied = True

        solver.step()
        time_values.append(float(solver.t))
        store(solver.y, solver.v)

        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    wall_time = time.perf_counter() - t_wall_start


    # =========================================================================
    # RESULT PROCESSING
    # =========================================================================

    t = np.asarray(time_values)
    speed = np.asarray(speed_values)
    p_m = np.asarray(p_m_values)
    agc_command = np.asarray(agc_values)
    hvdc_power = np.asarray(hvdc_values)

    coi_frequency = nominal_frequency * (
        1.0 + np.average(speed, axis=1, weights=inertia_weights)
    )
    regional_frequency = {
        region: nominal_frequency
        * (1.0 + np.average(speed[:, idx], axis=1, weights=inertia_weights[idx]))
        for region, idx in region_indices.items()
    }

    # Mechanical power change of all governed units (FCR + aFRR), in MW.
    governor_response = np.sum(
        (p_m[:, governed_indices] - p_m[0, governed_indices])
        * mechanical_base_mw[governed_indices],
        axis=1,
    )

    # aFRR set-points sent to the synchronous units and to HVDC.
    afrr_to_generators = (1.0 - HVDC_SHARE) * agc_command
    afrr_to_hvdc = HVDC_SHARE * agc_command

    hvdc_change = hvdc_power - hvdc_power[0]


    # =========================================================================
    # RESPONSE METRICS AND TERMINAL SUMMARY
    # =========================================================================

    post_event = t >= EVENT_TIME
    post_idx = np.where(post_event)[0]
    nadir_idx = int(post_idx[np.argmin(coi_frequency[post_event])])
    final_window = t >= T_END - FINAL_AVERAGING_WINDOW_S

    outside_band = post_event & (
        np.abs(coi_frequency - nominal_frequency) > RESTORATION_BAND_HZ
    )
    restoration_time = (
        float(t[np.where(outside_band)[0][-1]] - EVENT_TIME)
        if np.any(outside_band) else 0.0
    )

    print("\nResult")
    print("------")
    print("COI-frequency nadir:", f"{coi_frequency[nadir_idx]:.4f} Hz")
    print("Time to nadir:", f"{t[nadir_idx] - EVENT_TIME:.2f} s")
    for t_check in (30.0, 60.0, 120.0):
        if t_check < T_END:
            k = int(np.searchsorted(t, EVENT_TIME + t_check))
            print(f"COI frequency {t_check:.0f} s after event: {coi_frequency[k]:.4f} Hz")
    print(
        "Mean COI frequency in final 5 s:",
        f"{np.mean(coi_frequency[final_window]):.4f} Hz",
    )
    print(
        f"Last time outside +-{RESTORATION_BAND_HZ * 1e3:.0f} mHz:",
        f"{restoration_time:.1f} s after the event",
    )
    print("Final AGC command:", f"{np.mean(agc_command[final_window]):.1f} MW")
    print(
        "Final governor response (FCR + aFRR):",
        f"{np.mean(governor_response[final_window]):.1f} MW",
    )
    for k, name in enumerate(hvdc_names):
        print(f"  {name}: final change {np.mean(hvdc_change[final_window, k]):+.1f} MW")
    print(f"Wall-clock time: {wall_time:.0f} s (real-time factor {wall_time / T_END:.2f})")

    return {
        "time": t,
        "nominal_frequency": nominal_frequency,
        "coi_frequency": coi_frequency,
        "regional_frequency": regional_frequency,
        "governor_response": governor_response,
        "agc_command": agc_command,
        "afrr_to_generators": afrr_to_generators,
        "afrr_to_hvdc": afrr_to_hvdc,
        "hvdc_change": hvdc_change,
        "hvdc_names": hvdc_names,
        "nadir": float(coi_frequency[nadir_idx]),
        "nadir_time": float(t[nadir_idx]),
        "event_text": event_text,
    }


def plot_results(results):

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================

    t = results["time"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 12))
    fig.suptitle(
        "Nordic 45 (2025) – FCR and frequency-based AGC (aFRR)\n"
        f"{results['event_text']}, t = {EVENT_TIME:.1f} s, "
        f"T_i = {T_I:.0f} s, HVDC share = {HVDC_SHARE:.0%}",
        y=0.985,
    )

    # =========================================================================
    # SYSTEM AND REGIONAL FREQUENCIES
    # =========================================================================

    axes[0].plot(t, results["coi_frequency"], "k", lw=2, label="System COI")
    for region, frequency in results["regional_frequency"].items():
        axes[0].plot(t, frequency, lw=1.0, label=region)
    axes[0].axhline(results["nominal_frequency"], color="gray", ls=":", label="Nominal")
    axes[0].plot(
        results["nadir_time"],
        results["nadir"],
        "ko",
        label=f"Nadir {results['nadir']:.3f} Hz",
    )
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].legend(ncol=3)

    # =========================================================================
    # GOVERNOR RESPONSE: FCR + aFRR
    # =========================================================================

    axes[1].plot(
        t, results["governor_response"], "k", lw=2,
        label="Governor response (FCR + aFRR)",
    )
    axes[1].plot(
        t, results["afrr_to_generators"], color="tab:green", ls="--",
        label="aFRR set-point to generators",
    )
    axes[1].fill_between(
        t, results["afrr_to_generators"], results["governor_response"],
        color="tab:blue", alpha=0.15, label="FCR part (difference)",
    )
    axes[1].axhline(0.0, color="gray", ls=":")
    axes[1].set_ylabel("Power (MW)")
    axes[1].legend(ncol=3)

    # =========================================================================
    # AGC COMMAND
    # =========================================================================

    axes[2].plot(t, results["agc_command"], "k", lw=2, label="Total AGC command")
    axes[2].plot(t, results["afrr_to_generators"], color="tab:green", label="To generators")
    axes[2].plot(t, results["afrr_to_hvdc"], color="tab:red", label="To HVDC")
    axes[2].axhline(0.0, color="gray", ls=":")
    axes[2].set_ylabel("AGC command (MW)")
    axes[2].legend(ncol=3)

    # =========================================================================
    # HVDC POWER
    # =========================================================================

    for k, name in enumerate(results["hvdc_names"]):
        axes[3].plot(t, results["hvdc_change"][:, k], label=name)
    axes[3].axhline(0.0, color="gray", ls=":")
    axes[3].set_ylabel("$\\Delta P_{HVDC}$ (MW)\n(+ = more import)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=4)

    # =========================================================================
    # FINAL FIGURE LAYOUT
    # =========================================================================

    for ax in axes:
        ax.axvline(EVENT_TIME, color="black", ls="--", lw=1)
        ax.set_xlim(0.0, T_END)
        ax.grid(True)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.065, top=0.91, hspace=0.30)
    fig.align_ylabels(axes)
    plt.show()


# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    plot_results(run_simulation())
