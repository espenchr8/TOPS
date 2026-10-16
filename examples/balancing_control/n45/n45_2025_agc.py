"""N45 disturbance with conventional FCR and a central frequency-based AGC (aFRR).

Who regulates in this simulation (the whole N45 network is simulated):

    Unit                     FCR (droop)   In bias B   AGC (aFRR)
    HYGOV hydro (36)         yes           yes         yes (default: all)
    TGOV1 thermal (11)       yes           yes         only if listed in THERMAL_AGC_UNITS
    HVDC VSC (4)             no            no          only if HVDC_SHARE > 0
    Wind VSC (29)            no            no          no (constant power)
    Loads                    no            no          -

Same structure as n45_2025_load_step_fcr.py. The additions are marked "AGC".
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
LOAD_NAME = "L5240-1"
LOAD_STEP_MW = 1000.0
LOAD_STEP_MVAR = 0.0
T_END = 300.0
MAX_STEP = 5e-3
ROCOF_WINDOW_S = 0.5
FINAL_AVERAGING_WINDOW_S = 5.0

REGION_AREA_CODES = {
    "Norway": {11, 12, 13, 14, 15},
    "Sweden": {21, 22, 23, 24},
    "Finland": {31},
}

# Disturbance: "load_step" (settings above) or "hvdc_reduction" (below)
DISTURBANCE = "load_step"
HVDC_EVENT_LINK = "NO_2-DE"
HVDC_EVENT_NEW_MW = 0.0
RESTORATION_BAND_HZ = 0.01


# =============================================================================
# AGC SETTINGS
# =============================================================================

AGC_ENABLED = True
T_F = 2.0                       # Low-pass filter time constant (s)
T_I = 180.0                      # Integral time (s); K_i = B / T_i
K_P = 0.0                       # Proportional gain (MW/Hz); 0 = pure integral
HYDRO_AGC_UNITS = None          # None = all HYGOV units
THERMAL_AGC_UNITS = []          # e.g. ["G3000-1"]; empty = no thermal units
HVDC_SHARE = 0.0                # Share of the AGC command sent via HVDC
HVDC_LINK_SPLIT = {"NO_2-GB": 1.0}

# Plots: reserve overview (Statnett style) and/or the detailed 5-panel figure
SHOW_RESERVE_OVERVIEW = True
SHOW_DETAILED_PLOT = True


def run_simulation():

    # =========================================================================
    # MODEL LOADING AND INITIALIZATION
    # =========================================================================

    model = model_data.load()
    system_base_mva = float(model["base_mva"])
    nominal_frequency = float(model["f"])

    # AGC: frequency bias B = regulating strength (MW/Hz), from the governors.
    # The AGC must be added before PowerSystemModel so that its states become
    # part of the system state vector.
    strength = regulating_strength_mw_per_hz(model)
    bias_B = sum(strength.values())
    K_I = bias_B / T_I if AGC_ENABLED else 0.0
    add_agc_to_model(model, T_f=T_F, K_p=K_P if AGC_ENABLED else 0.0, K_i=K_I)

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_model_library())
    ps.init_dyn_sim()

    # AGC: connect frequency measurement and unit set-points.
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
    hygov = ps.gov["HYGOV"]
    tgov1 = ps.gov["TGOV1"]
    vsc = ps.vsc.get("VSC_SI") if hasattr(ps, "vsc") else None


    # =========================================================================
    # INITIALIZED MODEL OVERVIEW
    # =========================================================================

    # Count the units that TOPS actually created during ps.init_dyn_sim().
    generator_count = sum(
        generator_model.n_units
        for generator_model in ps.gen.values()
    )

    hygov_count = hygov.n_units
    tgov1_count = tgov1.n_units
    wind_vsc_count = 0
    hvdc_vsc_count = 0

    if hasattr(ps, "vsc"):
        for vsc_model in ps.vsc.values():
            names = np.asarray(vsc_model.par["name"], dtype=str)
            is_wind_vsc = np.char.startswith(names, "WG")
            wind_vsc_count += int(np.sum(is_wind_vsc))
            hvdc_vsc_count += int(np.sum(~is_wind_vsc))

    print("\nInitialized N45 model overview")
    print("------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Buses:", len(ps.buses))
    print("Synchronous generators:", generator_count)
    print("HYGOV hydro units:", hygov_count)
    print("TGOV1 thermal units:", tgov1_count)
    print("Wind-power VSC units:", wind_vsc_count)
    print("HVDC VSC units:", hvdc_vsc_count)
    print("Dynamic states:", ps.n_states)

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

    # Confirm that the initialized operating point is close to equilibrium.
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
    # GENERATOR AND REGIONAL DATA
    # =========================================================================

    # Generator data used for COI frequency and mechanical-power conversion.
    gen_names = np.asarray(gen.par["name"], dtype=str)
    gen_buses = np.asarray(gen.par["bus"], dtype=str)
    rating_mva = (
        np.asarray(gen.par["S_n"], dtype=float)
        * np.asarray(gen.par["N_par"], dtype=float)
    )
    inertia_weights = np.asarray(gen.par["H"], dtype=float) * rating_mva
    mechanical_base_mw = rating_mva * np.asarray(gen.par["PF_n"], dtype=float)
    gen_index = {name: i for i, name in enumerate(gen_names)}

    hygov_indices = np.asarray(
        [gen_index[str(name)] for name in hygov.par["gen"]], dtype=int
    )
    tgov1_indices = np.asarray(
        [gen_index[str(name)] for name in tgov1.par["gen"]], dtype=int
    )

    # Associate every generator with Norway, Sweden or Finland.
    bus_table = model["buses"]
    bus_header = list(bus_table[0])
    bus_area = {
        str(row[bus_header.index("name")]): int(row[bus_header.index("Area")])
        for row in bus_table[1:]
    }
    gen_area = np.asarray([bus_area[bus] for bus in gen_buses], dtype=int)
    region_indices = {
        region: np.where(np.isin(gen_area, list(area_codes)))[0]
        for region, area_codes in REGION_AREA_CODES.items()
    }

    # =========================================================================
    # LOAD DISTURBANCE
    # =========================================================================

    # Locate the load that receives the disturbance.
    load_names = np.asarray(load.par["name"], dtype=str)
    matches = np.where(load_names == LOAD_NAME)[0]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one load named {LOAD_NAME}.")
    load_idx = int(matches[0])
    load_bus_name = str(load.par["bus"][load_idx])
    load_bus_idx = int(load.bus_idx_red["terminal"][load_idx])
    initial_load_voltage = complex(v_initial[load_bus_idx])

    # The load model is constant impedance. Define the requested P/Q step at
    # the pre-event voltage and convert it to an admittance increment:
    # S = |V|^2 * conj(Y), hence delta_Y = conj(delta_S) / |V_0|^2.
    delta_s_pu = (LOAD_STEP_MW + 1j * LOAD_STEP_MVAR) / system_base_mva
    delta_y = np.conj(delta_s_pu) / abs(initial_load_voltage) ** 2

    # AGC: alternative disturbance, reduced HVDC import on one link.
    vsc_names = np.asarray(vsc.par["name"], dtype=str)
    vsc_s_n = np.asarray(vsc.par["S_n"], dtype=float)
    hvdc_idx = np.where(~np.char.startswith(vsc_names, "WG"))[0]
    if DISTURBANCE == "load_step":
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

    # =========================================================================
    # INITIAL POWER VALUES
    # =========================================================================

    # Initial values are subtracted later to obtain incremental responses.
    governor_p_m = gen.P_m
    initial_governor_p_m = np.asarray(
        governor_p_m(ps.x0, v_initial), dtype=float
    ).copy()
    initial_load_power_mw = (
        np.asarray(load.p(ps.x0, v_initial), dtype=float) * system_base_mva
    )

    initial_vsc_power_mw = None
    if vsc is not None:
        initial_vsc_power_mw = (
            np.asarray(vsc.p_e(ps.x0, v_initial), dtype=float)
            * np.asarray(vsc.par["S_n"], dtype=float)
        )

    print("Imported model-data file:", model_data.__file__)
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

    time_values = [0.0]
    speed_values = [np.asarray(gen.speed(ps.x0, v_initial)).copy()]
    governor_values = [initial_governor_p_m.copy()]
    load_power_values = [initial_load_power_mw.copy()]
    vsc_power_values = (
        [initial_vsc_power_mw.copy()] if vsc is not None else []
    )
    agc_values = [float(agc.output(ps.x0, v_initial)[0])]  # AGC

    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    event_applied = False
    next_progress = 10

    while solver.t < T_END:
        # Apply the admittance step once, at the selected event time.
        if solver.t >= EVENT_TIME and not event_applied:
            if DISTURBANCE == "load_step":
                load.y_load[load_idx] += delta_y
                ps.y_bus_red_mod[load_bus_idx, load_bus_idx] += delta_y
            else:
                # AGC: reduced HVDC import (scheduled set-point change)
                vsc.set_input(
                    "p_ref", HVDC_EVENT_NEW_MW / vsc_s_n[event_link_idx], idx=event_link_idx
                )
            solver.v[:] = ps.solve_algebraic(solver.t, solver.y)
            event_applied = True

            actual_step = (
                float(load.p(solver.y, solver.v)[load_idx]) * system_base_mva
                - initial_load_power_mw[load_idx]
            )
            print("Actual immediate load increase:", f"{actual_step:.3f} MW")

        solver.step()

        time_values.append(float(solver.t))
        speed_values.append(np.asarray(gen.speed(solver.y, solver.v)).copy())
        governor_values.append(
            np.asarray(governor_p_m(solver.y, solver.v), dtype=float).copy()
        )
        load_power_values.append(
            np.asarray(load.p(solver.y, solver.v), dtype=float) * system_base_mva
        )
        if vsc is not None:
            vsc_power_values.append(
                np.asarray(vsc.p_e(solver.y, solver.v), dtype=float)
                * np.asarray(vsc.par["S_n"], dtype=float)
            )
        agc_values.append(float(agc.output(solver.y, solver.v)[0]))  # AGC

        # Terminal progress indicator; this does not affect the simulation.
        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    # =========================================================================
    # RESULT PROCESSING
    # =========================================================================

    # Convert stored simulation results to arrays.
    t = np.asarray(time_values)
    speed = np.asarray(speed_values)
    governor_p_m = np.asarray(governor_values)
    load_power_mw = np.asarray(load_power_values)

    # System and regional center-of-inertia frequencies.
    generator_frequency = nominal_frequency * (1.0 + speed)
    coi_frequency = nominal_frequency * (
        1.0 + np.average(speed, axis=1, weights=inertia_weights)
    )
    regional_frequency = {
        region: nominal_frequency
        * (
            1.0
            + np.average(
                speed[:, idx], axis=1, weights=inertia_weights[idx]
            )
        )
        for region, idx in region_indices.items()
    }

    # Numerical COI RoCoF, smoothed with a 100 ms moving average.
    rocof_raw = np.gradient(coi_frequency, t)
    samples_100ms = max(1, int(round(0.1 / MAX_STEP)))
    rocof = np.convolve(
        rocof_raw,
        np.ones(samples_100ms) / samples_100ms,
        mode="same",
    )

    # Incremental mechanical power is interpreted as governor/FCR response.
    governor_response_mw = (
        governor_p_m - initial_governor_p_m[np.newaxis, :]
    ) * mechanical_base_mw[np.newaxis, :]
    hygov_response = np.sum(governor_response_mw[:, hygov_indices], axis=1)
    tgov1_response = np.sum(governor_response_mw[:, tgov1_indices], axis=1)
    total_fcr_response = np.sum(governor_response_mw, axis=1)

    actual_load_increase = (
        load_power_mw[:, load_idx] - initial_load_power_mw[load_idx]
    )

    # Change in ALL loads, including the negative loads representing HVDC import.
    total_load_change = np.sum(
        load_power_mw - initial_load_power_mw[np.newaxis, :], axis=1
    )

    aggregate_vsc_change = np.zeros_like(t)
    if vsc is not None:
        vsc_power_mw = np.asarray(vsc_power_values)
        aggregate_vsc_change = np.sum(
            vsc_power_mw - initial_vsc_power_mw[np.newaxis, :], axis=1
        )

    # AGC: total command, split to generators and HVDC, and HVDC power change.
    agc_command = np.asarray(agc_values)
    afrr_to_generators = (1.0 - HVDC_SHARE) * agc_command
    afrr_to_hvdc = HVDC_SHARE * agc_command
    hvdc_change = (
        vsc_power_mw[:, hvdc_idx] - initial_vsc_power_mw[np.newaxis, hvdc_idx]
    )

    # =========================================================================
    # RESPONSE METRICS AND TERMINAL SUMMARY
    # =========================================================================

    # Key response metrics.
    post_event = t >= EVENT_TIME
    post_idx = np.where(post_event)[0]
    nadir_idx = int(post_idx[np.argmin(coi_frequency[post_event])])

    event_idx = int(np.searchsorted(t, EVENT_TIME))
    rocof_end_idx = int(np.searchsorted(t, EVENT_TIME + ROCOF_WINDOW_S))
    average_rocof = (
        coi_frequency[rocof_end_idx] - coi_frequency[event_idx]
    ) / (t[rocof_end_idx] - t[event_idx])

    final_window = t >= T_END - FINAL_AVERAGING_WINDOW_S
    print("COI-frequency nadir:", f"{coi_frequency[nadir_idx]:.5f} Hz")
    print("Time to nadir:", f"{t[nadir_idx] - EVENT_TIME:.3f} s")
    print("Average RoCoF over first 0.5 s:", f"{average_rocof:+.5f} Hz/s")
    print(
        "Mean COI frequency in final 5 s:",
        f"{np.mean(coi_frequency[final_window]):.5f} Hz",
    )
    print(
        "Mean total governor/FCR response in final 5 s:",
        f"{np.mean(total_fcr_response[final_window]):.2f} MW",
    )
    print(
        "Mean total load change in final 5 s:",
        f"{np.mean(total_load_change[final_window]):.2f} MW",
    )

    # AGC: restoration metrics
    outside_band = post_event & (
        np.abs(coi_frequency - nominal_frequency) > RESTORATION_BAND_HZ
    )
    restoration_time = (
        float(t[np.where(outside_band)[0][-1]] - EVENT_TIME)
        if np.any(outside_band) else 0.0
    )
    for t_check in (60.0, 120.0):
        k = int(np.searchsorted(t, EVENT_TIME + t_check))
        if k < len(t):
            print(f"COI frequency {t_check:.0f} s after event: {coi_frequency[k]:.4f} Hz")
    print(
        f"Last time outside +-{RESTORATION_BAND_HZ * 1e3:.0f} mHz:",
        f"{restoration_time:.1f} s after the event",
    )
    print("Mean AGC command in final 5 s:", f"{np.mean(agc_command[final_window]):.1f} MW")
    for k, name in enumerate(vsc_names[hvdc_idx]):
        print(f"  {name}: final change {np.mean(hvdc_change[final_window, k]):+.1f} MW")

    return {
        "time": t,
        "nominal_frequency": nominal_frequency,
        "coi_frequency": coi_frequency,
        "regional_frequency": regional_frequency,
        "rocof": rocof,
        "hygov_response": hygov_response,
        "tgov1_response": tgov1_response,
        "total_fcr_response": total_fcr_response,
        "actual_load_increase": actual_load_increase,
        "total_load_change": total_load_change,
        "aggregate_vsc_change": aggregate_vsc_change,
        "nadir": float(coi_frequency[nadir_idx]),
        "nadir_time": float(t[nadir_idx]),
        # AGC
        "agc_command": agc_command,
        "afrr_to_generators": afrr_to_generators,
        "afrr_to_hvdc": afrr_to_hvdc,
        "hvdc_change": hvdc_change,
        "hvdc_names": vsc_names[hvdc_idx],
        "event_text": event_text,
        "nominal_disturbance_mw": nominal_disturbance_mw,
    }


def plot_results(results):

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================

    t = results["time"]
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(13, 15))
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
        axes[0].plot(t, frequency, lw=1.2, label=region)
    axes[0].axhline(
        results["nominal_frequency"], color="gray", ls=":", label="Nominal"
    )
    axes[0].plot(
        results["nadir_time"],
        results["nadir"],
        "ko",
        label=f"Nadir {results['nadir']:.3f} Hz",
    )
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[0].legend(ncol=3)

    # =========================================================================
    # RATE OF CHANGE OF FREQUENCY
    # =========================================================================

    axes[1].plot(t, results["rocof"], color="tab:purple", label="COI RoCoF (100 ms mean)")
    axes[1].axhline(0.0, color="gray", ls=":")
    axes[1].set_ylabel("RoCoF (Hz/s)")
    axes[1].legend()

    # =========================================================================
    # GOVERNOR AND FCR RESPONSE
    # =========================================================================

    axes[2].plot(t, results["hygov_response"], label="HYGOV (hydro)")
    axes[2].plot(t, results["tgov1_response"], label="TGOV1 (thermal)")
    axes[2].plot(t, results["total_fcr_response"], "k", lw=2, label="Total governor (FCR + aFRR)")
    axes[2].plot(t, results["afrr_to_generators"], color="tab:green", ls="--", label="aFRR set-point (AGC)")
    axes[2].axhline(results["nominal_disturbance_mw"], color="gray", ls=":", label="Nominal disturbance")
    axes[2].set_ylabel("Power response (MW)")
    axes[2].legend(ncol=2)

    # =========================================================================
    # LOAD AND VSC POWER CHANGES
    # =========================================================================

    axes[3].plot(t, results["actual_load_increase"], label="Actual load increase")
    axes[3].plot(t, results["total_load_change"], label="Total load change (all loads)")
    axes[3].plot(t, results["aggregate_vsc_change"], label="Aggregate VSC response")
    axes[3].axhline(results["nominal_disturbance_mw"], color="gray", ls=":", label="Nominal disturbance")
    axes[3].axhline(0.0, color="gray", ls=":")
    axes[3].set_ylabel("Power change (MW)")
    axes[3].legend(ncol=2)

    # =========================================================================
    # AGC COMMAND AND HVDC POWER
    # =========================================================================

    axes[4].plot(t, results["agc_command"], "k", lw=2, label="Total AGC command")
    axes[4].plot(t, results["afrr_to_generators"], color="tab:green", ls="--", label="AGC to generators")
    axes[4].plot(t, results["afrr_to_hvdc"], color="tab:red", ls="--", label="AGC to HVDC")
    for k, name in enumerate(results["hvdc_names"]):
        axes[4].plot(t, results["hvdc_change"][:, k], lw=1.2, label=f"HVDC {name} (actual)")
    axes[4].axhline(0.0, color="gray", ls=":")
    axes[4].set_ylabel("Power (MW)")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(ncol=4)

    # =========================================================================
    # FINAL FIGURE LAYOUT
    # =========================================================================

    for ax in axes:
        ax.axvline(EVENT_TIME, color="black", ls="--", lw=1)
        ax.set_xlim(0.0, T_END)
        ax.grid(True)

    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.065, top=0.91, hspace=0.30
    )
    fig.align_ylabels(axes)
    plt.show()


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

    fcr = results["total_fcr_response"] - results["afrr_to_generators"]
    total_response = results["total_fcr_response"] + results["afrr_to_hvdc"]

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
    if SHOW_RESERVE_OVERVIEW:
        plot_reserve_overview(results)
    if SHOW_DETAILED_PLOT:
        plot_results(results)
    else:
        plt.show()
