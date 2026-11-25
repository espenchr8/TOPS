"""N45 disturbance with conventional FCR and a central frequency-based AGC (aFRR).

Who regulates in this simulation (the whole N45 network is simulated):

    Unit                     FCR (droop)   In bias B   AGC (aFRR)
    HYGOV hydro (36)         yes           yes         yes (default: all)
    TGOV1 thermal (11)       yes           yes         only if listed in THERMAL_AGC_UNITS
    HVDC VSC (4)             no            no          only if HVDC_SHARE > 0
    Wind VSC (29)            no            no          no (constant power)
    Loads                    no            no          -

Disturbances: a load step, a reduced HVDC import on one link, or a
net-load variation (time series) made by changing the wind-power set-points.
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

EVENT_TIME = 1.0    #Forstyrrelsen skjer ved t = 1
T_END = 600.0       #Simuleringen varer i 600 s (10 min).
MAX_STEP = 5e-3
FINAL_AVERAGING_WINDOW_S = 5.0
RESTORATION_BAND_HZ = 0.01  #Frekvensen regnes som tilbake når den holder seg innenfor ±10 mHz av 50 Hz.

# Disturbance: "load_step", "hvdc_reduction" or "wind_variation"
DISTURBANCE = "wind_variation"

LOAD_NAME = "L5240-1"           # load_step
LOAD_STEP_MW = 1000.0

HVDC_EVENT_LINK = "NO_2-DE"     # hvdc_reduction #Ved HVDC-reduksjon: importen på NO_2-DE settes fra 900 til 0 MW.
HVDC_EVENT_NEW_MW = 0.0

# wind_variation: net-load increase = reduction of wind power. Shared between
# all wind VSC in proportion to their output. Ramp from EVENT_TIME, then held,
# with random fluctuations on top (same seed = same profile in every case).
WIND_VARIATION_MW = 500.0       # Size of the ramp (MW)
WIND_RAMP_S = 240.0             # Duration of the ramp (s)
WIND_FLUCTUATION_MW = 50.0      # Peak size of the fluctuations (MW)
WIND_FLUCTUATION_T_S = 10.0     # Time constant of the fluctuations (s)
RANDOM_SEED = 1

REGION_AREA_CODES = {
    "Norway": {11, 12, 13, 14, 15},
    "Sweden": {21, 22, 23, 24},
    "Finland": {31},
}

# Plots: reserve overview (Statnett style) and/or the detailed 5-panel figure
SHOW_RESERVE_OVERVIEW = True
SHOW_DETAILED_PLOT = True

# Case comparison: run case 0 (FCR only, AGC off) and case 1 (FCR + aFRR with
# the AGC settings below) one after the other, plot them together and print a
# table. Takes about twice as long. The two plots above are then not shown.
RUN_CASE_COMPARISON = True


# =============================================================================
# AGC SETTINGS
# =============================================================================

AGC_ENABLED = True
T_F = 2.0                       # Low-pass filter time constant (s)
T_I = 100.0                     # Integral time (s); K_i = B / T_i
K_P = 0.0                       # Proportional gain (MW/Hz); 0 = pure integral
HYDRO_AGC_UNITS = None          # None = all HYGOV units
THERMAL_AGC_UNITS = []          # e.g. ["G3000-1"]; empty = no thermal units
HVDC_SHARE = 0.0               # Share of the AGC command sent via HVDC
HVDC_LINK_SPLIT = {"NO_2-GB": 1.0}

# Text for plot titles
AGC_TEXT = (
    f"T_i = {T_I:.0f} s, HVDC share = {HVDC_SHARE:.0%}" if AGC_ENABLED else "AGC off (FCR only)"
)


def wind_variation_profile(t_end):
    """Net-load increase (MW) as a function of time for "wind_variation".

    A linear ramp of WIND_VARIATION_MW over WIND_RAMP_S from EVENT_TIME, then
    held. Fluctuations: random noise filtered with time constant
    WIND_FLUCTUATION_T_S (1 s grid), scaled to a peak of WIND_FLUCTUATION_MW
    and faded in over the first 30 s. Returns a function of time.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    t_grid = np.arange(0.0, t_end + 2.0, 1.0)
    a = np.exp(-1.0 / WIND_FLUCTUATION_T_S)
    noise = np.zeros_like(t_grid)
    for k in range(1, len(t_grid)):
        noise[k] = a * noise[k - 1] + np.sqrt(1.0 - a**2) * rng.standard_normal()
    noise *= WIND_FLUCTUATION_MW / np.max(np.abs(noise))
    noise *= np.clip((t_grid - EVENT_TIME) / 30.0, 0.0, 1.0)

    ramp = WIND_VARIATION_MW * np.clip((t_grid - EVENT_TIME) / WIND_RAMP_S, 0.0, 1.0)
    profile = ramp + noise
    return lambda t: np.interp(t, t_grid, profile)


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

    # Detailed plot: generators behind each governor type, and per region.
    gen_index = {name: i for i, name in enumerate(np.asarray(gen.par["name"], dtype=str))}
    hygov_indices = np.asarray([gen_index[str(g)] for g in ps.gov["HYGOV"].par["gen"]], dtype=int)
    tgov1_indices = np.asarray([gen_index[str(g)] for g in ps.gov["TGOV1"].par["gen"]], dtype=int)

    bus_header = list(model["buses"][0])
    bus_area = {
        str(row[bus_header.index("name")]): int(row[bus_header.index("Area")])
        for row in model["buses"][1:]
    }
    gen_area = np.asarray([bus_area[str(b)] for b in gen.par["bus"]], dtype=int)
    region_indices = {
        region: np.where(np.isin(gen_area, list(codes)))[0]
        for region, codes in REGION_AREA_CODES.items()
    }

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
    elif DISTURBANCE == "wind_variation":
        # All wind VSC, shared in proportion to their initial output.
        wind_idx = np.where(np.char.startswith(vsc_names, "WG"))[0]
        wind_p_ref_0 = np.asarray(vsc._input_values["p_ref"], dtype=float)[wind_idx].copy()
        wind_p0_mw = wind_p_ref_0 * vsc_s_n[wind_idx]
        wind_share = wind_p0_mw / np.sum(wind_p0_mw)
        net_load_mw = wind_variation_profile(T_END)
        nominal_disturbance_mw = WIND_VARIATION_MW
        event_text = (
            f"Net-load variation +{WIND_VARIATION_MW:.0f} MW over {WIND_RAMP_S:.0f} s "
            f"(wind power, {len(wind_idx)} units)"
        )
        print(f"Wind power before the variation: {np.sum(wind_p0_mw):.0f} MW")
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
        # Detailed plot: all loads and all VSC units (wind and HVDC), MW.
        load_values.append(np.asarray(load.p(x, v), dtype=float) * system_base_mva)
        vsc_values.append(np.asarray(vsc.p_e(x, v), dtype=float) * vsc_s_n)

    time_values, speed_values, governor_values, hvdc_values, agc_values = [], [], [], [], []
    load_values, vsc_values = [], []
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
            elif DISTURBANCE == "hvdc_reduction":
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

        if DISTURBANCE == "wind_variation":
            # Wind power follows the profile: less wind = more net load.
            delta_mw = net_load_mw(solver.t) * wind_share
            vsc.set_input("p_ref", wind_p_ref_0 - delta_mw / vsc_s_n[wind_idx], idx=wind_idx)

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

    # Detailed plot: regional frequencies, RoCoF, governor types, loads, VSC.
    regional_frequency = {
        region: nominal_frequency * (1.0 + np.average(speed[:, idx], axis=1, weights=inertia_weights[idx]))
        for region, idx in region_indices.items()
    }
    samples_100ms = max(1, int(round(0.1 / MAX_STEP)))
    rocof = np.convolve(np.gradient(coi_frequency, t), np.ones(samples_100ms) / samples_100ms, mode="same")

    governor_response_mw = (governor_p_m - governor_p_m[0]) * mechanical_base_mw
    hygov_response = np.sum(governor_response_mw[:, hygov_indices], axis=1)
    tgov1_response = np.sum(governor_response_mw[:, tgov1_indices], axis=1)

    load_power_mw = np.asarray(load_values)
    total_load_change = np.sum(load_power_mw - load_power_mw[0], axis=1)
    actual_load_increase = (
        load_power_mw[:, load_idx] - load_power_mw[0, load_idx]
        if DISTURBANCE == "load_step" else np.zeros_like(t)
    )
    vsc_power_mw = np.asarray(vsc_values)
    aggregate_vsc_change = np.sum(vsc_power_mw - vsc_power_mw[0], axis=1)

    # Disturbance as a function of time (MW of extra net load), for the plots.
    if DISTURBANCE == "wind_variation":
        disturbance_mw = net_load_mw(t)
    else:
        disturbance_mw = nominal_disturbance_mw * (t >= EVENT_TIME)

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
    if DISTURBANCE == "wind_variation":
        k_end = min(int(np.searchsorted(t, EVENT_TIME + WIND_RAMP_S)), len(t) - 1)
        print(f"COI frequency at end of ramp: {coi_frequency[k_end]:.4f} Hz")
        print(f"Std of COI frequency after start: {1e3 * np.std(coi_frequency[post_event]):.1f} mHz")
        print(f"Time outside +-10 mHz: {np.sum(outside_band) * MAX_STEP:.0f} s")
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
        "disturbance_mw": disturbance_mw,
        # Detailed plot
        "regional_frequency": regional_frequency,
        "rocof": rocof,
        "hygov_response": hygov_response,
        "tgov1_response": tgov1_response,
        "actual_load_increase": actual_load_increase,
        "total_load_change": total_load_change,
        "aggregate_vsc_change": aggregate_vsc_change,
        "hvdc_change": hvdc_change,
        "hvdc_names": vsc_names[hvdc_idx],
    }


def plot_results(results):
    """Detailed 5-panel figure: frequencies, RoCoF, governors, loads/VSC, AGC/HVDC."""

    t = results["time"]
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(13, 15))
    fig.suptitle(
        "Nordic 45 (2025) – FCR and frequency-based AGC (aFRR)\n"
        f"{results['event_text']}, t = {EVENT_TIME:.1f} s, {AGC_TEXT}",
        y=0.985,
    )

    # System and regional frequencies
    axes[0].plot(t, results["coi_frequency"], "k", lw=2, label="System COI")
    for region, frequency in results["regional_frequency"].items():
        axes[0].plot(t, frequency, lw=1.2, label=region)
    axes[0].axhline(results["nominal_frequency"], color="gray", ls=":", label="Nominal")
    axes[0].plot(results["nadir_time"], results["nadir"], "ko", label=f"Nadir {results['nadir']:.3f} Hz")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[0].legend(ncol=3)

    # Rate of change of frequency
    axes[1].plot(t, results["rocof"], color="tab:purple", label="COI RoCoF (100 ms mean)")
    axes[1].axhline(0.0, color="gray", ls=":")
    axes[1].set_ylabel("RoCoF (Hz/s)")
    axes[1].legend()

    # Governor response (FCR + aFRR)
    axes[2].plot(t, results["hygov_response"], label="HYGOV (hydro)")
    axes[2].plot(t, results["tgov1_response"], label="TGOV1 (thermal)")
    axes[2].plot(t, results["total_governor_response"], "k", lw=2, label="Total governor (FCR + aFRR)")
    axes[2].plot(t, results["afrr_to_generators"], color="tab:green", ls="--", label="aFRR set-point (AGC)")
    axes[2].plot(t, results["disturbance_mw"], color="gray", ls=":", label="Disturbance")
    axes[2].set_ylabel("Power response (MW)")
    axes[2].legend(ncol=2)

    # Load and VSC power changes
    axes[3].plot(t, results["actual_load_increase"], label="Actual load increase")
    axes[3].plot(t, results["total_load_change"], label="Total load change (all loads)")
    axes[3].plot(t, results["aggregate_vsc_change"], label="Aggregate VSC response")
    axes[3].plot(t, results["disturbance_mw"], color="gray", ls=":", label="Disturbance")
    axes[3].axhline(0.0, color="gray", ls=":")
    axes[3].set_ylabel("Power change (MW)")
    axes[3].legend(ncol=2)

    # AGC command and HVDC power
    axes[4].plot(t, results["agc_command"], "k", lw=2, label="Total AGC command")
    axes[4].plot(t, results["afrr_to_generators"], color="tab:green", ls="--", label="AGC to generators")
    axes[4].plot(t, results["afrr_to_hvdc"], color="tab:red", ls="--", label="AGC to HVDC")
    for k, name in enumerate(results["hvdc_names"]):
        axes[4].plot(t, results["hvdc_change"][:, k], lw=1.2, label=f"HVDC {name} (actual)")
    axes[4].axhline(0.0, color="gray", ls=":")
    axes[4].set_ylabel("Power (MW)")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(ncol=4)

    for ax in axes:
        ax.axvline(EVENT_TIME, color="black", ls="--", lw=1)
        ax.set_xlim(0.0, T_END)
        ax.grid(True)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.065, top=0.91, hspace=0.30)
    fig.align_ylabels(axes)


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
        f"{results['event_text']}, {AGC_TEXT}"
    )

    # =========================================================================
    # POWER PER RESERVE TYPE
    # =========================================================================

    ax_p.plot(t, fcr, color="tab:blue", lw=2.5, label="FCR (governor droop)")
    if AGC_ENABLED:
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
    ax_p.plot(
        t, results["disturbance_mw"], color="gray", ls="--", lw=1,
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

    markers = ((30.0, "30 s"), (120.0, "2 min"), (900.0, "15 min"))
    if DISTURBANCE == "wind_variation":
        markers = ((WIND_RAMP_S, "End of ramp"),)
    for delay, label in markers:
        t_mark = EVENT_TIME + delay
        if t_mark < t[-1]:
            for ax in (ax_p, ax_f):
                ax.axvline(t_mark, color="tab:red", ls="--", lw=1)
            ax_p.text(t_mark, 1.02, label, transform=ax_p.get_xaxis_transform(),
                      ha="center", va="bottom", color="tab:red")

    phase_labels = [("FCR", EVENT_TIME, EVENT_TIME + 120.0, "tab:blue"),
                    ("aFRR", EVENT_TIME + 120.0, min(EVENT_TIME + 900.0, t[-1]), "tab:green")]
    if DISTURBANCE == "wind_variation" or not AGC_ENABLED:
        phase_labels = []
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


def case_metrics(results):
    """Key numbers of one case, used in the comparison table."""
    t = results["time"]
    f = results["coi_frequency"]
    f_n = results["nominal_frequency"]
    after = t >= EVENT_TIME
    final = t >= t[-1] - FINAL_AVERAGING_WINDOW_S
    fcr = results["total_governor_response"] - results["afrr_to_generators"]

    metrics = {"Minimum frequency (Hz)": f"{np.min(f[after]):.3f}"}
    if DISTURBANCE == "wind_variation":
        k_end = min(int(np.searchsorted(t, EVENT_TIME + WIND_RAMP_S)), len(t) - 1)
        metrics["Frequency at end of ramp (Hz)"] = f"{f[k_end]:.3f}"
    metrics["Mean frequency, final 5 s (Hz)"] = f"{np.mean(f[final]):.3f}"
    metrics["Std of frequency (mHz)"] = f"{1e3 * np.std(f[after]):.1f}"
    metrics["Time outside +-10 mHz (s)"] = f"{np.sum(after & (np.abs(f - f_n) > RESTORATION_BAND_HZ)) * MAX_STEP:.0f}"
    metrics["FCR, final 5 s (MW)"] = f"{np.mean(fcr[final]):.0f}"
    metrics["aFRR, final 5 s (MW)"] = f"{np.mean(results['agc_command'][final]):.0f}"
    metrics["aFRR via HVDC, final 5 s (MW)"] = f"{np.mean(results['afrr_to_hvdc'][final]):.0f}"
    return metrics


def plot_case_comparison(results_0, results_1, label_1):
    """Case 0 (FCR only) and case 1 (FCR + aFRR) in one figure.

    Top:    frequency of both cases.
    Bottom: FCR of both cases, aFRR of case 1 and the disturbance.
    """
    t = results_0["time"]
    f_n = results_0["nominal_frequency"]
    fcr_0 = results_0["total_governor_response"] - results_0["afrr_to_generators"]
    fcr_1 = results_1["total_governor_response"] - results_1["afrr_to_generators"]

    fig, (ax_f, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={"hspace": 0.12})
    fig.suptitle(f"Case 0 and case 1 – Nordic 45 (2025)\n{results_0['event_text']}")

    # Frequency
    ax_f.plot(t, results_0["coi_frequency"], color="tab:red", lw=1.8, label="Case 0: FCR only")
    ax_f.plot(t, results_1["coi_frequency"], color="tab:blue", lw=1.8, label=f"Case 1: FCR + aFRR ({label_1})")
    ax_f.axhline(f_n, color="black", lw=1)
    ax_f.axhline(f_n - 0.1, color="gray", ls="--", lw=1)
    ax_f.text(t[-1], f_n - 0.1, " 49.9 Hz", va="center", ha="left", color="gray")
    ax_f.set_ylabel("Frequency, COI (Hz)")
    ax_f.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_f.legend(loc="lower left")

    # Power: who covers the disturbance
    ax_p.plot(t, results_0["disturbance_mw"], color="gray", ls="--", lw=1, label="Disturbance")
    ax_p.plot(t, fcr_0, color="tab:red", lw=1.8, label="FCR, case 0")
    ax_p.plot(t, fcr_1, color="tab:blue", lw=1.8, label="FCR, case 1")
    ax_p.plot(t, results_1["agc_command"], color="tab:green", lw=2.2, label="aFRR, case 1")
    if HVDC_SHARE > 0:
        ax_p.plot(t, results_1["afrr_to_hvdc"], color="tab:purple", lw=1.5, ls="--",
                  label="  of which via HVDC")
    ax_p.axhline(0.0, color="gray", lw=0.8)
    ax_p.set_ylabel("Power (MW)")
    ax_p.set_xlabel("Time (s)")
    ax_p.legend(loc="upper left", ncol=2)

    for ax in (ax_f, ax_p):
        ax.axvline(EVENT_TIME, color="black", ls=":", lw=1)
        if DISTURBANCE == "wind_variation":
            ax.axvline(EVENT_TIME + WIND_RAMP_S, color="tab:purple", ls="--", lw=1)
        ax.set_xlim(0.0, t[-1])
        ax.grid(True, alpha=0.4)
    if DISTURBANCE == "wind_variation":
        ax_f.text(EVENT_TIME + WIND_RAMP_S, 1.02, "End of ramp", transform=ax_f.get_xaxis_transform(),
                  ha="center", va="bottom", color="tab:purple")
    fig.align_ylabels((ax_f, ax_p))
    fig.subplots_adjust(left=0.08, right=0.93, top=0.88, bottom=0.08)


# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    if RUN_CASE_COMPARISON:
        label_1 = f"T_i = {T_I:.0f} s, HVDC share = {HVDC_SHARE:.0%}"
        print("\n=== Case 0: FCR only ===")
        AGC_ENABLED = False
        results_0 = run_simulation()
        print("\n=== Case 1: FCR + aFRR ===")
        AGC_ENABLED = True
        results_1 = run_simulation()

        m0, m1 = case_metrics(results_0), case_metrics(results_1)
        print("\nCase comparison")
        print("---------------")
        print(f"{'':34s}{'Case 0':>10s}{'Case 1':>10s}")
        for key in m0:
            print(f"{key:34s}{m0[key]:>10s}{m1[key]:>10s}")

        plot_case_comparison(results_0, results_1, label_1)
    else:
        results = run_simulation()
        if SHOW_RESERVE_OVERVIEW:
            plot_reserve_overview(results)
        if SHOW_DETAILED_PLOT:
            plot_results(results)
    plt.show()
