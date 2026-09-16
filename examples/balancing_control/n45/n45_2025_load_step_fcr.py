"""N45 load step with conventional FCR, without AGC or HVDC balancing."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import n45_2025 as model_data


# =============================================================================
# STUDY SETTINGS
# =============================================================================

EVENT_TIME = 1.0
LOAD_NAME = "L5240-1"
LOAD_STEP_MW = 100.0
LOAD_STEP_MVAR = 0.0
T_END = 60.0
MAX_STEP = 5e-3
ROCOF_WINDOW_S = 0.5
FINAL_AVERAGING_WINDOW_S = 5.0

REGION_AREA_CODES = {
    "Norway": {11, 12, 13, 14, 15},
    "Sweden": {21, 22, 23, 24},
    "Finland": {31},
}


def run_simulation():

    # =========================================================================
    # MODEL LOADING AND INITIALIZATION
    # =========================================================================

    model = model_data.load()
    system_base_mva = float(model["base_mva"])
    nominal_frequency = float(model["f"])

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    

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
    print(
        f"Applying nominal +{LOAD_STEP_MW:.1f} MW at {LOAD_NAME}, "
        f"bus {load_bus_name}, t={EVENT_TIME:.3f} s."
    )

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

    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    event_applied = False
    next_progress = 10

    while solver.t < T_END:
        # Apply the admittance step once, at the selected event time.
        if solver.t >= EVENT_TIME and not event_applied:
            load.y_load[load_idx] += delta_y
            ps.y_bus_red_mod[load_bus_idx, load_bus_idx] += delta_y
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

    aggregate_vsc_change = np.zeros_like(t)
    if vsc is not None:
        vsc_power_mw = np.asarray(vsc_power_values)
        aggregate_vsc_change = np.sum(
            vsc_power_mw - initial_vsc_power_mw[np.newaxis, :], axis=1
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
        "aggregate_vsc_change": aggregate_vsc_change,
        "nadir": float(coi_frequency[nadir_idx]),
        "nadir_time": float(t[nadir_idx]),
    }


def plot_results(results):

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================

    t = results["time"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 12))
    fig.suptitle(
        "Nordic 45 (2025) – conventional FCR reference\n"
        f"Nominal +{LOAD_STEP_MW:.0f} MW load step at {LOAD_NAME}, "
        f"t = {EVENT_TIME:.1f} s, no AGC or HVDC balancing",
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
    axes[2].plot(t, results["total_fcr_response"], "k", lw=2, label="Total governor/FCR")
    axes[2].axhline(LOAD_STEP_MW, color="gray", ls=":", label="Nominal disturbance")
    axes[2].set_ylabel("Power response (MW)")
    axes[2].legend(ncol=2)

    # =========================================================================
    # LOAD AND VSC POWER CHANGES
    # =========================================================================

    axes[3].plot(t, results["actual_load_increase"], label="Actual load increase")
    axes[3].plot(t, results["aggregate_vsc_change"], label="Aggregate VSC response")
    axes[3].axhline(LOAD_STEP_MW, color="gray", ls=":", label="Nominal load step")
    axes[3].axhline(0.0, color="gray", ls=":")
    axes[3].set_ylabel("Power change (MW)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=3)

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


# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    plot_results(run_simulation())
