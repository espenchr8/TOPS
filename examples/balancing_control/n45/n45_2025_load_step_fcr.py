"""N45-2025: verified 100 MW load step with conventional FCR active.

The event increases the admittance of one existing constant-impedance load.
No AGC/aFRR, HVDC balancing command, or synthetic inertia is added here.
The imported n45_2025.py file is never modified by this script.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import n45_2025 as model_data


SCRIPT_VERSION = "verified-N45-load-step-FCR-v1-20260915"

# Study settings. L5240-1 is an existing Norwegian constant-impedance load.
EVENT_TIME = 1.0
LOAD_NAME = "L5240-1"
LOAD_STEP_MW = 100.0
LOAD_STEP_MVAR = 0.0
T_END = 60.0
MAX_STEP = 5e-3
ROCOF_WINDOW_S = 0.5
FINAL_AVERAGING_WINDOW_S = 5.0
INITIAL_DERIVATIVE_TOL = 1e-5

# These are the 12 STAB1 units previously selected and tuned to K=15.
TUNED_PSS_GENERATORS = {
    "G5230-1", "G5230-2", "G5560-1",
    "G7000-1", "G7000-2", "G7000-3", "G7000-4",
    "G7000-5", "G7000-6", "G7000-7",
    "G7100-1", "G7100-2",
}
EXPECTED_PSS_GAIN = 15.0

REGION_AREA_CODES = {
    "Norway": {11, 12, 13, 14, 15},
    "Sweden": {21, 22, 23, 24},
    "Finland": {31},
}


def require_model(ps, category, model_name):
    """Return a required TOPS model or raise a clear error."""
    if not hasattr(ps, category):
        raise RuntimeError(f"TOPS did not create ps.{category}.")
    models = getattr(ps, category)
    if model_name not in models:
        raise RuntimeError(
            f"TOPS did not load {category}:{model_name}. "
            f"Loaded models: {list(models.keys())}"
        )
    return models[model_name]


def field_or_default(par, field, default):
    """Read a structured-array field, or return a full default vector."""
    if field in par.dtype.names:
        return np.asarray(par[field], dtype=float)
    return np.full(len(par), float(default))


def indices_for_names(all_names, requested_names):
    """Return unique indices for requested unit names."""
    indices = []
    for name in requested_names:
        matches = np.where(all_names == name)[0]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one unit named {name}.")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def build_bus_area_map(model):
    """Map each N45 bus name to its Area code."""
    table = model["buses"]
    header = list(table[0])
    i_name = header.index("name")
    i_area = header.index("Area")
    return {str(row[i_name]): int(row[i_area]) for row in table[1:]}


def verify_tuned_pss(stab1):
    """Verify that this case uses the same tuned N45 reference model."""
    pss_generators = np.asarray(stab1.par["gen"], dtype=str)
    pss_gains = np.asarray(stab1.par["K"], dtype=float)
    indices = indices_for_names(pss_generators, sorted(TUNED_PSS_GENERATORS))
    incorrect = [
        f"{pss_generators[i]} (K={pss_gains[i]:g})"
        for i in indices
        if not np.isclose(pss_gains[i], EXPECTED_PSS_GAIN)
    ]
    if incorrect:
        raise RuntimeError(
            "The 12 selected STAB1 gains must be K=15 before this test: "
            + ", ".join(incorrect)
        )


def run_simulation():
    """Apply a load step and return frequency- and reserve-response results."""
    print("Script version:", SCRIPT_VERSION)
    print("Imported model-data file:", model_data.__file__)
    print("No installed TOPS source files or model-data files are modified.")
    print("No AGC/aFRR or supplementary HVDC control is active.")

    model = model_data.load()
    nominal_frequency = float(model["f"])
    bus_area = build_bus_area_map(model)

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = require_model(ps, "gen", "GEN")
    hygov = require_model(ps, "gov", "HYGOV")
    tgov1 = require_model(ps, "gov", "TGOV1")
    require_model(ps, "avr", "SEXS")
    stab1 = require_model(ps, "pss", "STAB1")
    load = require_model(ps, "loads", "Load")
    verify_tuned_pss(stab1)

    vsc = None
    if hasattr(ps, "vsc") and "VSC_SI" in ps.vsc:
        vsc = ps.vsc["VSC_SI"]

    if not ps.power_flow_ready:
        raise RuntimeError("N45 power flow did not converge.")
    if len(ps.buses) != 46 or gen.n_units != 47:
        raise RuntimeError(
            f"Unexpected N45 size: {len(ps.buses)} buses, "
            f"{gen.n_units} generators."
        )

    v_initial = ps.solve_algebraic(0.0, ps.x0)
    dx_initial = ps.state_derivatives(0.0, ps.x0, v_initial)
    max_initial_derivative = float(np.max(np.abs(dx_initial)))
    if not np.isfinite(max_initial_derivative):
        raise RuntimeError("Initial derivatives contain NaN or infinity.")
    if max_initial_derivative > INITIAL_DERIVATIVE_TOL:
        raise RuntimeError(
            "N45 is not sufficiently close to equilibrium before the event: "
            f"max |dx/dt|={max_initial_derivative:.3e}."
        )

    generator_names = np.asarray(gen.par["name"], dtype=str)
    generator_buses = np.asarray(gen.par["bus"], dtype=str)
    rating_mva = field_or_default(gen.par, "S_n", 1.0)
    rating_mva *= field_or_default(gen.par, "N_par", 1.0)
    mechanical_base_mw = rating_mva * field_or_default(gen.par, "PF_n", 1.0)
    inertia_weights = field_or_default(gen.par, "H", 0.0) * rating_mva
    if np.sum(inertia_weights) <= 0.0:
        raise RuntimeError("Total synchronous-generator inertia is not positive.")

    generator_area = np.asarray(
        [bus_area[bus] for bus in generator_buses], dtype=int
    )
    region_indices = {
        region: np.where(np.isin(generator_area, list(area_codes)))[0]
        for region, area_codes in REGION_AREA_CODES.items()
    }
    if any(len(idx) == 0 for idx in region_indices.values()):
        raise RuntimeError("At least one requested regional generator group is empty.")

    hygov_indices = indices_for_names(
        generator_names, np.asarray(hygov.par["gen"], dtype=str)
    )
    tgov1_indices = indices_for_names(
        generator_names, np.asarray(tgov1.par["gen"], dtype=str)
    )

    load_names = np.asarray(load.par["name"], dtype=str)
    load_idx = int(indices_for_names(load_names, [LOAD_NAME])[0])
    load_bus_name = str(load.par["bus"][load_idx])
    load_bus_idx = int(load.bus_idx_red["terminal"][load_idx])
    initial_load_voltage = complex(v_initial[load_bus_idx])
    if abs(initial_load_voltage) <= 0.0:
        raise RuntimeError("Initial voltage at the disturbed load bus is zero.")

    # The N45 loads are constant impedances. The requested P/Q step is defined
    # at the pre-disturbance voltage and converted to an admittance increment.
    delta_s_pu = (LOAD_STEP_MW + 1j * LOAD_STEP_MVAR) / float(model["base_mva"])
    delta_y = np.conj(delta_s_pu) / abs(initial_load_voltage) ** 2

    # Save the governor-generated mechanical-power signal before wrapping it.
    # The event affects only the load; this wrapper is only used for logging FCR.
    governor_p_m = gen.P_m
    initial_governor_p_m_pu = np.asarray(
        governor_p_m(ps.x0, v_initial), dtype=float
    ).copy()
    initial_load_power_mw = np.asarray(
        load.p(ps.x0, v_initial), dtype=float
    ) * float(model["base_mva"])

    initial_vsc_power_mw = None
    if vsc is not None:
        initial_vsc_power_mw = (
            np.asarray(vsc.p_e(ps.x0, v_initial), dtype=float)
            * field_or_default(vsc.par, "S_n", float(model["base_mva"]))
        )

    print("\nN45 load-step FCR verification")
    print("-------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Dynamic states:", ps.n_states)
    print("HYGOV units:", hygov.n_units)
    print("TGOV1 units:", tgov1.n_units)
    print("Maximum initial derivative:", f"{max_initial_derivative:.6e}")
    print("Disturbed load:", LOAD_NAME)
    print("Bus:", load_bus_name, "Area code:", bus_area[load_bus_name])
    print("Nominal active-power increase:", f"{LOAD_STEP_MW:.1f} MW")
    print("Event time:", f"{EVENT_TIME:.3f} s")

    time_values = [0.0]
    speed_values = [np.asarray(gen.speed(ps.x0, v_initial)).copy()]
    governor_values = [initial_governor_p_m_pu.copy()]
    load_power_values = [initial_load_power_mw.copy()]
    load_voltage_values = [abs(initial_load_voltage)]
    vsc_power_values = []
    if vsc is not None:
        vsc_power_values.append(initial_vsc_power_mw.copy())

    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0,
        T_END,
        max_step=MAX_STEP,
    )

    event_applied = False
    start_time = time.perf_counter()
    next_progress = 10

    while solver.t < T_END:
        if solver.t >= EVENT_TIME and not event_applied:
            load.y_load[load_idx] += delta_y
            ps.y_bus_red_mod[load_bus_idx, load_bus_idx] += delta_y
            solver.v[:] = ps.solve_algebraic(solver.t, solver.y)
            event_applied = True

            actual_step = (
                float(load.p(solver.y, solver.v)[load_idx])
                * float(model["base_mva"])
                - initial_load_power_mw[load_idx]
            )
            print(
                f"Applied nominal +{LOAD_STEP_MW:.1f} MW load step at "
                f"t={solver.t:.3f} s."
            )
            print(
                "Actual immediate load increase after voltage response:",
                f"{actual_step:.3f} MW",
            )

        solver.step()
        if not np.all(np.isfinite(solver.y)) or not np.all(np.isfinite(solver.v)):
            raise FloatingPointError(
                f"Simulation became non-finite at t={solver.t:.6f} s."
            )

        time_values.append(float(solver.t))
        speed_values.append(np.asarray(gen.speed(solver.y, solver.v)).copy())
        governor_values.append(
            np.asarray(governor_p_m(solver.y, solver.v), dtype=float).copy()
        )
        load_power_values.append(
            np.asarray(load.p(solver.y, solver.v), dtype=float)
            * float(model["base_mva"])
        )
        load_voltage_values.append(abs(solver.v[load_bus_idx]))
        if vsc is not None:
            vsc_power_values.append(
                np.asarray(vsc.p_e(solver.y, solver.v), dtype=float)
                * field_or_default(vsc.par, "S_n", float(model["base_mva"]))
            )

        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    runtime = time.perf_counter() - start_time
    if not event_applied:
        raise RuntimeError("The load step was not applied.")

    sim_time = np.asarray(time_values)
    speed = np.asarray(speed_values)
    governor_p_m_pu = np.asarray(governor_values)
    load_power_mw = np.asarray(load_power_values)
    load_voltage = np.asarray(load_voltage_values)

    generator_frequency = nominal_frequency * (1.0 + speed)
    coi_frequency = nominal_frequency * (
        1.0 + np.average(speed, axis=1, weights=inertia_weights)
    )
    regional_frequency = {
        region: nominal_frequency * (
            1.0
            + np.average(
                speed[:, idx], axis=1, weights=inertia_weights[idx]
            )
        )
        for region, idx in region_indices.items()
    }

    # Numerical derivative is shown after mild smoothing to avoid emphasizing
    # solver-scale noise. The 0.5 s average is the reported RoCoF metric.
    rocof_raw = np.gradient(coi_frequency, sim_time)
    samples_100ms = max(1, int(round(0.1 / MAX_STEP)))
    kernel = np.ones(samples_100ms) / samples_100ms
    rocof = np.convolve(rocof_raw, kernel, mode="same")

    governor_response_mw = (
        governor_p_m_pu - initial_governor_p_m_pu[np.newaxis, :]
    ) * mechanical_base_mw[np.newaxis, :]
    hygov_response = np.sum(governor_response_mw[:, hygov_indices], axis=1)
    tgov1_response = np.sum(governor_response_mw[:, tgov1_indices], axis=1)
    total_fcr_response = np.sum(governor_response_mw, axis=1)

    actual_load_increase = load_power_mw[:, load_idx] - initial_load_power_mw[load_idx]

    aggregate_vsc_change = np.zeros_like(sim_time)
    if vsc is not None:
        vsc_power_mw = np.asarray(vsc_power_values)
        aggregate_vsc_change = np.sum(
            vsc_power_mw - initial_vsc_power_mw[np.newaxis, :], axis=1
        )

    post_event = sim_time >= EVENT_TIME
    post_indices = np.where(post_event)[0]
    nadir_idx = int(post_indices[np.argmin(coi_frequency[post_event])])
    nadir = float(coi_frequency[nadir_idx])
    nadir_time = float(sim_time[nadir_idx])

    event_idx = int(np.searchsorted(sim_time, EVENT_TIME, side="left"))
    rocof_end_idx = int(
        np.searchsorted(sim_time, EVENT_TIME + ROCOF_WINDOW_S, side="left")
    )
    rocof_end_idx = min(rocof_end_idx, len(sim_time) - 1)
    average_rocof = float(
        (coi_frequency[rocof_end_idx] - coi_frequency[event_idx])
        / (sim_time[rocof_end_idx] - sim_time[event_idx])
    )

    final_window = sim_time >= max(EVENT_TIME, T_END - FINAL_AVERAGING_WINDOW_S)
    final_frequency = float(np.mean(coi_frequency[final_window]))
    final_fcr = float(np.mean(total_fcr_response[final_window]))
    max_regional_separation = float(
        np.max(
            np.ptp(
                np.column_stack(list(regional_frequency.values())), axis=1
            )[post_event]
        )
    )

    print("\nResponse summary")
    print("----------------")
    print("Simulation runtime:", f"{runtime:.2f} s")
    print("COI-frequency nadir:", f"{nadir:.5f} Hz")
    print("Time to nadir:", f"{nadir_time - EVENT_TIME:.3f} s")
    print("Average RoCoF over first 0.5 s:", f"{average_rocof:+.5f} Hz/s")
    print("Mean COI frequency in final 5 s:", f"{final_frequency:.5f} Hz")
    print("Mean total governor/FCR response in final 5 s:", f"{final_fcr:.2f} MW")
    print("Maximum regional-frequency separation:", f"{max_regional_separation:.6f} Hz")

    return {
        "time": sim_time,
        "nominal_frequency": nominal_frequency,
        "coi_frequency": coi_frequency,
        "regional_frequency": regional_frequency,
        "rocof": rocof,
        "hygov_response": hygov_response,
        "tgov1_response": tgov1_response,
        "total_fcr_response": total_fcr_response,
        "actual_load_increase": actual_load_increase,
        "load_voltage": load_voltage,
        "aggregate_vsc_change": aggregate_vsc_change,
        "nadir": nadir,
        "nadir_time": nadir_time,
    }


def plot_results(results):
    """Plot the quantities needed for the N45 FCR reference case."""
    t = results["time"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 12))
    fig.suptitle(
        "Nordic 45 (2025) – conventional FCR reference\n"
        f"Nominal +{LOAD_STEP_MW:.0f} MW load step at {LOAD_NAME}, "
        f"t = {EVENT_TIME:.1f} s; no AGC or HVDC balancing"
    )

    axes[0].plot(t, results["coi_frequency"], color="black", lw=2, label="System COI")
    for region, frequency in results["regional_frequency"].items():
        axes[0].plot(t, frequency, lw=1.2, label=region)
    axes[0].axhline(results["nominal_frequency"], color="gray", ls=":", label="Nominal")
    axes[0].plot(results["nadir_time"], results["nadir"], "ko", label=f"Nadir {results['nadir']:.3f} Hz")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend(ncol=3)
    axes[0].grid(True)

    axes[1].plot(t, results["rocof"], color="tab:purple", label="COI RoCoF (100 ms mean)")
    axes[1].axhline(0.0, color="gray", ls=":")
    axes[1].set_ylabel("RoCoF (Hz/s)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, results["hygov_response"], label="HYGOV (hydro)")
    axes[2].plot(t, results["tgov1_response"], label="TGOV1 (thermal)")
    axes[2].plot(t, results["total_fcr_response"], color="black", lw=2, label="Total governor/FCR")
    axes[2].axhline(LOAD_STEP_MW, color="gray", ls=":", label="Nominal disturbance")
    axes[2].set_ylabel("Power response (MW)")
    axes[2].legend(ncol=2)
    axes[2].grid(True)

    axes[3].plot(t, results["actual_load_increase"], label="Actual load increase")
    axes[3].plot(t, results["aggregate_vsc_change"], label="Aggregate VSC response")
    axes[3].axhline(LOAD_STEP_MW, color="gray", ls=":", label="Nominal load step")
    axes[3].axhline(0.0, color="gray", ls=":")
    axes[3].set_ylabel("Power change (MW)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=3)
    axes[3].grid(True)

    for ax in axes:
        ax.axvline(EVENT_TIME, color="black", ls="--", lw=1)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulation_results = run_simulation()
    plot_results(simulation_results)
