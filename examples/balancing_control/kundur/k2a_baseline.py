"""Undisturbed baseline verification of the Kundur two-area HYGOV model."""

import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import k2a as model_data


T_END = 60.0
MAX_STEP = 5e-3
INITIAL_DERIVATIVE_TOL = 1e-5


def require_model(ps, category, model_name):
    """Return a loaded model or raise a clear verification error."""
    if not hasattr(ps, category):
        raise RuntimeError(f"TOPS did not create model category ps.{category}.")

    models = getattr(ps, category)
    if model_name not in models:
        raise RuntimeError(
            f"TOPS did not load {category}:{model_name}. "
            f"Loaded models: {list(models.keys())}"
        )
    return models[model_name]


def run_baseline():
    print("Imported model-data file:", model_data.__file__)
    print("No disturbances or model parameters are applied by this script.")

    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = require_model(ps, "gen", "GEN")
    hygov = require_model(ps, "gov", "HYGOV")
    sexs = require_model(ps, "avr", "SEXS")
    stab1 = require_model(ps, "pss", "STAB1")

    if not ps.power_flow_ready:
        raise RuntimeError("Kundur power flow did not converge.")
    if len(ps.buses) != 11:
        raise RuntimeError(f"Expected 11 buses, found {len(ps.buses)}.")
    if gen.n_units != 4:
        raise RuntimeError(f"Expected 4 generators, found {gen.n_units}.")
    if hygov.n_units != 4:
        raise RuntimeError(f"Expected 4 HYGOV units, found {hygov.n_units}.")
    if sexs.n_units != 4 or stab1.n_units != 4:
        raise RuntimeError("Expected 4 SEXS units and 4 STAB1 units.")

    v_initial = ps.solve_algebraic(0.0, ps.x0)
    dx_initial = ps.state_derivatives(0.0, ps.x0, v_initial)
    max_initial_derivative = float(np.max(np.abs(dx_initial)))

    print("\nKundur two-area HYGOV baseline")
    print("--------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Buses:", len(ps.buses))
    print("Generators:", gen.n_units)
    print("Governor models:", list(ps.gov.keys()))
    print("AVR models:", list(ps.avr.keys()))
    print("PSS models:", list(ps.pss.keys()))
    print("VSC models:", list(ps.vsc.keys()) if hasattr(ps, "vsc") else [])
    print("Dynamic states:", ps.n_states)
    print("Maximum initial derivative:", f"{max_initial_derivative:.6e}")

    if not np.isfinite(max_initial_derivative):
        raise RuntimeError("Initial derivatives contain NaN or infinity.")
    if max_initial_derivative > INITIAL_DERIVATIVE_TOL:
        print(
            "WARNING: Initial state is not sufficiently close to equilibrium "
            f"(limit {INITIAL_DERIVATIVE_TOL:.1e})."
        )

    generator_names = np.asarray(gen.par["name"], dtype=str)
    inertia = np.asarray(gen.par["H"], dtype=float)
    rating = np.asarray(gen.par["S_n"], dtype=float)
    if "N_par" in gen.par.dtype.names:
        rating = rating * np.asarray(gen.par["N_par"], dtype=float)

    inertia_weights = inertia * rating
    if np.sum(inertia_weights) <= 0.0:
        raise RuntimeError("Total generator inertia weight is not positive.")

    time_values = [0.0]
    speed_values = [np.asarray(gen.speed(ps.x0, v_initial)).copy()]

    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0,
        T_END,
        max_step=MAX_STEP,
    )

    start_time = time.perf_counter()
    next_progress = 10

    while solver.t < T_END:
        solver.step()
        time_values.append(float(solver.t))
        speed_values.append(np.asarray(gen.speed(solver.y, solver.v)).copy())

        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    runtime = time.perf_counter() - start_time
    sim_time = np.asarray(time_values)
    speed = np.asarray(speed_values)

    nominal_frequency = float(model["f"])
    generator_frequency = nominal_frequency * (1.0 + speed)
    coi_speed = np.average(speed, axis=1, weights=inertia_weights)
    coi_frequency = nominal_frequency * (1.0 + coi_speed)
    frequency_spread_microhz = (
        np.max(generator_frequency, axis=1)
        - np.min(generator_frequency, axis=1)
    ) * 1e6

    max_coi_deviation = float(
        np.max(np.abs(coi_frequency - nominal_frequency))
    )
    max_frequency_spread = float(np.max(frequency_spread_microhz))

    print("\nBaseline result")
    print("---------------")
    print("Simulation runtime:", f"{runtime:.2f} s")
    print("Maximum COI-frequency deviation:", f"{max_coi_deviation:.6e} Hz")
    print(
        "Maximum generator-frequency spread:",
        f"{max_frequency_spread:.6e} microHz",
    )
    print("Final COI frequency:", f"{coi_frequency[-1]:.12f} Hz")

    return {
        "time": sim_time,
        "generator_names": generator_names,
        "generator_frequency": generator_frequency,
        "coi_frequency": coi_frequency,
        "frequency_spread_microhz": frequency_spread_microhz,
        "nominal_frequency": nominal_frequency,
    }


def plot_results(results):
    time_axis = results["time"]
    nominal_frequency = results["nominal_frequency"]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 8))
    fig.suptitle(
        "Kundur two-area system – undisturbed baseline\n"
        "Synchronous generators with HYGOV, SEXS and STAB1"
    )

    axes[0].plot(
        time_axis,
        results["coi_frequency"],
        color="black",
        linewidth=1.8,
        label="COI frequency",
    )
    axes[0].axhline(
        nominal_frequency,
        color="gray",
        linestyle=":",
        label="Nominal frequency",
    )
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_ylim(nominal_frequency - 0.05, nominal_frequency + 0.05)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        time_axis,
        results["frequency_spread_microhz"],
        color="tab:blue",
        linewidth=1.5,
        label="Generator-frequency spread (max − min)",
    )
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_ylabel("Frequency spread ($\\mu$Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    baseline_results = run_baseline()
    plot_results(baseline_results)
