"""Undisturbed baseline simulation of the original TOPS Kundur model."""

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import k2a as model_data


T_END = 60.0
MAX_STEP = 5e-3


def run_simulation():
    # Load the Kundur data, solve the power flow and initialize dynamic states.
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]
    nominal_frequency = float(model["f"])

    # In an undisturbed baseline, derivatives should be close to zero because
    # the initialized model should start near dynamic equilibrium.
    v_initial = ps.solve_algebraic(0.0, ps.x0)
    dx_initial = ps.state_derivatives(0.0, ps.x0, v_initial)
    print(
        "Maximum initial derivative:",
        f"{np.max(np.abs(dx_initial)):.6e}",
    )

    # COI weights: inertia multiplied by generator rating. N_par is included
    # only if the model contains several identical units in parallel.
    rating = np.asarray(gen.par["S_n"], dtype=float).copy()
    if "N_par" in gen.par.dtype.names:
        rating *= np.asarray(gen.par["N_par"], dtype=float)

    inertia_weights = np.asarray(gen.par["H"], dtype=float) * rating

    # Modified Euler integrates the dynamic states, while TOPS solves the
    # algebraic network equations after each time step.
    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0,
        T_END,
        max_step=MAX_STEP,
    )

    # Store the initialized operating point at t = 0 before stepping forward.
    time_values = [0.0]
    speed_values = [
        np.asarray(gen.speed(ps.x0, v_initial), dtype=float).copy()
    ]

    next_progress = 10

    while solver.t < T_END:
        solver.step()

        time_values.append(float(solver.t))
        speed_values.append(
            np.asarray(gen.speed(solver.y, solver.v), dtype=float).copy()
        )

        # Terminal progress indicator; it does not affect the simulation.
        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    time_array = np.asarray(time_values)
    speed = np.asarray(speed_values)

    # Convert per-unit speed deviations to generator and COI frequencies.
    generator_frequency = nominal_frequency * (1.0 + speed)
    coi_speed = np.average(speed, axis=1, weights=inertia_weights)
    coi_frequency = nominal_frequency * (1.0 + coi_speed)

    # Maximum minus minimum generator frequency reveals relative motion
    # between the four machines that is not visible in the COI frequency.
    frequency_spread_microhz = (
        np.max(generator_frequency, axis=1)
        - np.min(generator_frequency, axis=1)
    ) * 1e6

    print(
        "Maximum COI-frequency deviation:",
        f"{np.max(np.abs(coi_frequency - nominal_frequency)):.6e} Hz",
    )
    print(
        "Maximum generator-frequency spread:",
        f"{np.max(frequency_spread_microhz):.6e} microHz",
    )
    print(
        "Final COI frequency:",
        f"{coi_frequency[-1]:.12f} Hz",
    )

    return {
        "time": time_array,
        "coi_frequency": coi_frequency,
        "frequency_spread": frequency_spread_microhz,
        "nominal_frequency": nominal_frequency,
    }


def plot_results(results):
    t = results["time"]
    nominal_frequency = results["nominal_frequency"]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 8))
    fig.suptitle(
        "Kundur two-area system – undisturbed baseline\n"
        "Original TOPS model with TGOV1, SEXS and STAB1"
    )

    axes[0].plot(
        t,
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
    axes[0].set_ylim(nominal_frequency - 0.05, nominal_frequency + 0.05)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        t,
        results["frequency_spread"],
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
    results = run_simulation()
    plot_results(results)
