import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.ps_models.n45_2025 as model_data


T_END = 60.0
MAX_STEP = 5e-3


def run_simulation():
    # Load the N45 data, solve the power flow and initialize dynamic states.
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]
    nominal_frequency = float(model["f"])

    # In an undisturbed baseline, derivatives should be close to zero because
    # the initialized model should start near dynamic equilibrium.
    dx_initial = ps.state_derivatives(0.0, ps.x_0, ps.v_0)
    print(
        "Maximum initial derivative:",
        f"{np.max(np.abs(dx_initial)):.6e}",
    )

    # COI weights: inertia × generator rating × number of parallel units.
    # Larger machines with higher inertia contribute more to COI frequency.
    inertia_weights = (
        np.asarray(gen.par["H"], dtype=float)
        * np.asarray(gen.par["S_n"], dtype=float)
        * np.asarray(gen.par["N_par"], dtype=float)
    )

    # Modified Euler integrates the dynamic states, while TOPS solves the
    # algebraic network equations after each time step.
    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x_0.copy(),
        T_END,
        max_step=MAX_STEP,
    )

    # Store the initialized operating point at t = 0 before stepping forward.
    time_values = [0.0]
    speed_values = [
        np.asarray(gen.speed(ps.x_0, ps.v_0), dtype=float).copy()
    ]

    next_progress = 10

    while solver.t < T_END:
        solver.step()

        time_values.append(float(solver.t))
        speed_values.append(
            np.asarray(
                gen.speed(solver.y, solver.v),
                dtype=float,
            ).copy()
        )

        # Terminal progress indicator; it does not affect the simulation.
        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    time_array = np.asarray(time_values)
    speed = np.asarray(speed_values)

    # TOPS returns relative generator-speed deviations in per unit.
    generator_frequency = nominal_frequency * (1.0 + speed)

    # COI frequency is the inertia-weighted system frequency.
    coi_speed = np.average(
        speed,
        axis=1,
        weights=inertia_weights,
    )
    coi_frequency = nominal_frequency * (1.0 + coi_speed)

    # Maximum minus minimum generator frequency reveals small relative motion
    # that is not visible in the common COI-frequency curve.
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

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7),
    )

    fig.suptitle(
        "Nordic 45 (2025) – undisturbed baseline\n"
        "Initialized model without disturbance"
    )

    axes[0].plot(
        t,
        results["coi_frequency"],
        color="black",
        label="COI frequency",
    )
    axes[0].axhline(
        nominal_frequency,
        color="gray",
        linestyle=":",
        label="Nominal frequency",
    )
    axes[0].set_ylim(
        nominal_frequency - 0.05,
        nominal_frequency + 0.05,
    )
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        t,
        results["frequency_spread"],
        color="tab:blue",
        label="Generator-frequency spread (max − min)",
    )
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_ylabel("Frequency spread\n($\\mu$Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_simulation()
    plot_results(results)
