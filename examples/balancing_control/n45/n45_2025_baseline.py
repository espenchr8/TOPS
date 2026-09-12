import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.ps_models.n45_2025 as model_data


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------

T_END = 60.0
MAX_STEP = 5e-3
NOMINAL_FREQUENCY_HZ = 50.0

# Fixed frequency limits make baseline figures directly comparable.
FREQUENCY_MIN_HZ = 49.95
FREQUENCY_MAX_HZ = 50.05


def run_simulation():
    """Run the undisturbed Nordic 45 (2025) baseline."""

    print("Imported model-data file:", model_data.__file__)
    print("No disturbances or model parameters are applied by this script.")

    # Load and initialize the complete N45 model.
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen_model = ps.gen["GEN"]

    # -----------------------------------------------------------------------
    # Initial operating-point information
    # -----------------------------------------------------------------------

    dx_initial = ps.state_derivatives(
        0.0,
        ps.x_0,
        ps.v_0,
    )

    maximum_initial_derivative = float(
        np.max(np.abs(dx_initial))
    )

    print("\nN45 baseline")
    print("------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Buses:", len(ps.buses))
    print("Generators:", gen_model.n_units)

    if hasattr(ps, "gov"):
        print("Governor models:", list(ps.gov.keys()))

    if hasattr(ps, "avr"):
        print("AVR models:", list(ps.avr.keys()))

    if hasattr(ps, "pss"):
        print("PSS models:", list(ps.pss.keys()))

    if hasattr(ps, "vsc"):
        print("VSC models:", list(ps.vsc.keys()))

    print(
        "Maximum initial derivative:",
        f"{maximum_initial_derivative:.6e}",
    )

    # -----------------------------------------------------------------------
    # Center-of-inertia weights
    # -----------------------------------------------------------------------

    inertia_weights = (
        np.asarray(gen_model.par["H"], dtype=float)
        * np.asarray(gen_model.par["S_n"], dtype=float)
        * np.asarray(gen_model.par["N_par"], dtype=float)
    )

    # -----------------------------------------------------------------------
    # Create numerical solver
    # -----------------------------------------------------------------------

    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x_0.copy(),
        T_END,
        max_step=MAX_STEP,
    )

    # -----------------------------------------------------------------------
    # Result storage
    # -----------------------------------------------------------------------

    time_values = []
    generator_frequency_values = []
    coi_frequency_values = []

    def store_results(t, x, v):
        speed = np.asarray(
            gen_model.speed(x, v),
            dtype=float,
        )

        generator_frequency_hz = (
            NOMINAL_FREQUENCY_HZ
            * (1.0 + speed)
        )

        coi_speed = np.average(
            speed,
            weights=inertia_weights,
        )

        coi_frequency_hz = (
            NOMINAL_FREQUENCY_HZ
            * (1.0 + coi_speed)
        )

        time_values.append(float(t))
        generator_frequency_values.append(
            generator_frequency_hz.copy()
        )
        coi_frequency_values.append(
            float(coi_frequency_hz)
        )

    # Store the initialized state at t = 0.
    store_results(
        0.0,
        ps.x_0,
        ps.v_0,
    )

    # -----------------------------------------------------------------------
    # Run simulation
    # -----------------------------------------------------------------------

    next_progress = 10
    start_time = time.time()

    while solver.t < T_END:
        solver.step()

        store_results(
            solver.t,
            solver.y,
            solver.v,
        )

        progress = int(
            100 * solver.t / T_END
        )

        if progress >= next_progress:
            print(
                f"Simulation progress: "
                f"{min(progress, 100)}%"
            )
            next_progress += 10

    simulation_runtime = time.time() - start_time

    # -----------------------------------------------------------------------
    # Process results
    # -----------------------------------------------------------------------

    time_array = np.asarray(time_values)

    generator_frequency_hz = np.asarray(
        generator_frequency_values
    )

    coi_frequency_hz = np.asarray(
        coi_frequency_values
    )

    generator_frequency_spread_microhz = (
        np.max(generator_frequency_hz, axis=1)
        - np.min(generator_frequency_hz, axis=1)
    ) * 1e6

    maximum_coi_frequency_deviation_hz = float(
        np.max(
            np.abs(
                coi_frequency_hz
                - NOMINAL_FREQUENCY_HZ
            )
        )
    )

    maximum_generator_frequency_spread_microhz = float(
        np.max(generator_frequency_spread_microhz)
    )

    final_coi_frequency_hz = float(
        coi_frequency_hz[-1]
    )

    print("\nBaseline result")
    print("---------------")
    print(
        "Simulation runtime:",
        f"{simulation_runtime:.2f} s",
    )
    print(
        "Maximum COI-frequency deviation:",
        f"{maximum_coi_frequency_deviation_hz:.6e} Hz",
    )
    print(
        "Maximum generator-frequency spread:",
        f"{maximum_generator_frequency_spread_microhz:.6e} microHz",
    )
    print(
        "Final COI frequency:",
        f"{final_coi_frequency_hz:.12f} Hz",
    )

    return {
        "time": time_array,
        "coi_frequency_hz": coi_frequency_hz,
        "generator_frequency_spread_microhz": (
            generator_frequency_spread_microhz
        ),
    }


def plot_results(results):
    """Plot the undisturbed system-frequency behaviour."""

    time_array = results["time"]

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7),
    )

    fig.suptitle(
        "Nordic 45 (2025) – undisturbed baseline\n"
        "Original model parameters"
    )

    # -----------------------------------------------------------------------
    # COI frequency
    # -----------------------------------------------------------------------

    axes[0].plot(
        time_array,
        results["coi_frequency_hz"],
        color="black",
        linewidth=1.8,
        label="COI frequency",
    )

    axes[0].axhline(
        NOMINAL_FREQUENCY_HZ,
        color="gray",
        linestyle=":",
        linewidth=1.3,
        label="Nominal frequency",
    )

    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_ylim(
        FREQUENCY_MIN_HZ,
        FREQUENCY_MAX_HZ,
    )

    axes[0].ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )

    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # -----------------------------------------------------------------------
    # Generator-frequency spread
    # -----------------------------------------------------------------------

    axes[1].plot(
        time_array,
        results["generator_frequency_spread_microhz"],
        color="tab:blue",
        linewidth=1.5,
        label="Generator-frequency spread (max − min)",
    )

    axes[1].axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.2,
    )

    axes[1].set_ylabel(
        "Frequency spread\n($\\mu$Hz)"
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulation_results = run_simulation()
    plot_results(simulation_results)
