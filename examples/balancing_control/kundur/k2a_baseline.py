"""Undisturbed baseline simulation of the original TOPS Kundur model."""

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import k2a as model_data


# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

T_END = 60.0
MAX_STEP = 5e-3


def run_simulation():

    # =========================================================================
    # MODEL LOADING AND INITIALIZATION
    # =========================================================================

    # Load the Kundur data, solve the power flow and initialize dynamic states.
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]
    nominal_frequency = float(model["f"])


    # =========================================================================
    # INITIALIZED MODEL OVERVIEW
    # =========================================================================

    # Count the dynamic units that TOPS actually created during initialization.
    generator_count = sum(
        generator_model.n_units
        for generator_model in ps.gen.values()
    )

    hygov_count = (
        ps.gov["HYGOV"].n_units
        if hasattr(ps, "gov") and "HYGOV" in ps.gov
        else 0
    )

    tgov1_count = (
        ps.gov["TGOV1"].n_units
        if hasattr(ps, "gov") and "TGOV1" in ps.gov
        else 0
    )

    wind_vsc_count = 0
    hvdc_vsc_count = 0

    if hasattr(ps, "vsc"):
        for vsc_model in ps.vsc.values():
            names = np.asarray(
                vsc_model.par["name"],
                dtype=str,
            )

            # Wind-power VSC names in N45 begin with "WG".
            is_wind_vsc = np.char.startswith(names, "WG")

            wind_vsc_count += int(
                np.sum(is_wind_vsc)
            )

            # The remaining N45 VSC units represent HVDC connections.
            hvdc_vsc_count += int(
                np.sum(~is_wind_vsc)
            )

    print("\nInitialized model overview")
    print("--------------------------")
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

    # In an undisturbed baseline, derivatives should be close to zero because
    # the initialized model should start near dynamic equilibrium.
    v_initial = ps.solve_algebraic(
        0.0,
        ps.x0,
    )

    dx_initial = ps.state_derivatives(
        0.0,
        ps.x0,
        v_initial,
    )

    print(
        "Maximum initial derivative:",
        f"{np.max(np.abs(dx_initial)):.6e}",
    )


    # =========================================================================
    # CENTER-OF-INERTIA WEIGHTS
    # =========================================================================

    # COI weights: inertia multiplied by generator rating. N_par is included
    # only if the model contains several identical units in parallel.
    rating = np.asarray(
        gen.par["S_n"],
        dtype=float,
    ).copy()

    if "N_par" in gen.par.dtype.names:
        rating *= np.asarray(
            gen.par["N_par"],
            dtype=float,
        )

    inertia_weights = (
        np.asarray(gen.par["H"], dtype=float)
        * rating
    )


    # =========================================================================
    # NUMERICAL SOLVER
    # =========================================================================

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


    # =========================================================================
    # RESULT STORAGE
    # =========================================================================

    # Store the initialized operating point at t = 0 before stepping forward.
    time_values = [0.0]

    speed_values = [
        np.asarray(
            gen.speed(ps.x0, v_initial),
            dtype=float,
        ).copy()
    ]


    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    next_progress = 10

    while solver.t < T_END:
        solver.step()

        time_values.append(
            float(solver.t)
        )

        speed_values.append(
            np.asarray(
                gen.speed(solver.y, solver.v),
                dtype=float,
            ).copy()
        )

        # Terminal progress indicator; it does not affect the simulation.
        progress = int(
            100 * solver.t / T_END
        )

        if progress >= next_progress:
            print(
                f"Simulation progress: "
                f"{min(progress, 100)}%"
            )
            next_progress += 10


    # =========================================================================
    # RESULT PROCESSING
    # =========================================================================

    time_array = np.asarray(
        time_values
    )

    speed = np.asarray(
        speed_values
    )

    # Convert per-unit speed deviations to generator frequencies.
    generator_frequency = (
        nominal_frequency
        * (1.0 + speed)
    )

    # Calculate the inertia-weighted center-of-inertia frequency.
    coi_speed = np.average(
        speed,
        axis=1,
        weights=inertia_weights,
    )

    coi_frequency = (
        nominal_frequency
        * (1.0 + coi_speed)
    )

    # Maximum minus minimum generator frequency reveals relative motion
    # between the four machines that is not visible in the COI frequency.
    frequency_spread_microhz = (
        np.max(generator_frequency, axis=1)
        - np.min(generator_frequency, axis=1)
    ) * 1e6


    # =========================================================================
    # TERMINAL SUMMARY
    # =========================================================================

    print("\nBaseline result")
    print("---------------")

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


    # =========================================================================
    # RETURN RESULTS
    # =========================================================================

    return {
        "time": time_array,
        "coi_frequency": coi_frequency,
        "frequency_spread": frequency_spread_microhz,
        "nominal_frequency": nominal_frequency,
    }


def plot_results(results):

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================

    t = results["time"]
    nominal_frequency = results["nominal_frequency"]

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(13, 8),
    )

    fig.suptitle(
        "Kundur two-area system – undisturbed baseline\n"
        "Original TOPS model with TGOV1, SEXS and STAB1"
    )


    # =========================================================================
    # COI FREQUENCY
    # =========================================================================

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

    axes[0].set_ylim(
        nominal_frequency - 0.05,
        nominal_frequency + 0.05,
    )

    axes[0].set_ylabel(
        "Frequency (Hz)"
    )

    axes[0].ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )

    axes[0].legend()
    axes[0].grid(True)


    # =========================================================================
    # GENERATOR-FREQUENCY SPREAD
    # =========================================================================

    axes[1].plot(
        t,
        results["frequency_spread"],
        color="tab:blue",
        linewidth=1.5,
        label="Generator-frequency spread (max − min)",
    )

    axes[1].axhline(
        0.0,
        color="gray",
        linestyle=":",
    )

    axes[1].set_ylabel(
        "Frequency spread ($\\mu$Hz)"
    )

    axes[1].set_xlabel(
        "Time (s)"
    )

    axes[1].legend()
    axes[1].grid(True)


    # =========================================================================
    # FINAL FIGURE LAYOUT
    # =========================================================================

    fig.tight_layout()
    plt.show()


# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    results = run_simulation()
    plot_results(results)
