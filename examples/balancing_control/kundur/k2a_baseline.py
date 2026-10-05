"""Undisturbed baseline simulation of the original TOPS Kundur model."""

import time

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

# Tie lines between area 1 (B1-B2, B5-B7) and area 2 (B3-B4, B8-B11).
# Positive flow = transfer from area 1 to area 2 (measured at bus B7).
TIE_LINE_NAMES = ["L7-8-1", "L7-8-2"]


def run_simulation():

    # =========================================================================
    # MODEL LOADING AND INITIALIZATION
    # =========================================================================

    # Load the Kundur data, solve the power flow and initialize dynamic states.
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]
    line = ps.lines["Line"]
    vsc = ps.vsc.get("VSC_SI") if hasattr(ps, "vsc") else None
    nominal_frequency = float(model["f"])
    system_base_mva = float(model["base_mva"])


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
    hvdc_idx = np.zeros(0, dtype=int)
    hvdc_names = np.zeros(0, dtype=str)
    hvdc_s_n = np.zeros(0)

    if vsc is not None:
        vsc_names = np.asarray(
            vsc.par["name"],
            dtype=str,
        )

        # Wind-power VSC names begin with "WG"; the remaining VSC units are
        # treated as HVDC connections (same convention as in N45).
        is_wind_vsc = np.char.startswith(vsc_names, "WG")

        wind_vsc_count = int(
            np.sum(is_wind_vsc)
        )

        hvdc_idx = np.where(~is_wind_vsc)[0]
        hvdc_vsc_count = len(hvdc_idx)
        hvdc_names = vsc_names[hvdc_idx]
        hvdc_s_n = np.asarray(
            vsc.par["S_n"],
            dtype=float,
        )[hvdc_idx]

    print("\nInitialized K2A model overview")
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
    # TIE-LINE INDICES
    # =========================================================================

    # Locate the lines between the two areas.
    line_names = np.asarray(
        line.par["name"],
        dtype=str,
    )

    tie_line_idx = np.asarray(
        [int(np.where(line_names == name)[0][0]) for name in TIE_LINE_NAMES],
        dtype=int,
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

    def total_gen_power_mw(x, v):
        # Sum of the electrical power from all synchronous generators.
        return float(
            np.sum(gen.P_e(x, v))
        )

    def tie_line_power_mw(x, v):
        # Total transfer from area 1 to area 2, measured at the B7 end.
        return float(
            np.sum(line.p_from(x, v)[tie_line_idx])
            * system_base_mva
        )

    def hvdc_power_mw(x, v):
        # p_e is in p.u. of each VSC's own rating. Positive = injection into
        # the AC grid. Empty array if the model has no HVDC VSC units.
        if vsc is None or hvdc_vsc_count == 0:
            return np.zeros(0)

        p_e = np.asarray(
            vsc.p_e(x, v),
            dtype=float,
        )

        return p_e[hvdc_idx] * hvdc_s_n

    # Store the initialized operating point at t = 0 before stepping forward.
    time_values = [0.0]

    speed_values = [
        np.asarray(
            gen.speed(ps.x0, v_initial),
            dtype=float,
        ).copy()
    ]

    gen_power_values = [
        total_gen_power_mw(ps.x0, v_initial)
    ]

    tie_line_values = [
        tie_line_power_mw(ps.x0, v_initial)
    ]

    hvdc_values = [
        hvdc_power_mw(ps.x0, v_initial).copy()
    ]


    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    next_progress = 10
    t_wall_start = time.perf_counter()

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

        gen_power_values.append(
            total_gen_power_mw(solver.y, solver.v)
        )

        tie_line_values.append(
            tie_line_power_mw(solver.y, solver.v)
        )

        hvdc_values.append(
            hvdc_power_mw(solver.y, solver.v).copy()
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

    wall_time = time.perf_counter() - t_wall_start
    n_steps = len(time_values) - 1


    # =========================================================================
    # RESULT PROCESSING
    # =========================================================================

    time_array = np.asarray(
        time_values
    )

    speed = np.asarray(
        speed_values
    )

    gen_power = np.asarray(
        gen_power_values
    )

    tie_line_power = np.asarray(
        tie_line_values
    )

    hvdc_power = np.asarray(
        hvdc_values
    ).reshape(len(time_values), hvdc_vsc_count)

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
    # between the machines that is not visible in the COI frequency.
    # Calculated from the speed deviations directly, so the result is not
    # limited by floating-point resolution around 50 Hz.
    frequency_spread_microhz = (
        nominal_frequency
        * (np.max(speed, axis=1) - np.min(speed, axis=1))
    ) * 1e6

    # Deviations from the initial operating point (should be ~0 in baseline).
    gen_power_deviation = gen_power - gen_power[0]
    tie_line_deviation = tie_line_power - tie_line_power[0]
    hvdc_deviation = hvdc_power - hvdc_power[0]


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

    print(
        "Maximum total generator-power deviation:",
        f"{np.max(np.abs(gen_power_deviation)):.6e} MW",
    )

    print("\nTie-line operating point (positive = area 1 to area 2)")
    print(
        f"  {' + '.join(TIE_LINE_NAMES)}   P0 = {tie_line_power[0]:9.2f} MW"
        f"   max |dP| = {np.max(np.abs(tie_line_deviation)):.3e} MW"
    )

    if hvdc_vsc_count > 0:
        print("\nHVDC operating point (positive = injection into AC grid)")
        for name, p0, dp in zip(
            hvdc_names,
            hvdc_power[0],
            np.max(np.abs(hvdc_deviation), axis=0),
        ):
            print(f"  {name:10s} P0 = {p0:9.2f} MW   max |dP| = {dp:.3e} MW")

    print("\nRuntime")
    print("-------")
    print(f"Simulated time: {T_END:.1f} s ({n_steps} steps of {MAX_STEP*1e3:.1f} ms)")
    print(f"Wall-clock time: {wall_time:.1f} s")
    print(f"Time per step: {wall_time / n_steps * 1e3:.2f} ms")
    print(f"Real-time factor: {wall_time / T_END:.2f} (wall time / simulated time)")


    # =========================================================================
    # RETURN RESULTS
    # =========================================================================

    return {
        "time": time_array,
        "coi_frequency": coi_frequency,
        "frequency_spread": frequency_spread_microhz,
        "gen_power_deviation": gen_power_deviation,
        "tie_line_deviation": tie_line_deviation,
        "hvdc_deviation": hvdc_deviation,
        "hvdc_names": hvdc_names,
        "nominal_frequency": nominal_frequency,
    }


def plot_results(results):

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================

    t = results["time"]
    nominal_frequency = results["nominal_frequency"]

    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(12, 10),
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
        label="Generator-frequency spread (max − min)",
    )

    axes[1].axhline(
        0.0,
        color="gray",
        linestyle=":",
    )

    axes[1].set_ylabel(
        "Frequency spread\n($\\mu$Hz)"
    )

    axes[1].legend()
    axes[1].grid(True)


    # =========================================================================
    # TOTAL GENERATOR POWER
    # =========================================================================

    axes[2].plot(
        t,
        results["gen_power_deviation"],
        color="tab:green",
        label="Total generator power − initial",
    )

    axes[2].axhline(
        0.0,
        color="gray",
        linestyle=":",
    )

    axes[2].set_ylabel(
        "$\\Delta P_{gen}$ (MW)"
    )

    axes[2].legend()
    axes[2].grid(True)


    # =========================================================================
    # TIE-LINE AND HVDC POWER
    # =========================================================================

    axes[3].plot(
        t,
        results["tie_line_deviation"],
        color="tab:red",
        label="Tie line area 1 → 2",
    )

    for k, name in enumerate(results["hvdc_names"]):
        axes[3].plot(
            t,
            results["hvdc_deviation"][:, k],
            label=name,
        )

    axes[3].axhline(
        0.0,
        color="gray",
        linestyle=":",
    )

    axes[3].set_ylabel(
        "$\\Delta P_{tie}$, $\\Delta P_{HVDC}$\n(MW)"
    )

    axes[3].set_xlabel(
        "Time (s)"
    )

    axes[3].legend()
    axes[3].grid(True)


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
