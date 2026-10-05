import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.ps_models.n45_2025 as model_data


# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

T_END = 60.0
MAX_STEP = 5e-3


def run_simulation():

    # =========================================================================
    # MODEL LOADING AND INITIALIZATION
    # =========================================================================

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]
    vsc = ps.vsc["VSC_SI"]
    nominal_frequency = float(model["f"])

    # Split VSC units into wind (names starting with "WG") and HVDC.
    vsc_names = np.asarray(vsc.par["name"], dtype=str)
    is_wind_vsc = np.char.startswith(vsc_names, "WG")
    hvdc_idx = np.where(~is_wind_vsc)[0]
    hvdc_names = vsc_names[hvdc_idx]
    hvdc_s_n = np.asarray(vsc.par["S_n"], dtype=float)[hvdc_idx]


    # =========================================================================
    # INITIALIZED MODEL OVERVIEW
    # =========================================================================

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

    print("\nInitialized N45 model overview")
    print("------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Buses:", len(ps.buses))
    print("Synchronous generators:", generator_count)
    print("HYGOV hydro units:", hygov_count)
    print("TGOV1 thermal units:", tgov1_count)
    print("Wind-power VSC units:", int(np.sum(is_wind_vsc)))
    print("HVDC VSC units:", len(hvdc_idx))
    print("Dynamic states:", ps.n_states)


    # =========================================================================
    # INITIAL EQUILIBRIUM CHECK
    # =========================================================================

    # In an undisturbed baseline, derivatives should be close to zero because
    # the initialized model should start near dynamic equilibrium.
    dx_initial = ps.state_derivatives(
        0.0,
        ps.x_0,
        ps.v_0,
    )

    print(
        "Maximum initial derivative:",
        f"{np.max(np.abs(dx_initial)):.6e}",
    )


    # =========================================================================
    # CENTER-OF-INERTIA WEIGHTS
    # =========================================================================

    # COI weights: inertia x generator rating x number of parallel units.
    inertia_weights = (
        np.asarray(gen.par["H"], dtype=float)
        * np.asarray(gen.par["S_n"], dtype=float)
        * np.asarray(gen.par["N_par"], dtype=float)
    )


    # =========================================================================
    # NUMERICAL SOLVER
    # =========================================================================

    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x_0.copy(),
        T_END,
        max_step=MAX_STEP,
    )


    # =========================================================================
    # RESULT STORAGE
    # =========================================================================

    def hvdc_power_mw(x, v):
        # p_e is in p.u. of each VSC's own rating. Positive = injection into
        # the Nordic grid (import).
        p_e = np.asarray(vsc.p_e(x, v), dtype=float)
        return p_e[hvdc_idx] * hvdc_s_n

    def total_gen_power_mw(x, v):
        return float(np.sum(gen.P_e(x, v)))

    time_values = [0.0]
    speed_values = [np.asarray(gen.speed(ps.x_0, ps.v_0), dtype=float).copy()]
    hvdc_values = [hvdc_power_mw(ps.x_0, ps.v_0).copy()]
    gen_power_values = [total_gen_power_mw(ps.x_0, ps.v_0)]


    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    next_progress = 10
    t_wall_start = time.perf_counter()

    while solver.t < T_END:
        solver.step()

        time_values.append(float(solver.t))
        speed_values.append(
            np.asarray(gen.speed(solver.y, solver.v), dtype=float).copy()
        )
        hvdc_values.append(hvdc_power_mw(solver.y, solver.v).copy())
        gen_power_values.append(total_gen_power_mw(solver.y, solver.v))

        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    wall_time = time.perf_counter() - t_wall_start
    n_steps = len(time_values) - 1


    # =========================================================================
    # RESULT PROCESSING
    # =========================================================================

    time_array = np.asarray(time_values)
    speed = np.asarray(speed_values)
    hvdc_power = np.asarray(hvdc_values)
    gen_power = np.asarray(gen_power_values)

    generator_frequency = nominal_frequency * (1.0 + speed)

    coi_speed = np.average(speed, axis=1, weights=inertia_weights)
    coi_frequency = nominal_frequency * (1.0 + coi_speed)

    frequency_spread_microhz = (
        np.max(generator_frequency, axis=1)
        - np.min(generator_frequency, axis=1)
    ) * 1e6

    # Deviations from the initial operating point (should be ~0 in baseline).
    hvdc_deviation = hvdc_power - hvdc_power[0]
    gen_power_deviation = gen_power - gen_power[0]


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
    print("Final COI frequency:", f"{coi_frequency[-1]:.12f} Hz")

    print(
        "Maximum total generator-power deviation:",
        f"{np.max(np.abs(gen_power_deviation)):.6e} MW",
    )

    print("\nHVDC operating point (positive = import to Nordic grid)")
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

    return {
        "time": time_array,
        "coi_frequency": coi_frequency,
        "frequency_spread": frequency_spread_microhz,
        "hvdc_deviation": hvdc_deviation,
        "hvdc_names": hvdc_names,
        "gen_power_deviation": gen_power_deviation,
        "nominal_frequency": nominal_frequency,
    }


def plot_results(results):

    t = results["time"]
    nominal_frequency = results["nominal_frequency"]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))

    fig.suptitle(
        "Nordic 45 (2025) - undisturbed baseline\n"
        "Initialized model without disturbance"
    )

    # COI frequency
    axes[0].plot(t, results["coi_frequency"], color="black", label="COI frequency")
    axes[0].axhline(nominal_frequency, color="gray", linestyle=":", label="Nominal frequency")
    axes[0].set_ylim(nominal_frequency - 0.05, nominal_frequency + 0.05)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].legend()
    axes[0].grid(True)

    # Generator-frequency spread
    axes[1].plot(
        t,
        results["frequency_spread"],
        color="tab:blue",
        label="Generator-frequency spread (max - min)",
    )
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_ylabel("Frequency spread\n($\\mu$Hz)")
    axes[1].legend()
    axes[1].grid(True)

    # Total generator power deviation
    axes[2].plot(
        t,
        results["gen_power_deviation"],
        color="tab:green",
        label="Total generator power - initial",
    )
    axes[2].axhline(0.0, color="gray", linestyle=":")
    axes[2].set_ylabel("$\\Delta P_{gen}$ (MW)")
    axes[2].legend()
    axes[2].grid(True)

    # HVDC power deviation
    for k, name in enumerate(results["hvdc_names"]):
        axes[3].plot(t, results["hvdc_deviation"][:, k], label=name)
    axes[3].axhline(0.0, color="gray", linestyle=":")
    axes[3].set_ylabel("$\\Delta P_{HVDC}$ (MW)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=4)
    axes[3].grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = run_simulation()
    plot_results(results)
