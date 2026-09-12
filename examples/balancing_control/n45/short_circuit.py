import time
from collections import defaultdict

import matplotlib.pyplot as plt

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.ps_models.n45_2025 as model_data


# Simulation settings.
T_END = 10.0
MAX_STEP = 5e-3

# Short-circuit settings.
FAULT_TIME = 1.0
FAULT_CLEARING_TIME = 1.05
FAULT_ADMITTANCE = 1e6

# Handles small floating-point errors in the simulation time.
TIME_TOLERANCE = 1e-9


def run_simulation():
    """Run an N45 simulation with a temporary three-phase short circuit."""

    # Load and initialize the N45 model.
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen_model = ps.gen["GEN"]

    # The short circuit is applied at the terminal bus of the first generator.
    fault_generator_index = 0

    fault_bus_index = int(
        gen_model.bus_idx_red["terminal"][fault_generator_index]
    )

    fault_generator_name = str(
        gen_model.par["name"][fault_generator_index]
    )

    fault_bus_name = str(
        gen_model.par["bus"][fault_generator_index]
    )

    # Create the numerical solver.
    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x_0.copy(),
        T_END,
        max_step=MAX_STEP,
    )

    results = defaultdict(list)

    t = 0.0
    fault_active = False
    next_progress = 10
    start_time = time.time()

    print("\nShort-circuit simulation")
    print("------------------------")
    print("Generator:", fault_generator_name)
    print("Terminal bus:", fault_bus_name)
    print(
        "Fault interval:",
        f"{FAULT_TIME:.3f}–{FAULT_CLEARING_TIME:.3f} s",
    )

    while t < T_END:
        should_fault_be_active = (
            t + TIME_TOLERANCE >= FAULT_TIME
            and t < FAULT_CLEARING_TIME - TIME_TOLERANCE
        )

        if should_fault_be_active:
            ps.y_bus_red_mod[
                fault_bus_index,
                fault_bus_index,
            ] = FAULT_ADMITTANCE

            if not fault_active:
                print(
                    f"Applied short circuit at t = {t:.3f} s."
                )
                fault_active = True

        else:
            ps.y_bus_red_mod[
                fault_bus_index,
                fault_bus_index,
            ] = 0.0

            if fault_active:
                print(
                    f"Cleared short circuit at t = {t:.3f} s."
                )
                fault_active = False

        # Simulate the next time step.
        solver.step()

        t = solver.t
        x = solver.y
        v = solver.v

        # Store results.
        results["time"].append(float(t))

        results["generator_speed"].append(
            float(
                gen_model.speed(x, v)[fault_generator_index]
            )
        )

        results["fault_bus_voltage"].append(
            float(abs(v[fault_bus_index]))
        )

        progress = int(100 * t / T_END)

        if progress >= next_progress:
            print(
                f"Simulation progress: "
                f"{min(progress, 100)}%"
            )
            next_progress += 10

    # Remove the temporary network modification.
    ps.y_bus_red_mod[
        fault_bus_index,
        fault_bus_index,
    ] = 0.0

    print(
        "Simulation completed in "
        f"{time.time() - start_time:.2f} seconds."
    )

    return {
        "time": results["time"],
        "generator_speed": results["generator_speed"],
        "fault_bus_voltage": results["fault_bus_voltage"],
        "fault_generator_name": fault_generator_name,
        "fault_bus_name": fault_bus_name,
    }


def plot_results(results):
    """Plot voltage and generator-speed response."""

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
    )

    fig.suptitle(
        "Nordic 45 (2025) – temporary three-phase short circuit\n"
        f"Bus {results['fault_bus_name']}, "
        f"{FAULT_TIME:.2f}–{FAULT_CLEARING_TIME:.2f} s"
    )

    # Fault-bus voltage.
    axes[0].plot(
        results["time"],
        results["fault_bus_voltage"],
        color="tab:blue",
        label=f"Bus {results['fault_bus_name']}",
    )

    axes[0].axvspan(
        FAULT_TIME,
        FAULT_CLEARING_TIME,
        color="tab:red",
        alpha=0.2,
        label="Short circuit",
    )

    axes[0].set_ylabel("Voltage (p.u.)")
    axes[0].legend()
    axes[0].grid(True)

    # Generator-speed deviation.
    axes[1].plot(
        results["time"],
        results["generator_speed"],
        color="tab:blue",
        label=results["fault_generator_name"],
    )

    axes[1].axhline(
        0.0,
        color="gray",
        linestyle=":",
    )

    axes[1].axvspan(
        FAULT_TIME,
        FAULT_CLEARING_TIME,
        color="tab:red",
        alpha=0.2,
    )

    axes[1].set_ylabel("Speed deviation (p.u.)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulation_results = run_simulation()
    plot_results(simulation_results)
