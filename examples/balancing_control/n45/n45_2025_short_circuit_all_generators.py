"""Temporary three-phase N45 fault; plot all 47 generator responses."""

import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import n45_2025 as model_data


SCRIPT_VERSION = "N45-SC-all-generators-v1-20260915"

# Simulation settings.
T_END = 30.0
MAX_STEP = 5e-3

# Choose the fault location here: 0 is the first generator and 46 the last.
FAULT_GENERATOR_INDEX = 0

# Temporary three-phase short circuit.
FAULT_TIME = 1.0
FAULT_CLEARING_TIME = 1.05
FAULT_ADMITTANCE = 1e6
TIME_TOLERANCE = 1e-9


def run_simulation():
    """Run one short circuit and store the response of all generators."""
    print("Script version:", SCRIPT_VERSION)
    print("Imported model-data file:", model_data.__file__)
    print("No model parameters or installed TOPS files are modified.")

    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    if not ps.power_flow_ready:
        raise RuntimeError("N45 power flow did not converge.")

    gen = ps.gen["GEN"]
    generator_names = np.asarray(gen.par["name"], dtype=str)
    generator_buses = np.asarray(gen.par["bus"], dtype=str)

    if gen.n_units != 47:
        raise RuntimeError(f"Expected 47 generators, found {gen.n_units}.")
    if not 0 <= FAULT_GENERATOR_INDEX < gen.n_units:
        raise ValueError(
            f"FAULT_GENERATOR_INDEX must be between 0 and {gen.n_units - 1}."
        )

    # Print the index list so another fault location is easy to select.
    print("\nAvailable generator terminal buses")
    print("----------------------------------")
    for index, (name, bus) in enumerate(zip(generator_names, generator_buses)):
        marker = "  <-- selected" if index == FAULT_GENERATOR_INDEX else ""
        print(f"{index:2d}: {name:<16s} bus {bus}{marker}")

    fault_generator_name = generator_names[FAULT_GENERATOR_INDEX]
    fault_bus_name = generator_buses[FAULT_GENERATOR_INDEX]
    fault_bus_index = int(
        gen.bus_idx_red["terminal"][FAULT_GENERATOR_INDEX]
    )

    v_initial = ps.solve_algebraic(0.0, ps.x0)
    dx_initial = ps.state_derivatives(0.0, ps.x0, v_initial)
    max_initial_derivative = float(np.max(np.abs(dx_initial)))
    if not np.isfinite(max_initial_derivative):
        raise RuntimeError("Initial derivatives contain NaN or infinity.")

    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0.copy(),
        T_END,
        max_step=MAX_STEP,
    )

    results = defaultdict(list)
    results["time"].append(0.0)
    results["generator_speed"].append(
        np.asarray(gen.speed(ps.x0, v_initial)).copy()
    )
    results["fault_bus_voltage"].append(float(abs(v_initial[fault_bus_index])))

    fault_active = False
    fault_applied = False
    fault_cleared = False
    next_progress = 10
    start_time = time.perf_counter()

    print("\nShort-circuit simulation")
    print("------------------------")
    print("Generator index:", FAULT_GENERATOR_INDEX)
    print("Generator:", fault_generator_name)
    print("Terminal bus:", fault_bus_name)
    print("Fault interval:", f"{FAULT_TIME:.3f}-{FAULT_CLEARING_TIME:.3f} s")
    print("Maximum initial derivative:", f"{max_initial_derivative:.6e}")

    while solver.t < T_END:
        t = solver.t
        should_fault_be_active = (
            t + TIME_TOLERANCE >= FAULT_TIME
            and t < FAULT_CLEARING_TIME - TIME_TOLERANCE
        )

        if should_fault_be_active and not fault_active:
            ps.y_bus_red_mod[fault_bus_index, fault_bus_index] = FAULT_ADMITTANCE
            solver.v[:] = ps.solve_algebraic(t, solver.y)
            fault_active = True
            fault_applied = True
            print(f"Applied short circuit at t={t:.3f} s.")

        elif not should_fault_be_active and fault_active:
            ps.y_bus_red_mod[fault_bus_index, fault_bus_index] = 0.0
            solver.v[:] = ps.solve_algebraic(t, solver.y)
            fault_active = False
            fault_cleared = True
            print(f"Cleared short circuit at t={t:.3f} s.")

        solver.step()

        if not np.all(np.isfinite(solver.y)) or not np.all(np.isfinite(solver.v)):
            raise FloatingPointError(
                f"Simulation became non-finite at t={solver.t:.6f} s."
            )

        results["time"].append(float(solver.t))
        results["generator_speed"].append(
            np.asarray(gen.speed(solver.y, solver.v)).copy()
        )
        results["fault_bus_voltage"].append(
            float(abs(solver.v[fault_bus_index]))
        )

        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    ps.y_bus_red_mod[fault_bus_index, fault_bus_index] = 0.0

    if not fault_applied or not fault_cleared:
        raise RuntimeError("The short circuit was not both applied and cleared.")

    simulation_time = time.perf_counter() - start_time
    sim_time = np.asarray(results["time"])
    generator_speed = np.asarray(results["generator_speed"])
    bus_voltage = np.asarray(results["fault_bus_voltage"])

    max_abs_speed = np.max(np.abs(generator_speed), axis=0)
    largest_response_index = int(np.argmax(max_abs_speed))

    print("\nSimulation result")
    print("-----------------")
    print("Runtime:", f"{simulation_time:.2f} s")
    print(
        "Largest absolute generator-speed deviation:",
        f"{max_abs_speed[largest_response_index]:.6e} p.u.",
    )
    print("Generator with largest response:", generator_names[largest_response_index])
    print("Minimum fault-bus voltage:", f"{np.min(bus_voltage):.6f} p.u.")

    return {
        "time": sim_time,
        "generator_speed": generator_speed,
        "generator_names": generator_names,
        "fault_bus_voltage": bus_voltage,
        "fault_generator_index": FAULT_GENERATOR_INDEX,
        "fault_generator_name": fault_generator_name,
        "fault_bus_name": fault_bus_name,
    }


def plot_results(results):
    """Plot fault-bus voltage and all 47 generator-speed responses."""
    time_axis = results["time"]
    speed = results["generator_speed"]
    selected = results["fault_generator_index"]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 9))
    fig.suptitle(
        "Nordic 45 (2025) - temporary three-phase short circuit\n"
        f"Generator index {selected}: {results['fault_generator_name']}, "
        f"bus {results['fault_bus_name']}, "
        f"{FAULT_TIME:.2f}-{FAULT_CLEARING_TIME:.2f} s"
    )

    axes[0].plot(
        time_axis,
        results["fault_bus_voltage"],
        color="tab:blue",
        linewidth=1.7,
        label=f"Bus {results['fault_bus_name']}",
    )
    axes[0].set_ylabel("Voltage (p.u.)")
    axes[0].legend()
    axes[0].grid(True)

    colors = plt.cm.turbo(np.linspace(0.0, 1.0, len(results["generator_names"])))
    for index, name in enumerate(results["generator_names"]):
        is_fault_generator = index == selected
        axes[1].plot(
            time_axis,
            speed[:, index],
            color="black" if is_fault_generator else colors[index],
            linewidth=2.4 if is_fault_generator else 0.8,
            alpha=1.0 if is_fault_generator else 0.8,
            label=f"{index}: {name}" + (" (fault bus)" if is_fault_generator else ""),
            zorder=5 if is_fault_generator else 2,
        )

    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_ylabel("Speed deviation (p.u.)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True)
    axes[1].legend(
        fontsize=6.5,
        ncol=3,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )

    for axis in axes:
        axis.axvspan(
            FAULT_TIME,
            FAULT_CLEARING_TIME,
            color="tab:red",
            alpha=0.18,
            label="Short circuit" if axis is axes[0] else None,
        )

    fig.tight_layout(rect=(0.0, 0.0, 0.79, 0.95))
    plt.show()


if __name__ == "__main__":
    simulation_results = run_simulation()
    plot_results(simulation_results)
