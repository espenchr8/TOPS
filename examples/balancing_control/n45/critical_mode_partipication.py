import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.n45_2025 as model_data


def as_text(value):
    """Convert NumPy strings/bytes to ordinary stripped text."""
    if isinstance(value, bytes):
        return value.decode().strip()
    return str(value).strip()


if __name__ == "__main__":
    linearization_epsilon = 1e-7
    number_to_print = 20

    print("Imported model-data file:", model_data.__file__)
    print("No model parameters are modified by this script.")

    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]

    print("\nN45 critical-mode participation analysis")
    print("----------------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Dynamic states:", ps.n_states)

    dx_initial = ps.state_derivatives(0.0, ps.x_0, ps.v_0)
    print(
        "Maximum initial derivative:",
        f"{np.max(np.abs(dx_initial)):.10e}",
    )

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.eps = linearization_epsilon

    print(f"\nLinearizing with eps={linearization_epsilon:.0e} ...")
    start_time = time.time()
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    print(f"Linearization runtime: {time.time() - start_time:.2f} s")

    eigs = ps_lin.eigs

    # Select the rightmost oscillatory mode in the upper half-plane.
    oscillatory = np.where(eigs.imag > 1e-6)[0]
    if oscillatory.size == 0:
        raise RuntimeError("No oscillatory eigenvalue was found.")

    critical_mode_index = oscillatory[
        np.argmax(eigs[oscillatory].real)
    ]
    critical_eigenvalue = eigs[critical_mode_index]
    critical_frequency = critical_eigenvalue.imag / (2.0 * np.pi)
    critical_damping = (
        -critical_eigenvalue.real
        / np.abs(critical_eigenvalue)
        * 100.0
    )

    print("\nCritical oscillatory mode")
    print("-------------------------")
    print("Eigenvalue index:", critical_mode_index)
    print(
        "Eigenvalue:",
        f"{critical_eigenvalue.real:+.10f} "
        f"{critical_eigenvalue.imag:+.10f}j",
    )
    print("Frequency:", f"{critical_frequency:.6f} Hz")
    print("Damping:", f"{critical_damping:.4f} %")

    speed_indices = np.asarray(
        gen.state_idx_global["speed"], dtype=int
    )
    angle_indices = np.asarray(
        gen.state_idx_global["angle"], dtype=int
    )

    # Right eigenvectors describe the mode shape. Left and right
    # eigenvectors together give the participation factors.
    right_vector = ps_lin.rev[:, critical_mode_index]
    left_vector = ps_lin.lev[critical_mode_index, :]
    state_participation = right_vector * left_vector

    speed_participation = np.abs(
        state_participation[speed_indices]
    )
    angle_participation = np.abs(
        state_participation[angle_indices]
    )
    generator_participation = (
        speed_participation + angle_participation
    )

    maximum_participation = np.max(generator_participation)
    if maximum_participation > 0.0:
        generator_participation /= maximum_participation

    speed_mode_shape = right_vector[speed_indices]
    maximum_mode_shape = np.max(np.abs(speed_mode_shape))
    if maximum_mode_shape > 0.0:
        normalized_mode_shape = speed_mode_shape / maximum_mode_shape
    else:
        normalized_mode_shape = speed_mode_shape.copy()

    mode_shape_magnitude = np.abs(normalized_mode_shape)
    mode_shape_phase = np.angle(
        normalized_mode_shape, deg=True
    )

    generator_names = [as_text(name) for name in gen.par["name"]]
    generator_buses = [as_text(bus) for bus in gen.par["bus"]]

    pss_generator_names = set()
    if hasattr(ps, "pss") and "STAB1" in ps.pss:
        pss_generator_names = {
            as_text(name)
            for name in ps.pss["STAB1"].par["gen"]
        }

    ranking = np.argsort(generator_participation)[::-1]
    number_to_print = min(number_to_print, gen.n_units)

    print("\nGenerator ranking for the critical mode")
    print("-" * 94)
    print(
        "Rank  Generator       Bus       Participation  "
        "Speed shape   Phase       STAB1"
    )
    print("-" * 94)

    for rank, generator_index in enumerate(
        ranking[:number_to_print], start=1
    ):
        generator_name = generator_names[generator_index]
        has_pss = generator_name in pss_generator_names

        print(
            f"{rank:>4}  "
            f"{generator_name:<14}  "
            f"{generator_buses[generator_index]:<8}  "
            f"{generator_participation[generator_index]:>13.6f}  "
            f"{mode_shape_magnitude[generator_index]:>11.6f}  "
            f"{mode_shape_phase[generator_index]:>8.2f} deg  "
            f"{'Yes' if has_pss else 'No'}"
        )

    dominant_threshold = 0.10
    dominant_indices = ranking[
        generator_participation[ranking] >= dominant_threshold
    ]
    dominant_without_pss = [
        generator_names[index]
        for index in dominant_indices
        if generator_names[index] not in pss_generator_names
    ]

    print("\nPSS coverage summary")
    print("--------------------")
    print("Generators:", gen.n_units)
    print("Generators with STAB1:", len(pss_generator_names))
    print(
        "Dominant generators (normalized participation >= 0.10):",
        len(dominant_indices),
    )
    print(
        "Dominant generators without STAB1:",
        dominant_without_pss if dominant_without_pss else "None",
    )

    # Plot the speed mode shape. Opposite directions indicate generators
    # or areas oscillating against each other.
    dominant_for_plot = ranking[:number_to_print]
    plot_names = [generator_names[i] for i in dominant_for_plot]
    plot_values = normalized_mode_shape[dominant_for_plot]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    fig.suptitle(
        "Nordic 45 (2025) – critical-mode participation\n"
        f"{critical_frequency:.3f} Hz, "
        f"damping {critical_damping:.3f} %"
    )

    colors = [
        "tab:blue" if generator_names[i] in pss_generator_names
        else "tab:orange"
        for i in dominant_for_plot
    ]

    y_positions = np.arange(number_to_print)
    axes[0].barh(
        y_positions,
        generator_participation[dominant_for_plot],
        color=colors,
    )
    axes[0].set_yticks(y_positions, labels=plot_names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Normalized participation")
    axes[0].set_title("Generator participation")
    axes[0].grid(True, axis="x")

    axes[1].scatter(
        plot_values.real,
        plot_values.imag,
        c=colors,
        s=55,
    )
    for name, value in zip(plot_names, plot_values):
        axes[1].annotate(
            name,
            (value.real, value.imag),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].axvline(0.0, color="gray", linestyle=":")
    axes[1].set_xlabel("Real part of normalized speed mode shape")
    axes[1].set_ylabel("Imaginary part")
    axes[1].set_title("Speed mode shape")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(True)

    fig.text(
        0.5,
        0.01,
        "Blue: STAB1 installed   Orange: no STAB1",
        ha="center",
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    plt.show()
