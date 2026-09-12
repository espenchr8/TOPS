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


def linearize_system(ps, epsilon):
    """Return one completed TOPS linearization."""
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.eps = epsilon
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    return ps_lin


def rightmost_oscillatory_mode(eigenvalues):
    """Find the rightmost oscillatory eigenvalue in the upper half-plane."""
    candidates = np.where(eigenvalues.imag > 1e-6)[0]
    if candidates.size == 0:
        raise RuntimeError("No oscillatory eigenvalue was found.")
    return candidates[np.argmax(eigenvalues[candidates].real)]


def matching_mode(eigenvalues, reference_eigenvalue):
    """Find the mode closest to the baseline critical eigenvalue."""
    candidates = np.where(eigenvalues.imag > 1e-6)[0]
    if candidates.size == 0:
        raise RuntimeError("No oscillatory eigenvalue was found.")
    distances = np.abs(eigenvalues[candidates] - reference_eigenvalue)
    return candidates[np.argmin(distances)]


if __name__ == "__main__":
    linearization_epsilon = 1e-7
    relative_gain_change = 0.10

    print("Imported model-data file:", model_data.__file__)
    print("No installed TOPS source files are modified.")
    print(
        "Each STAB1 gain is changed temporarily by",
        f"{100 * relative_gain_change:.1f}%.",
    )

    ps = dps.PowerSystemModel(model=model_data.load())
    ps.init_dyn_sim()

    if not hasattr(ps, "pss") or "STAB1" not in ps.pss:
        raise RuntimeError("The initialized model contains no STAB1 units.")

    pss = ps.pss["STAB1"]

    print("\nN45 PSS-gain sensitivity analysis")
    print("---------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Dynamic states:", ps.n_states)
    print("STAB1 units:", pss.n_units)

    dx_initial = ps.state_derivatives(0.0, ps.x_0, ps.v_0)
    print(
        "Maximum initial derivative:",
        f"{np.max(np.abs(dx_initial)):.10e}",
    )

    print(f"\nBaseline linearization with eps={linearization_epsilon:.0e} ...")
    start_time = time.time()
    baseline_linearization = linearize_system(
        ps,
        linearization_epsilon,
    )
    baseline_mode_index = rightmost_oscillatory_mode(
        baseline_linearization.eigs
    )
    baseline_eigenvalue = baseline_linearization.eigs[
        baseline_mode_index
    ]
    baseline_frequency = baseline_eigenvalue.imag / (2.0 * np.pi)
    baseline_damping = (
        -baseline_eigenvalue.real
        / np.abs(baseline_eigenvalue)
        * 100.0
    )

    print(f"Baseline runtime: {time.time() - start_time:.2f} s")
    print(
        "Critical eigenvalue:",
        f"{baseline_eigenvalue.real:+.10f} "
        f"{baseline_eigenvalue.imag:+.10f}j",
    )
    print("Critical frequency:", f"{baseline_frequency:.6f} Hz")
    print("Critical damping:", f"{baseline_damping:.4f} %")

    # STAB1.add_blocks() creates this Gain object from the K column.
    # Changing this array therefore changes the gain actually evaluated
    # during linearization.
    gain_values = pss.gain.par["K"]
    original_gains = gain_values.copy()
    pss_names = [as_text(value) for value in pss.par["name"]]
    generator_names = [as_text(value) for value in pss.par["gen"]]

    results = []
    total_start_time = time.time()

    print("\nTesting STAB1 units")
    print("-------------------")

    try:
        for unit_index in range(pss.n_units):
            original_gain = float(original_gains[unit_index])
            gain_step = relative_gain_change * original_gain

            if gain_step == 0.0:
                print(
                    f"[{unit_index + 1:>2}/{pss.n_units}] "
                    f"{generator_names[unit_index]} skipped: K = 0"
                )
                continue

            gain_values[unit_index] = original_gain + gain_step

            trial_start_time = time.time()
            trial_linearization = linearize_system(
                ps,
                linearization_epsilon,
            )
            trial_mode_index = matching_mode(
                trial_linearization.eigs,
                baseline_eigenvalue,
            )
            trial_eigenvalue = trial_linearization.eigs[trial_mode_index]
            trial_frequency = trial_eigenvalue.imag / (2.0 * np.pi)
            trial_damping = (
                -trial_eigenvalue.real
                / np.abs(trial_eigenvalue)
                * 100.0
            )

            # Restore this gain before the next unit is tested.
            gain_values[unit_index] = original_gain

            real_part_change = (
                trial_eigenvalue.real - baseline_eigenvalue.real
            )
            real_part_sensitivity = real_part_change / gain_step
            damping_change = trial_damping - baseline_damping

            results.append(
                {
                    "pss": pss_names[unit_index],
                    "generator": generator_names[unit_index],
                    "original_gain": original_gain,
                    "trial_gain": original_gain + gain_step,
                    "eigenvalue": trial_eigenvalue,
                    "frequency": trial_frequency,
                    "damping": trial_damping,
                    "delta_real": real_part_change,
                    "dreal_dk": real_part_sensitivity,
                    "delta_damping": damping_change,
                }
            )

            direction = (
                "improves"
                if real_part_change < 0.0
                else "worsens"
            )
            print(
                f"[{unit_index + 1:>2}/{pss.n_units}] "
                f"{generator_names[unit_index]:<14}  "
                f"Delta real={real_part_change:+.3e}  "
                f"{direction:<8}  "
                f"{time.time() - trial_start_time:.2f} s"
            )
    finally:
        # Protect the active model even if a trial fails.
        gain_values[:] = original_gains

    if not results:
        raise RuntimeError("No non-zero STAB1 gains were available to test.")

    results.sort(key=lambda item: item["delta_real"])

    print("\nPSS gain-sensitivity ranking")
    print("-" * 126)
    print(
        "Rank  PSS             Generator       K original  K trial  "
        "Delta real      dReal/dK       Damping     Delta damping  Effect"
    )
    print("-" * 126)

    for rank, result in enumerate(results, start=1):
        effect = (
            "IMPROVES"
            if result["delta_real"] < 0.0
            else "WORSENS"
        )
        print(
            f"{rank:>4}  "
            f"{result['pss']:<14}  "
            f"{result['generator']:<14}  "
            f"{result['original_gain']:>10.4f}  "
            f"{result['trial_gain']:>7.4f}  "
            f"{result['delta_real']:>+12.5e}  "
            f"{result['dreal_dk']:>+12.5e}  "
            f"{result['damping']:>9.4f}%  "
            f"{result['delta_damping']:>+12.5f}%  "
            f"{effect}"
        )

    print("\nInterpretation")
    print("--------------")
    print("Negative Delta real: a 10% gain increase moves the mode left.")
    print("Positive Delta real: a 10% gain increase moves the mode right.")
    print("This script has restored every original STAB1 gain.")
    print(f"Total sensitivity runtime: {time.time() - total_start_time:.2f} s")

    generators = [result["generator"] for result in results]
    delta_real_values = np.array(
        [result["delta_real"] for result in results]
    )
    bar_colors = np.where(
        delta_real_values < 0.0,
        "tab:green",
        "tab:red",
    )

    figure_height = max(7.0, 0.28 * len(results))
    fig, ax = plt.subplots(figsize=(11, figure_height))
    y_positions = np.arange(len(results))
    ax.barh(
        y_positions,
        delta_real_values,
        color=bar_colors,
    )
    ax.set_yticks(y_positions, labels=generators)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linestyle="--")
    ax.set_xlabel(
        "Change in critical-mode real part for +10% STAB1 gain (1/s)"
    )
    ax.set_ylabel("Generator")
    ax.set_title(
        "Nordic 45 (2025) – STAB1 gain sensitivity\n"
        f"Baseline mode: {baseline_frequency:.3f} Hz, "
        f"damping {baseline_damping:.3f}%"
    )
    ax.grid(True, axis="x")
    fig.text(
        0.5,
        0.01,
        "Green: increased gain improves damping   "
        "Red: increased gain worsens damping",
        ha="center",
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    plt.show()
