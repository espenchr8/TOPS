import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.n45_2025 as model_data


# Electromechanical frequency range.
EM_FREQUENCY_MIN_HZ = 0.1
EM_FREQUENCY_MAX_HZ = 3.0

# Numerical tolerance used when classifying eigenvalues.
REAL_PART_TOLERANCE = 1e-6

# Modes below this damping ratio are printed as weakly damped.
WEAK_DAMPING_LIMIT = 0.05


def calculate_modal_quantities(eigenvalues):
    """Calculate modal frequency and damping ratio."""

    magnitude = np.abs(eigenvalues)

    frequency_hz = (
        np.abs(eigenvalues.imag)
        / (2.0 * np.pi)
    )

    damping_ratio = np.divide(
        -eigenvalues.real,
        magnitude,
        out=np.full(
            eigenvalues.shape,
            np.nan,
            dtype=float,
        ),
        where=magnitude > 0.0,
    )

    return frequency_hz, damping_ratio


def run_modal_analysis():
    """Linearize the N45 model and calculate its eigenvalues."""

    print("Imported model-data file:", model_data.__file__)
    print("No model parameters are modified by this script.")

    # -----------------------------------------------------------------------
    # Load and initialize N45
    # -----------------------------------------------------------------------

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    dx_initial = ps.state_derivatives(
        0.0,
        ps.x_0,
        ps.v_0,
    )

    maximum_initial_derivative = float(
        np.max(np.abs(dx_initial))
    )

    print("\nN45 modal-analysis initialization")
    print("---------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Dynamic states:", ps.n_states)
    print(
        "Maximum initial derivative:",
        f"{maximum_initial_derivative:.6e}",
    )

    # -----------------------------------------------------------------------
    # Linearize the nonlinear DAE model
    # -----------------------------------------------------------------------

    print("\nLinearizing N45 model...")

    start_time = time.time()

    ps_linear = dps_mdl.PowerSystemModelLinearization(ps)

    ps_linear.linearize()
    ps_linear.eigenvalue_decomposition()

    calculation_time = time.time() - start_time

    eigenvalues = np.asarray(
        ps_linear.eigs,
        dtype=complex,
    )

    frequency_hz, damping_ratio = (
        calculate_modal_quantities(eigenvalues)
    )

    print(
        "Linearization completed in",
        f"{calculation_time:.2f} seconds.",
    )

    # -----------------------------------------------------------------------
    # Classify eigenvalues
    # -----------------------------------------------------------------------

    finite_modes = (
        np.isfinite(eigenvalues.real)
        & np.isfinite(eigenvalues.imag)
    )

    unstable_modes = (
        finite_modes
        & (
            eigenvalues.real
            > REAL_PART_TOLERANCE
        )
    )

    near_zero_modes = (
        finite_modes
        & (
            np.abs(eigenvalues.real)
            <= REAL_PART_TOLERANCE
        )
        & (
            np.abs(eigenvalues.imag)
            <= REAL_PART_TOLERANCE
        )
    )

    # Keep only the positive-imaginary member of each conjugate pair.
    electromechanical_modes = (
        finite_modes
        & (
            eigenvalues.imag
            > REAL_PART_TOLERANCE
        )
        & (
            frequency_hz
            >= EM_FREQUENCY_MIN_HZ
        )
        & (
            frequency_hz
            <= EM_FREQUENCY_MAX_HZ
        )
    )

    weakly_damped_modes = (
        electromechanical_modes
        & (
            damping_ratio
            < WEAK_DAMPING_LIMIT
        )
    )

    print("\nEigenvalue summary")
    print("------------------")
    print("Total eigenvalues:", len(eigenvalues))
    print(
        "Unstable eigenvalues:",
        int(np.count_nonzero(unstable_modes)),
    )
    print(
        "Near-zero eigenvalues:",
        int(np.count_nonzero(near_zero_modes)),
    )
    print(
        "Electromechanical mode pairs:",
        int(np.count_nonzero(electromechanical_modes)),
    )
    print(
        "Weakly damped electromechanical mode pairs:",
        int(np.count_nonzero(weakly_damped_modes)),
    )

    # -----------------------------------------------------------------------
    # Print unstable eigenvalues
    # -----------------------------------------------------------------------

    unstable_indices = np.where(
        unstable_modes
    )[0]

    if len(unstable_indices) > 0:
        unstable_indices = unstable_indices[
            np.argsort(
                eigenvalues[unstable_indices].real
            )[::-1]
        ]

        print("\nUnstable eigenvalues")
        print("--------------------")
        print(
            "Index       Real          Imag"
            "       Frequency      Damping"
        )

        for index in unstable_indices:
            print(
                f"{index:5d}"
                f"  {eigenvalues[index].real:12.6f}"
                f"  {eigenvalues[index].imag:12.6f}"
                f"  {frequency_hz[index]:10.4f} Hz"
                f"  {100.0 * damping_ratio[index]:9.3f} %"
            )
    else:
        print(
            "\nNo eigenvalues have a real part greater than "
            f"{REAL_PART_TOLERANCE:.1e}."
        )

    # -----------------------------------------------------------------------
    # Print electromechanical modes
    # -----------------------------------------------------------------------

    electromechanical_indices = np.where(
        electromechanical_modes
    )[0]

    electromechanical_indices = (
        electromechanical_indices[
            np.argsort(
                damping_ratio[
                    electromechanical_indices
                ]
            )
        ]
    )

    print("\nElectromechanical modes")
    print("-----------------------")
    print(
        "Mode        Eigenvalue"
        "                 Frequency      Damping"
    )

    if len(electromechanical_indices) == 0:
        print(
            "No modes were found in the selected "
            f"{EM_FREQUENCY_MIN_HZ:.1f}–"
            f"{EM_FREQUENCY_MAX_HZ:.1f} Hz range."
        )

    for mode_number, index in enumerate(
        electromechanical_indices,
        start=1,
    ):
        eigenvalue = eigenvalues[index]

        stability_label = (
            "UNSTABLE"
            if eigenvalue.real > REAL_PART_TOLERANCE
            else ""
        )

        print(
            f"{mode_number:4d}"
            f"   {eigenvalue.real:10.5f}"
            f" {eigenvalue.imag:+10.5f}j"
            f"   {frequency_hz[index]:9.4f} Hz"
            f"   {100.0 * damping_ratio[index]:8.3f} %"
            f"   {stability_label}"
        )

    return {
        "eigenvalues": eigenvalues,
        "frequency_hz": frequency_hz,
        "damping_ratio": damping_ratio,
        "unstable_modes": unstable_modes,
        "electromechanical_modes": (
            electromechanical_modes
        ),
        "weakly_damped_modes": (
            weakly_damped_modes
        ),
    }


def plot_eigenvalues(results):
    """Plot the complete spectrum and electromechanical modes."""

    eigenvalues = results["eigenvalues"]

    unstable_modes = results["unstable_modes"]

    electromechanical_modes = (
        results["electromechanical_modes"]
    )

    stable_modes = ~unstable_modes

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    fig.suptitle(
        "Nordic 45 (2025) – small-signal eigenvalue analysis\n"
        "Original model parameters"
    )

    # -----------------------------------------------------------------------
    # Complete eigenvalue spectrum
    # -----------------------------------------------------------------------

    axes[0].scatter(
        eigenvalues[stable_modes].real,
        eigenvalues[stable_modes].imag,
        s=22,
        color="tab:blue",
        alpha=0.75,
        label="Stable or neutral",
    )

    if np.any(unstable_modes):
        axes[0].scatter(
            eigenvalues[unstable_modes].real,
            eigenvalues[unstable_modes].imag,
            s=40,
            color="tab:red",
            marker="x",
            linewidth=1.8,
            label="Unstable",
        )

    axes[0].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Stability boundary",
    )

    axes[0].axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
    )

    axes[0].set_title("Complete eigenvalue spectrum")
    axes[0].set_xlabel(
        r"Real part, $\sigma$ (s$^{-1}$)"
    )
    axes[0].set_ylabel(
        r"Imaginary part, $\omega$ (rad/s)"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True)

    # -----------------------------------------------------------------------
    # Electromechanical modes
    # -----------------------------------------------------------------------

    em_eigenvalues = eigenvalues[
        electromechanical_modes
    ]

    if len(em_eigenvalues) > 0:
        em_unstable = (
            em_eigenvalues.real
            > REAL_PART_TOLERANCE
        )

        axes[1].scatter(
            em_eigenvalues[~em_unstable].real,
            em_eigenvalues[~em_unstable].imag,
            s=45,
            color="tab:blue",
            label="Stable electromechanical modes",
        )

        if np.any(em_unstable):
            axes[1].scatter(
                em_eigenvalues[em_unstable].real,
                em_eigenvalues[em_unstable].imag,
                s=55,
                color="tab:red",
                marker="x",
                linewidth=2.0,
                label="Unstable electromechanical modes",
            )

    axes[1].axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Stability boundary",
    )

    axes[1].axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
    )

    axes[1].set_title(
        "Electromechanical modes\n"
        f"{EM_FREQUENCY_MIN_HZ:.1f}–"
        f"{EM_FREQUENCY_MAX_HZ:.1f} Hz"
    )

    axes[1].set_xlabel(
        r"Real part, $\sigma$ (s$^{-1}$)"
    )

    axes[1].set_ylabel(
        r"Imaginary part, $\omega$ (rad/s)"
    )

    axes[1].legend(loc="best")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    modal_results = run_modal_analysis()
    plot_eigenvalues(modal_results)
