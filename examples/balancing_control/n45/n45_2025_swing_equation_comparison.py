import time
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.n45_2025 as model_data


LINEARIZATION_EPS = 1e-7
FREQUENCY_MIN_HZ = 0.1
FREQUENCY_MAX_HZ = 3.0
REAL_PART_TOLERANCE = 1e-6


VARIANTS = {
    "new_tops": "Current equation in new TOPS",
    "torque_balance": "Both P_m and P_e divided by speed",
    "power_balance": "Standard linearized power-balance equation",
}


def calculate_modal_quantities(eigenvalues):
    frequency_hz = (
        np.abs(eigenvalues.imag)
        / (2.0 * np.pi)
    )

    damping_ratio = np.divide(
        -eigenvalues.real,
        np.abs(eigenvalues),
        out=np.full(
            eigenvalues.shape,
            np.nan,
            dtype=float,
        ),
        where=np.abs(eigenvalues) > 0.0,
    )

    return frequency_hz, damping_ratio


def install_swing_equation(gen_model, variant):
    """
    Replace only the generator-speed derivative.

    All voltage-state equations and other GEN functionality
    remain unchanged.
    """

    if variant == "new_tops":
        # Retain the original TOPS implementation.
        return

    original_state_derivatives = (
        gen_model.state_derivatives
    )

    def modified_state_derivatives(self, dx, x, v):
        # First calculate every derivative using original TOPS.
        original_state_derivatives(dx, x, v)

        # Then overwrite only the speed derivative.
        dX = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        speed = X["speed"]
        speed_factor = 1.0 + speed

        mechanical_power = self.P_m(x, v)
        electrical_power = self.p_e(x, v)

        PF_n = p["PF_n"]
        H = p["H"] / PF_n

        if variant == "torque_balance":
            # Exact power-to-torque conversion for both
            # mechanical and electrical power.
            accelerating_torque = (
                mechanical_power
                - electrical_power / PF_n
            ) / speed_factor

        elif variant == "power_balance":
            # Conventional power-balance approximation
            # around nominal speed.
            accelerating_torque = (
                mechanical_power
                - electrical_power / PF_n
            )

        else:
            raise ValueError(
                f"Unknown swing-equation variant: {variant}"
            )

        dX["speed"][:] = (
            accelerating_torque
            - p["D"] * speed
        ) / (2.0 * H)

    gen_model.state_derivatives = MethodType(
        modified_state_derivatives,
        gen_model,
    )


def run_variant(variant):
    print("\n" + "=" * 72)
    print("Variant:", variant)
    print("Description:", VARIANTS[variant])
    print("=" * 72)

    # A completely new model is initialized for each variant.
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen_model = ps.gen["GEN"]

    install_swing_equation(
        gen_model,
        variant,
    )

    initial_derivative = ps.state_derivatives(
        0.0,
        ps.x_0,
        ps.v_0,
    )

    maximum_initial_derivative = float(
        np.max(np.abs(initial_derivative))
    )

    print("Power flow ready:", ps.power_flow_ready)
    print("Dynamic states:", ps.n_states)
    print(
        "Maximum initial derivative:",
        f"{maximum_initial_derivative:.10e}",
    )

    start_time = time.time()

    ps_linear = (
        dps_mdl.PowerSystemModelLinearization(ps)
    )

    ps_linear.eps = LINEARIZATION_EPS

    ps_linear.linearize()
    ps_linear.eigenvalue_decomposition()

    runtime = time.time() - start_time

    eigenvalues = np.asarray(
        ps_linear.eigs,
        dtype=complex,
    )

    frequency_hz, damping_ratio = (
        calculate_modal_quantities(eigenvalues)
    )

    electromechanical_mask = (
        np.isfinite(eigenvalues.real)
        & np.isfinite(eigenvalues.imag)
        & (
            eigenvalues.imag
            > REAL_PART_TOLERANCE
        )
        & (
            frequency_hz
            >= FREQUENCY_MIN_HZ
        )
        & (
            frequency_hz
            <= FREQUENCY_MAX_HZ
        )
    )

    electromechanical_indices = np.where(
        electromechanical_mask
    )[0]

    critical_index = electromechanical_indices[
        np.argmax(
            eigenvalues[
                electromechanical_indices
            ].real
        )
    ]

    critical_eigenvalue = eigenvalues[
        critical_index
    ]

    unstable_count = int(
        np.count_nonzero(
            eigenvalues.real
            > REAL_PART_TOLERANCE
        )
    )

    print(f"Linearization runtime: {runtime:.2f} s")
    print("Unstable eigenvalues:", unstable_count)
    print(
        "Critical eigenvalue:",
        f"{critical_eigenvalue.real:+.10f}",
        f"{critical_eigenvalue.imag:+.10f}j",
    )
    print(
        "Critical frequency:",
        f"{frequency_hz[critical_index]:.6f} Hz",
    )
    print(
        "Critical damping:",
        f"{100.0 * damping_ratio[critical_index]:.4f} %",
    )

    return {
        "variant": variant,
        "eigenvalues": eigenvalues,
        "frequency_hz": frequency_hz,
        "damping_ratio": damping_ratio,
        "electromechanical_mask": (
            electromechanical_mask
        ),
        "critical_index": critical_index,
    }


def print_comparison(results):
    print("\n")
    print("Swing-equation comparison")
    print("-" * 90)
    print(
        "Variant                 Real part"
        "       Imaginary part"
        "    Frequency"
        "     Damping"
    )

    for result in results:
        index = result["critical_index"]
        eigenvalue = result["eigenvalues"][index]

        print(
            f"{result['variant']:<22}"
            f"{eigenvalue.real:+14.8f}"
            f"{eigenvalue.imag:+17.8f}"
            f"{result['frequency_hz'][index]:12.6f} Hz"
            f"{100.0 * result['damping_ratio'][index]:11.4f} %"
        )


def plot_comparison(results):
    fig, axis = plt.subplots(
        figsize=(10, 7)
    )

    colors = {
        "new_tops": "tab:blue",
        "torque_balance": "tab:orange",
        "power_balance": "tab:green",
    }

    for result in results:
        variant = result["variant"]

        eigenvalues = result["eigenvalues"][
            result["electromechanical_mask"]
        ]

        axis.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            s=35,
            alpha=0.55,
            color=colors[variant],
            label=VARIANTS[variant],
        )

        critical = result["eigenvalues"][
            result["critical_index"]
        ]

        axis.scatter(
            critical.real,
            critical.imag,
            s=130,
            marker="x",
            linewidth=3,
            color=colors[variant],
        )

    axis.axvline(
        0.0,
        color="black",
        linestyle="--",
        label="Stability boundary",
    )

    axis.axhline(
        0.0,
        color="gray",
        linestyle=":",
    )

    axis.set_title(
        "Nordic 45 (2025) – swing-equation comparison\n"
        "Electromechanical modes, 0.1–3.0 Hz"
    )

    axis.set_xlabel(
        r"Real part, $\sigma$ (s$^{-1}$)"
    )

    axis.set_ylabel(
        r"Imaginary part, $\omega$ (rad/s)"
    )

    axis.grid(True)
    axis.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(
        "Imported model-data file:",
        model_data.__file__,
    )
    print(
        "Linearization epsilon:",
        f"{LINEARIZATION_EPS:.0e}",
    )
    print(
        "The installed TOPS source files "
        "are not modified."
    )

    results = [
        run_variant(variant)
        for variant in VARIANTS
    ]

    print_comparison(results)
    plot_comparison(results)
