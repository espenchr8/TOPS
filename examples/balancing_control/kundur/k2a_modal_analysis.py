"""Small-signal modal analysis of the Kundur two-area HYGOV baseline."""

import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
from tops.ps_models import k2a as model_data


JACOBIAN_EPS = 1e-7
UNSTABLE_TOL = 1e-6
NEAR_ZERO_TOL = 1e-6
EM_FREQUENCY_MIN_HZ = 0.1
EM_FREQUENCY_MAX_HZ = 3.0
WEAK_DAMPING_LIMIT = 0.05


def main():
    print("Imported model-data file:", model_data.__file__)
    print("No model parameters are modified by this script.")

    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    if not ps.power_flow_ready:
        raise RuntimeError("Kundur power flow did not converge.")

    if "HYGOV" not in ps.gov:
        raise RuntimeError(
            "Expected HYGOV in k2a.py; loaded governors: "
            f"{list(ps.gov.keys())}"
        )

    v0 = ps.solve_algebraic(0.0, ps.x0)
    dx0 = ps.state_derivatives(0.0, ps.x0, v0)
    max_dx0 = float(np.max(np.abs(dx0)))

    print("\nKundur modal-analysis initialization")
    print("-------------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Dynamic states:", ps.n_states)
    print("Maximum initial derivative:", f"{max_dx0:.6e}")

    print("\nLinearizing Kundur model...")
    start_time = time.perf_counter()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.eps = JACOBIAN_EPS
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    runtime = time.perf_counter() - start_time
    print(f"Linearization completed in {runtime:.2f} seconds.")

    eigenvalues = np.asarray(ps_lin.eigs)
    frequencies = np.abs(eigenvalues.imag) / (2.0 * np.pi)

    unstable_mask = eigenvalues.real > UNSTABLE_TOL
    near_zero_mask = np.abs(eigenvalues) <= NEAR_ZERO_TOL
    em_mask = (
        (eigenvalues.imag > 0.0)
        & (frequencies >= EM_FREQUENCY_MIN_HZ)
        & (frequencies <= EM_FREQUENCY_MAX_HZ)
    )

    em_indices = np.where(em_mask)[0]
    em_damping = np.array(
        [-eigenvalues[index].real / abs(eigenvalues[index]) for index in em_indices]
    )
    weak_count = int(np.sum(em_damping < WEAK_DAMPING_LIMIT))

    print("\nEigenvalue summary")
    print("------------------")
    print("Total eigenvalues:", len(eigenvalues))
    print("Unstable eigenvalues:", int(np.sum(unstable_mask)))
    print("Near-zero eigenvalues:", int(np.sum(near_zero_mask)))
    print("Electromechanical mode pairs:", len(em_indices))
    print("Weakly damped electromechanical mode pairs:", weak_count)

    unstable_indices = np.where(unstable_mask)[0]
    if len(unstable_indices):
        print("\nUnstable eigenvalues")
        print("--------------------")
        print("Index       Real          Imag       Frequency      Damping")
        for index in unstable_indices:
            value = eigenvalues[index]
            damping = -value.real / abs(value)
            print(
                f"{index:5d}  {value.real:12.6f}  {value.imag:12.6f}  "
                f"{frequencies[index]:10.4f} Hz  {100*damping:9.3f} %"
            )
    else:
        print(
            f"\nNo eigenvalues have a real part greater than "
            f"{UNSTABLE_TOL:.1e}."
        )

    if len(em_indices):
        order = np.argsort(em_damping)
        print("\nElectromechanical modes")
        print("-----------------------")
        print("Mode        Eigenvalue                 Frequency      Damping")
        for mode_number, position in enumerate(order, start=1):
            index = em_indices[position]
            value = eigenvalues[index]
            damping = em_damping[position]
            status = "UNSTABLE" if value.real > UNSTABLE_TOL else ""
            print(
                f"{mode_number:4d}  {value.real:11.5f} "
                f"{value.imag:+10.5f}j  {frequencies[index]:10.4f} Hz  "
                f"{100*damping:9.3f} %  {status}"
            )
    else:
        print("\nNo electromechanical modes were found in the selected band.")

    plot_eigenvalues(eigenvalues, frequencies, em_mask, unstable_mask)


def plot_eigenvalues(eigenvalues, frequencies, em_mask, unstable_mask):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Kundur two-area system – small-signal eigenvalue analysis\n"
        "HYGOV baseline with original GEN, SEXS and STAB1 parameters"
    )

    stable_mask = ~unstable_mask
    axes[0].scatter(
        eigenvalues[stable_mask].real,
        eigenvalues[stable_mask].imag,
        s=24,
        alpha=0.75,
        label="Stable or neutral",
    )
    if np.any(unstable_mask):
        axes[0].scatter(
            eigenvalues[unstable_mask].real,
            eigenvalues[unstable_mask].imag,
            marker="x",
            s=70,
            linewidth=2,
            color="tab:red",
            label="Unstable",
        )
    axes[0].axvline(0.0, color="black", linestyle="--", label="Stability boundary")
    axes[0].axhline(0.0, color="gray", linestyle=":", linewidth=1)
    axes[0].set_title("Complete eigenvalue spectrum")
    axes[0].set_xlabel(r"Real part, $\sigma$ (s$^{-1}$)")
    axes[0].set_ylabel(r"Imaginary part, $\omega$ (rad/s)")
    axes[0].legend()
    axes[0].grid(True)

    em_stable = em_mask & ~unstable_mask
    em_unstable = em_mask & unstable_mask
    axes[1].scatter(
        eigenvalues[em_stable].real,
        eigenvalues[em_stable].imag,
        s=50,
        label="Stable electromechanical modes",
    )
    if np.any(em_unstable):
        axes[1].scatter(
            eigenvalues[em_unstable].real,
            eigenvalues[em_unstable].imag,
            marker="x",
            s=80,
            linewidth=2,
            color="tab:red",
            label="Unstable electromechanical modes",
        )
    axes[1].axvline(0.0, color="black", linestyle="--", label="Stability boundary")
    axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title(
        "Electromechanical modes\n"
        f"{EM_FREQUENCY_MIN_HZ:.1f}–{EM_FREQUENCY_MAX_HZ:.1f} Hz"
    )
    axes[1].set_xlabel(r"Real part, $\sigma$ (s$^{-1}$)")
    axes[1].set_ylabel(r"Imaginary part, $\omega$ (rad/s)")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

