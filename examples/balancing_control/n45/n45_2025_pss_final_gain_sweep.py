import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.n45_2025 as model_data


SCRIPT_VERSION = "verified-final-PSS-sweep-v1-20260912"
EPSILON = 1e-7
GAIN_VALUES = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0)
TRACKED_MODE_BAND_HZ = (0.50, 1.00)
ELECTROMECHANICAL_BAND_HZ = (0.10, 3.00)
UNSTABLE_TOLERANCE = 1e-6

SELECTED_GENERATORS = {
    "G7100-1",
    "G7100-2",
    "G7000-1",
    "G7000-2",
    "G7000-3",
    "G7000-4",
    "G7000-5",
    "G7000-6",
    "G7000-7",
    "G5230-1",
    "G5230-2",
    "G5560-1",
}


def as_text(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    return str(value).strip()


def set_selected_gain(model, gain):
    table = model["pss"]["STAB1"]
    header = list(table[0])
    generator_column = header.index("gen")
    gain_column = header.index("K")
    changed = set()
    rows = []

    for original_row in table[1:]:
        row = list(original_row)
        generator = as_text(row[generator_column])
        if generator in SELECTED_GENERATORS:
            row[gain_column] = float(gain)
            changed.add(generator)
        rows.append(row)

    missing = SELECTED_GENERATORS - changed
    if missing:
        raise RuntimeError(
            "Missing STAB1 units for: " + ", ".join(sorted(missing))
        )

    model["pss"]["STAB1"] = [header, *rows]
    return model


def build_system(gain):
    model = set_selected_gain(model_data.load(), gain)
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    return ps


def verify_active_gains(ps, expected_gain):
    pss = ps.pss["STAB1"]
    verified = set()

    for index, value in enumerate(pss.par["gen"]):
        generator = as_text(value)
        if generator not in SELECTED_GENERATORS:
            continue

        data_gain = float(pss.par["K"][index])
        block_gain = float(pss.gain.par["K"][index])
        if not np.isclose(data_gain, expected_gain, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Data gain check failed for {generator}.")
        if not np.isclose(block_gain, expected_gain, atol=1e-12, rtol=0.0):
            raise RuntimeError(f"Gain-block check failed for {generator}.")
        verified.add(generator)

    if verified != SELECTED_GENERATORS:
        raise RuntimeError("Not all selected STAB1 blocks were verified.")


def linearize(ps):
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.eps = EPSILON
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    return ps_lin


def indices_in_band(eigenvalues, frequency_band):
    frequency = eigenvalues.imag / (2.0 * np.pi)
    return np.where(
        (eigenvalues.imag > 0.0)
        & (frequency >= frequency_band[0])
        & (frequency <= frequency_band[1])
    )[0]


def find_reference_mode(eigenvalues):
    candidates = indices_in_band(eigenvalues, TRACKED_MODE_BAND_HZ)
    if candidates.size == 0:
        raise RuntimeError("No reference mode found between 0.5 and 1.0 Hz.")
    return candidates[np.argmax(eigenvalues[candidates].real)]


def find_matching_mode(eigenvalues, reference):
    candidates = indices_in_band(eigenvalues, TRACKED_MODE_BAND_HZ)
    if candidates.size == 0:
        raise RuntimeError("No tracked mode found between 0.5 and 1.0 Hz.")
    return candidates[np.argmin(np.abs(eigenvalues[candidates] - reference))]


def damping_percent(eigenvalues):
    return -eigenvalues.real / np.abs(eigenvalues) * 100.0


if __name__ == "__main__":
    print("Script version:", SCRIPT_VERSION)
    print("Imported model-data file:", model_data.__file__)
    print("No installed TOPS source files are modified.")
    print("Selected STAB1 units:", len(SELECTED_GENERATORS))

    reference_ps = build_system(5.0)
    verify_active_gains(reference_ps, 5.0)
    reference_linearization = linearize(reference_ps)
    reference_index = find_reference_mode(reference_linearization.eigs)
    reference_mode = reference_linearization.eigs[reference_index]

    print("\nReference mode")
    print("--------------")
    print(
        "Eigenvalue:",
        f"{reference_mode.real:+.10f} {reference_mode.imag:+.10f}j",
    )
    print(
        "Frequency:",
        f"{reference_mode.imag / (2.0 * np.pi):.6f} Hz",
    )
    print(
        "Damping:",
        f"{float(damping_percent(np.array([reference_mode]))[0]):.4f} %",
    )

    results = []
    total_start = time.time()

    print("\nFinal selected-group gain sweep")
    print("-------------------------------")

    for gain in GAIN_VALUES:
        ps = build_system(gain)
        verify_active_gains(ps, gain)

        dx_initial = ps.state_derivatives(0.0, ps.x_0, ps.v_0)
        maximum_initial_derivative = float(np.max(np.abs(dx_initial)))

        start = time.time()
        ps_lin = linearize(ps)
        runtime = time.time() - start
        eigenvalues = ps_lin.eigs

        tracked_index = find_matching_mode(eigenvalues, reference_mode)
        tracked_mode = eigenvalues[tracked_index]
        tracked_frequency = tracked_mode.imag / (2.0 * np.pi)
        tracked_damping = float(
            damping_percent(np.array([tracked_mode]))[0]
        )

        em_indices = indices_in_band(
            eigenvalues,
            ELECTROMECHANICAL_BAND_HZ,
        )
        em_eigenvalues = eigenvalues[em_indices]
        em_damping = damping_percent(em_eigenvalues)
        weakest_position = int(np.argmin(em_damping))
        weakest_mode = em_eigenvalues[weakest_position]
        weakest_damping = float(em_damping[weakest_position])
        weakest_frequency = weakest_mode.imag / (2.0 * np.pi)

        unstable_physical = eigenvalues.real > UNSTABLE_TOLERANCE
        unstable_count = int(np.count_nonzero(unstable_physical))

        results.append(
            {
                "gain": gain,
                "tracked_mode": tracked_mode,
                "tracked_frequency": tracked_frequency,
                "tracked_damping": tracked_damping,
                "weakest_mode": weakest_mode,
                "weakest_frequency": weakest_frequency,
                "weakest_damping": weakest_damping,
                "unstable_count": unstable_count,
                "initial_derivative": maximum_initial_derivative,
                "runtime": runtime,
            }
        )

        status = "STABLE" if unstable_count == 0 else "UNSTABLE"
        print(
            f"K={gain:>4.1f}  "
            f"tracked zeta={tracked_damping:+6.3f}%  "
            f"weakest EM={weakest_damping:+6.3f}% "
            f"at {weakest_frequency:.4f} Hz  "
            f"unstable={unstable_count:>2}  "
            f"{status:<8}  "
            f"dx0={maximum_initial_derivative:.2e}  "
            f"{runtime:.2f} s"
        )

    print("\nDetailed summary")
    print("----------------")
    print(
        "K     Tracked real   Tracked freq   Tracked damping  "
        "Weakest EM damping  Weakest EM freq  Unstable"
    )

    for result in results:
        print(
            f"{result['gain']:>4.1f}  "
            f"{result['tracked_mode'].real:>+12.6f}  "
            f"{result['tracked_frequency']:>10.4f} Hz  "
            f"{result['tracked_damping']:>+13.3f}%  "
            f"{result['weakest_damping']:>+16.3f}%  "
            f"{result['weakest_frequency']:>11.4f} Hz  "
            f"{result['unstable_count']:>8}"
        )

    acceptable = [
        result
        for result in results
        if result["unstable_count"] == 0
        and result["weakest_damping"] >= 3.0
    ]

    print("\nCandidate selection")
    print("-------------------")
    if acceptable:
        candidate = min(acceptable, key=lambda result: result["gain"])
        print("Lowest tested gain with at least 3% EM damping:")
        print("Selected-group gain:", candidate["gain"])
        print(
            "Tracked-mode damping:",
            f"{candidate['tracked_damping']:.3f} %",
        )
        print(
            "Weakest EM damping:",
            f"{candidate['weakest_damping']:.3f} %",
        )
    else:
        print("No tested gain achieved at least 3% damping for every EM mode.")
        print("Do not choose a final gain from this sweep automatically.")

    print("Total runtime:", f"{time.time() - total_start:.2f} s")

    gains = [result["gain"] for result in results]
    tracked_damping = [
        result["tracked_damping"] for result in results
    ]
    weakest_damping = [
        result["weakest_damping"] for result in results
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        gains,
        tracked_damping,
        marker="o",
        label="Tracked 0.725 Hz mode",
    )
    ax.plot(
        gains,
        weakest_damping,
        marker="s",
        label="Weakest electromechanical mode",
    )
    ax.axhline(0.0, color="black", linestyle="--", label="Stability limit")
    ax.axhline(3.0, color="tab:green", linestyle=":", label="3% target")
    ax.axhline(5.0, color="tab:gray", linestyle=":", label="5% target")
    ax.set_xlabel("Selected STAB1 gain K")
    ax.set_ylabel("Damping ratio (%)")
    ax.set_title(
        "Nordic 45 (2025) – final selected-group PSS gain sweep"
    )
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
