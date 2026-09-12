import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.n45_2025 as model_data


SCRIPT_VERSION = "verified-PSS-group-sweep-v1-20260912"
EPSILON = 1e-7
GAIN_VALUES = (5.0, 5.5, 6.0, 7.5, 10.0)
MODE_FREQUENCY_BAND_HZ = (0.50, 1.00)
UNSTABLE_TOLERANCE = 1e-6

G7100 = {"G7100-1", "G7100-2"}
G7000 = {
    "G7000-1",
    "G7000-2",
    "G7000-3",
    "G7000-4",
    "G7000-5",
    "G7000-6",
    "G7000-7",
}
SUPPORTING = {"G5230-1", "G5230-2", "G5560-1"}

GROUPS = {
    "G7100": G7100,
    "G7000": G7000,
    "G7100 + G7000": G7100 | G7000,
    "All selected": G7100 | G7000 | SUPPORTING,
}


def as_text(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    return str(value).strip()


def set_group_gain(model, selected_generators, gain):
    """Set selected STAB1 gains in a fresh in-memory data dictionary."""
    table = model["pss"]["STAB1"]
    header = list(table[0])
    generator_column = header.index("gen")
    gain_column = header.index("K")
    changed_generators = set()
    rows = []

    for original_row in table[1:]:
        row = list(original_row)
        generator = as_text(row[generator_column])
        if generator in selected_generators:
            row[gain_column] = float(gain)
            changed_generators.add(generator)
        rows.append(row)

    missing = selected_generators - changed_generators
    if missing:
        raise RuntimeError(
            "Missing STAB1 units for: " + ", ".join(sorted(missing))
        )

    model["pss"]["STAB1"] = [header, *rows]
    return model


def build_system(model):
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    return ps


def verify_group_gain(ps, selected_generators, expected_gain):
    """Verify the active Gain blocks for every selected generator."""
    pss = ps.pss["STAB1"]
    names = [as_text(value) for value in pss.par["gen"]]
    verified = set()

    for index, generator in enumerate(names):
        if generator not in selected_generators:
            continue

        data_gain = float(pss.par["K"][index])
        block_gain = float(pss.gain.par["K"][index])
        if not np.isclose(data_gain, expected_gain, atol=1e-12, rtol=0.0):
            raise RuntimeError(
                f"STAB1 data gain check failed for {generator}."
            )
        if not np.isclose(block_gain, expected_gain, atol=1e-12, rtol=0.0):
            raise RuntimeError(
                f"Active Gain-block check failed for {generator}."
            )
        verified.add(generator)

    if verified != selected_generators:
        raise RuntimeError("Not all selected PSS gain blocks were verified.")


def linearize(ps):
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.eps = EPSILON
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()
    return ps_lin


def candidate_modes(eigenvalues):
    frequency = eigenvalues.imag / (2.0 * np.pi)
    return np.where(
        (eigenvalues.imag > 0.0)
        & (frequency >= MODE_FREQUENCY_BAND_HZ[0])
        & (frequency <= MODE_FREQUENCY_BAND_HZ[1])
    )[0]


def find_baseline_mode(eigenvalues):
    candidates = candidate_modes(eigenvalues)
    if candidates.size == 0:
        raise RuntimeError("No baseline mode found between 0.5 and 1.0 Hz.")
    return candidates[np.argmax(eigenvalues[candidates].real)]


def find_matching_mode(eigenvalues, reference):
    candidates = candidate_modes(eigenvalues)
    if candidates.size == 0:
        raise RuntimeError("No trial mode found between 0.5 and 1.0 Hz.")
    return candidates[np.argmin(np.abs(eigenvalues[candidates] - reference))]


def damping_percent(eigenvalue):
    return -eigenvalue.real / abs(eigenvalue) * 100.0


def analyze_case(group_name, selected_generators, gain, reference_mode):
    model = set_group_gain(
        model_data.load(),
        selected_generators,
        gain,
    )
    ps = build_system(model)
    verify_group_gain(ps, selected_generators, gain)

    dx_initial = ps.state_derivatives(0.0, ps.x_0, ps.v_0)
    maximum_initial_derivative = float(np.max(np.abs(dx_initial)))

    start = time.time()
    ps_lin = linearize(ps)
    runtime = time.time() - start

    mode_index = find_matching_mode(ps_lin.eigs, reference_mode)
    mode = ps_lin.eigs[mode_index]
    frequency = mode.imag / (2.0 * np.pi)
    damping = damping_percent(mode)

    unstable = ps_lin.eigs.real > UNSTABLE_TOLERANCE
    unstable_count = int(np.count_nonzero(unstable))
    rightmost_real = float(np.max(ps_lin.eigs.real))

    return {
        "group": group_name,
        "gain": gain,
        "mode": mode,
        "frequency": frequency,
        "damping": damping,
        "unstable_count": unstable_count,
        "rightmost_real": rightmost_real,
        "initial_derivative": maximum_initial_derivative,
        "runtime": runtime,
    }


if __name__ == "__main__":
    print("Script version:", SCRIPT_VERSION)
    print("Imported model-data file:", model_data.__file__)
    print("No installed TOPS source files are modified.")

    print("\nBaseline N45 linearization")
    print("--------------------------")
    baseline_ps = build_system(model_data.load())
    baseline_start = time.time()
    baseline_linearization = linearize(baseline_ps)
    baseline_runtime = time.time() - baseline_start
    baseline_index = find_baseline_mode(baseline_linearization.eigs)
    baseline_mode = baseline_linearization.eigs[baseline_index]
    baseline_frequency = baseline_mode.imag / (2.0 * np.pi)
    baseline_damping = damping_percent(baseline_mode)
    baseline_unstable_count = int(
        np.count_nonzero(
            baseline_linearization.eigs.real > UNSTABLE_TOLERANCE
        )
    )

    print(
        "Critical eigenvalue:",
        f"{baseline_mode.real:+.10f} {baseline_mode.imag:+.10f}j",
    )
    print("Frequency:", f"{baseline_frequency:.6f} Hz")
    print("Damping:", f"{baseline_damping:.4f} %")
    print("Unstable eigenvalues:", baseline_unstable_count)
    print("Runtime:", f"{baseline_runtime:.2f} s")

    results = []
    total_start = time.time()

    print("\nPSS group sweep")
    print("---------------")

    for group_name, selected_generators in GROUPS.items():
        print(
            f"\nGroup: {group_name} "
            f"({len(selected_generators)} STAB1 units)"
        )

        for gain in GAIN_VALUES:
            result = analyze_case(
                group_name,
                selected_generators,
                gain,
                baseline_mode,
            )
            results.append(result)

            status = "STABLE" if result["unstable_count"] == 0 else "UNSTABLE"
            print(
                f"K={gain:>4.1f}  "
                f"lambda={result['mode'].real:+.6f} "
                f"{result['mode'].imag:+.6f}j  "
                f"f={result['frequency']:.4f} Hz  "
                f"zeta={result['damping']:+.3f}%  "
                f"unstable={result['unstable_count']:>2}  "
                f"{status:<8}  "
                f"dx0={result['initial_derivative']:.2e}  "
                f"{result['runtime']:.2f} s"
            )

    print("\nSummary")
    print("-------")
    print(
        "Group                 K     Real part    Frequency   "
        "Damping    Unstable  Rightmost real"
    )

    for result in results:
        print(
            f"{result['group']:<21} "
            f"{result['gain']:>4.1f}  "
            f"{result['mode'].real:>+11.6f}  "
            f"{result['frequency']:>8.4f} Hz  "
            f"{result['damping']:>+8.3f}%  "
            f"{result['unstable_count']:>8}  "
            f"{result['rightmost_real']:>+14.6f}"
        )

    stable_results = [
        result for result in results if result["unstable_count"] == 0
    ]

    print("\nCandidate selection")
    print("-------------------")
    if stable_results:
        stable_results.sort(
            key=lambda result: (
                result["gain"],
                -result["damping"],
            )
        )
        candidate = stable_results[0]
        print("Lowest-gain fully stable case:")
        print("Group:", candidate["group"])
        print("Gain:", candidate["gain"])
        print("Critical-mode damping:", f"{candidate['damping']:.4f} %")
    else:
        print("No tested case made all eigenvalues stable.")

    print("Total sweep runtime:", f"{time.time() - total_start:.2f} s")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for group_name in GROUPS:
        group_results = [
            result for result in results if result["group"] == group_name
        ]
        gains = [result["gain"] for result in group_results]
        real_parts = [result["mode"].real for result in group_results]
        damping = [result["damping"] for result in group_results]

        axes[0].plot(gains, real_parts, marker="o", label=group_name)
        axes[1].plot(gains, damping, marker="o", label=group_name)

    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].set_ylabel("Critical-mode real part (1/s)")
    axes[0].set_title("N45 verified STAB1 group-gain sweep")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("Selected STAB1 gain K")
    axes[1].set_ylabel("Critical-mode damping (%)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
