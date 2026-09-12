import time

import matplotlib.pyplot as plt
import numpy as np

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.n45_2025 as model_data


SCRIPT_VERSION = "verified-fresh-model-v4-20260912"
EPSILON = 1e-7
GAIN_CHANGE = 0.10
FREQUENCY_BAND_HZ = (0.50, 1.00)
ZERO_EFFECT_TOLERANCE = 1e-10


def as_text(value):
    if isinstance(value, bytes):
        return value.decode().strip()
    return str(value).strip()


def change_one_pss_gain(model, pss_name, new_gain):
    """Change one STAB1 row in a fresh in-memory model dictionary."""
    table = model["pss"]["STAB1"]
    header = list(table[0])
    name_column = header.index("name")
    gain_column = header.index("K")
    changed_count = 0
    rows = []

    for original_row in table[1:]:
        row = list(original_row)
        if as_text(row[name_column]) == pss_name:
            row[gain_column] = float(new_gain)
            changed_count += 1
        rows.append(row)

    if changed_count != 1:
        raise RuntimeError(
            f"Expected one STAB1 named {pss_name!r}; changed {changed_count}."
        )

    model["pss"]["STAB1"] = [header, *rows]
    return model


def build_system(model):
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    return ps


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
        & (frequency >= FREQUENCY_BAND_HZ[0])
        & (frequency <= FREQUENCY_BAND_HZ[1])
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


def find_pss_index(pss, pss_name):
    names = np.array([as_text(value) for value in pss.par["name"]])
    matches = np.where(names == pss_name)[0]
    if matches.size != 1:
        raise RuntimeError(
            f"Expected one initialized STAB1 named {pss_name!r}."
        )
    return int(matches[0])


def verify_active_gain(ps, pss_name, expected_gain):
    """Check both STAB1 data and the Gain block used in its output."""
    pss = ps.pss["STAB1"]
    index = find_pss_index(pss, pss_name)
    model_gain = float(pss.par["K"][index])
    block_gain = float(pss.gain.par["K"][index])

    if not np.isclose(model_gain, expected_gain, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"STAB1 gain check failed for {pss_name}: "
            f"expected {expected_gain}, got {model_gain}."
        )
    if not np.isclose(block_gain, expected_gain, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Active Gain-block check failed for {pss_name}: "
            f"expected {expected_gain}, got {block_gain}."
        )
    return block_gain


if __name__ == "__main__":
    print("Script version:", SCRIPT_VERSION)
    print("Imported model-data file:", model_data.__file__)
    print("No installed TOPS source files are modified.")
    print(
        "Each trial builds a fresh N45 system with one STAB1 gain "
        f"increased by {100.0 * GAIN_CHANGE:.1f}%."
    )

    baseline_ps = build_system(model_data.load())
    baseline_pss = baseline_ps.pss["STAB1"]
    pss_names = [as_text(value) for value in baseline_pss.par["name"]]
    generator_names = [
        as_text(value) for value in baseline_pss.par["gen"]
    ]
    original_gains = np.asarray(baseline_pss.par["K"], dtype=float).copy()

    print("\nN45 PSS-gain sensitivity analysis")
    print("---------------------------------")
    print("Power flow ready:", baseline_ps.power_flow_ready)
    print("Dynamic states:", baseline_ps.n_states)
    print("STAB1 units:", baseline_pss.n_units)

    dx_initial = baseline_ps.state_derivatives(
        0.0, baseline_ps.x_0, baseline_ps.v_0
    )
    print(
        "Maximum initial derivative:",
        f"{np.max(np.abs(dx_initial)):.10e}",
    )

    print(f"\nBaseline linearization with eps={EPSILON:.0e} ...")
    start = time.time()
    baseline_linearization = linearize(baseline_ps)
    baseline_index = find_baseline_mode(baseline_linearization.eigs)
    baseline_eigenvalue = baseline_linearization.eigs[baseline_index]
    baseline_damping = damping_percent(baseline_eigenvalue)
    baseline_frequency = baseline_eigenvalue.imag / (2.0 * np.pi)

    print(f"Baseline runtime: {time.time() - start:.2f} s")
    print(
        "Critical eigenvalue:",
        f"{baseline_eigenvalue.real:+.10f} "
        f"{baseline_eigenvalue.imag:+.10f}j",
    )
    print("Critical frequency:", f"{baseline_frequency:.6f} Hz")
    print("Critical damping:", f"{baseline_damping:.4f} %")

    print("\nTesting independently rebuilt N45 systems")
    print("-----------------------------------------")

    results = []
    total_start = time.time()

    for index, pss_name in enumerate(pss_names):
        original_gain = float(original_gains[index])
        if original_gain == 0.0:
            print(
                f"[{index + 1:>2}/{len(pss_names)}] "
                f"{generator_names[index]:<14} skipped: K=0"
            )
            continue

        trial_gain = original_gain * (1.0 + GAIN_CHANGE)
        gain_step = trial_gain - original_gain

        trial_model = change_one_pss_gain(
            model_data.load(), pss_name, trial_gain
        )
        trial_ps = build_system(trial_model)
        active_gain = verify_active_gain(trial_ps, pss_name, trial_gain)

        trial_start = time.time()
        trial_linearization = linearize(trial_ps)
        trial_index = find_matching_mode(
            trial_linearization.eigs, baseline_eigenvalue
        )
        trial_eigenvalue = trial_linearization.eigs[trial_index]
        trial_damping = damping_percent(trial_eigenvalue)

        delta_real = trial_eigenvalue.real - baseline_eigenvalue.real
        delta_damping = trial_damping - baseline_damping
        sensitivity = delta_real / gain_step

        if delta_real < -ZERO_EFFECT_TOLERANCE:
            effect = "IMPROVES"
        elif delta_real > ZERO_EFFECT_TOLERANCE:
            effect = "WORSENS"
        else:
            effect = "NEGLIGIBLE"

        results.append(
            {
                "pss": pss_name,
                "generator": generator_names[index],
                "original_gain": original_gain,
                "trial_gain": trial_gain,
                "delta_real": delta_real,
                "sensitivity": sensitivity,
                "damping": trial_damping,
                "delta_damping": delta_damping,
                "effect": effect,
            }
        )

        print(
            f"[{index + 1:>2}/{len(pss_names)}] "
            f"{generator_names[index]:<14} "
            f"K(block)={active_gain:.4f}  "
            f"Delta real={delta_real:+.6e}  {effect:<10} "
            f"{time.time() - trial_start:.2f} s"
        )

    results.sort(key=lambda item: item["delta_real"])

    print("\nPSS gain-sensitivity ranking")
    print("-" * 130)
    print(
        "Rank  PSS             Generator       K original  K trial   "
        "Delta real       dReal/dK       Damping    Delta damping  Effect"
    )
    print("-" * 130)

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>4}  {result['pss']:<14}  "
            f"{result['generator']:<14}  "
            f"{result['original_gain']:>10.4f}  "
            f"{result['trial_gain']:>7.4f}  "
            f"{result['delta_real']:>+13.6e}  "
            f"{result['sensitivity']:>+13.6e}  "
            f"{result['damping']:>9.4f}%  "
            f"{result['delta_damping']:>+12.5f}%  "
            f"{result['effect']}"
        )

    print("\nEvery trial used a fresh in-memory model.")
    print("The original n45_2025.py file was not changed.")
    print(f"Total sensitivity runtime: {time.time() - total_start:.2f} s")

    generators = [result["generator"] for result in results]
    changes = np.array([result["delta_real"] for result in results])
    colors = [
        "tab:green" if value < -ZERO_EFFECT_TOLERANCE
        else "tab:red" if value > ZERO_EFFECT_TOLERANCE
        else "tab:gray"
        for value in changes
    ]

    fig, ax = plt.subplots(figsize=(11, max(8.0, 0.25 * len(results))))
    positions = np.arange(len(results))
    ax.barh(positions, changes, color=colors)
    ax.set_yticks(positions, labels=generators)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linestyle="--")
    ax.set_xlabel("Change in critical-mode real part (1/s)")
    ax.set_title(
        "Nordic 45 (2025) – verified STAB1 gain sensitivity\n"
        "Green improves damping; red reduces damping"
    )
    ax.grid(True, axis="x")
    plt.tight_layout()
    plt.show()
