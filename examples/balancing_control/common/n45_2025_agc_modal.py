"""Small-signal stability (eigenvalues) of N45 with the central AGC (aFRR).

Based on the TOPS modal-analysis example for N45. The only addition is the
AGC: it is added to the model and connected exactly as in n45_2025_agc.py.

All AGC settings (AGC_ENABLED, T_I, T_F, K_P, HVDC_SHARE, ...) are read from
n45_2025_agc.py, so this analysis always matches the simulated case.
Changes in common/agc.py are also included automatically.
Place this file in balancing_control/common/ (or n45/); both work.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.ps_models.n45_2025 as model_data

# AGC model (common/agc.py) and the settings of the simulation file
# (n45/n45_2025_agc.py). Works whether this file is placed in common/ or n45/.
BALANCING_CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BALANCING_CONTROL / "common"))
sys.path.insert(0, str(BALANCING_CONTROL / "n45"))
from agc import add_agc_to_model, connect_agc, regulating_strength_mw_per_hz, user_model_library
import n45_2025_agc as sim


if __name__ == '__main__':

    # =========================================================================
    # MODEL WITH AGC (same as in n45_2025_agc.py)
    # =========================================================================

    model = model_data.load()

    if sim.AGC_ENABLED:
        B = sum(regulating_strength_mw_per_hz(model).values())
        K_I = B / sim.T_I
        add_agc_to_model(model, T_f=sim.T_F, K_p=sim.K_P, K_i=K_I)

        ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_model_library())
        ps.init_dyn_sim()

        agc = ps.agc["AGC"]
        connect_agc(
            agc, ps,
            hydro_units=sim.HYDRO_AGC_UNITS,
            thermal_units=sim.THERMAL_AGC_UNITS,
            hvdc_share=sim.HVDC_SHARE,
            hvdc_link_split=sim.HVDC_LINK_SPLIT,
        )
        print(f"\nCase: T_i = {sim.T_I:.0f} s, HVDC share = {sim.HVDC_SHARE:.0%}, K_i = {K_I:.1f} MW/(Hz s)")
    else:
        # AGC off: plain N45 model, exactly as in the TOPS example
        ps = dps.PowerSystemModel(model=model)
        ps.init_dyn_sim()
        print("\nCase: AGC off (FCR only)")

    # =========================================================================
    # LINEARIZATION AND EIGENVALUES (TOPS)
    # =========================================================================

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    eigs, damp, freq = ps_lin.eigs, ps_lin.damping, ps_lin.freq

    # Least damped oscillating modes
    print("\nLeast damped oscillating modes:")
    osc = freq > 0.005
    idx = np.where(osc)[0][np.argsort(damp[osc])][:10]
    for i in idx:
        print(f'{eigs[i].real:8.3f} {eigs[i].imag:8.3f}  f={freq[i]:.3f} Hz  zeta={100*damp[i]:.1f} %')
    print('Re > 1e-6:', np.sum(eigs.real > 1e-6), '| tilnærmet null:', np.sum(abs(eigs) < 1e-6))

    # AGC mode: the mode where the AGC integrator state participates most
    if sim.AGC_ENABLED:
        k_int = int(agc.pi.integrator.state_idx_global["x_i"][0])
        participation = np.abs(ps_lin.rev[k_int, :] * ps_lin.lev[:, k_int])
        k = int(np.argmax(participation))
        print(
            f"\nAGC mode: {eigs[k].real:.4f} {eigs[k].imag:+.4f}j  "
            f"tau = {-1 / eigs[k].real:.0f} s  f = {abs(freq[k]) * 1000:.1f} mHz  "
            f"zeta = {100 * damp[k]:.0f} %"
        )

    # =========================================================================
    # PLOT 1: EIGENVALUES (two zoomed views)
    # =========================================================================

    # Left:  electromechanical modes (0.1-2 Hz), with 5 % and 10 % damping lines.
    # Right: slow modes near the origin (governors/FCR and the AGC mode).
    # Frequency is shown in Hz (imaginary part / 2 pi), upper half only.
    f_hz = eigs.imag / (2 * np.pi)
    upper = f_hz >= 0
    title = (
        f"T_i = {sim.T_I:.0f} s, HVDC share = {sim.HVDC_SHARE:.0%}"
        if sim.AGC_ENABLED else "AGC off (FCR only)"
    )

    fig, (ax_em, ax_slow) = plt.subplots(1, 2, figsize=(13, 5.5))
    # Re > 1e-3 ignores the rotor-angle reference mode, which is zero in theory
    # but can come out slightly positive (~3e-4) in the numerical linearization.
    fig.suptitle(f"Eigenvalues of N45 – {title}\n"
                 f"Unstable modes (Re > 0.001): {int(np.sum(eigs.real > 1e-3))}")

    ax_em.scatter(eigs.real[upper], f_hz[upper], s=14, color="tab:blue")
    f_line = np.linspace(0, 2.0, 50)
    for zeta, ls in ((0.05, "--"), (0.10, ":")):
        sigma = -zeta * 2 * np.pi * f_line / np.sqrt(1 - zeta**2)
        ax_em.plot(sigma, f_line, color="gray", ls=ls, lw=1, label=f"{zeta:.0%} damping")
    em = np.where((freq > 0.1) & (freq < 2.0))[0]
    for i in em[np.argsort(damp[em])][:2]:   # mark the two least damped EM modes
        ax_em.annotate(f"{freq[i]:.2f} Hz, {100 * damp[i]:.1f} %", (eigs[i].real, f_hz[i]),
                       xytext=(10, -4), textcoords="offset points", color="tab:red")
        ax_em.plot(eigs[i].real, f_hz[i], "o", mfc="none", mec="tab:red", ms=10)
    ax_em.axvline(0, color="black", lw=0.8)
    ax_em.set_xlim(-2.5, 0.2)
    ax_em.set_ylim(0.1, 2.0)
    ax_em.set_xlabel("Real part (1/s)")
    ax_em.set_ylabel("Frequency (Hz)")
    ax_em.set_title("Electromechanical modes")
    ax_em.legend(loc="upper left")
    ax_em.grid(True, alpha=0.4)

    ax_slow.scatter(eigs.real[upper], f_hz[upper] * 1000, s=14, color="tab:blue")
    f_line = np.linspace(0, 0.05, 50)
    for zeta, ls in ((0.05, "--"), (0.10, ":")):
        sigma = -zeta * 2 * np.pi * f_line / np.sqrt(1 - zeta**2)
        ax_slow.plot(sigma, f_line * 1000, color="gray", ls=ls, lw=1)

    # Slowest real mode apart from the AGC mode and the zero eigenvalues = FCR mode
    other = np.where((np.abs(eigs.imag) < 1e-6) & (eigs.real < -1e-3))[0]
    if sim.AGC_ENABLED:
        other = other[other != k]
    k_fcr = other[np.argmax(eigs.real[other])]
    ax_slow.plot(eigs[k_fcr].real, 0, "s", ms=10, mfc="none", mec="tab:blue",
                 label=f"Slowest governor mode (FCR), tau = {-1 / eigs[k_fcr].real:.0f} s")
    if sim.AGC_ENABLED:
        agc_label = (f"AGC mode, {abs(f_hz[k]) * 1000:.1f} mHz, damping {100 * damp[k]:.0f} %"
                     if abs(f_hz[k]) > 1e-4 else f"AGC mode, tau = {-1 / eigs[k].real:.0f} s")
        ax_slow.plot(eigs[k].real, abs(f_hz[k]) * 1000, "*", ms=16, color="tab:green", label=agc_label)
    ax_slow.legend(loc="upper left")
    ax_slow.axvline(0, color="black", lw=0.8)
    ax_slow.set_xlim(-0.1, 0.01)
    ax_slow.set_ylim(-1, 50)
    ax_slow.set_xlabel("Real part (1/s)")
    ax_slow.set_ylabel("Frequency (mHz)")
    ax_slow.set_title("Slow modes near the origin (FCR and AGC)")
    ax_slow.grid(True, alpha=0.4)
    fig.tight_layout()

    # =========================================================================
    # PLOT 2: MODE SHAPES OF THE THREE LEAST DAMPED ELECTROMECHANICAL MODES
    # =========================================================================

    # Generator speed in each mode, as arrows coloured by country. Arrows that
    # point in opposite directions mean that those generators swing against
    # each other. Small arrows (< 10 % of the largest) are left out.
    mode_idx = ps_lin.get_mode_idx(['em', 'non_conj'], damp_threshold=0.3)[:3]
    mode_shape = ps_lin.rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    regions = {"Norway": ({11, 12, 13, 14, 15}, "tab:red"),
               "Sweden": ({21, 22, 23, 24}, "tab:blue"),
               "Finland": ({31}, "tab:green")}
    bus_header = list(model["buses"][0])
    bus_area = {str(r[bus_header.index("name")]): int(r[bus_header.index("Area")]) for r in model["buses"][1:]}
    gen_area = np.array([bus_area[str(b)] for b in ps.gen['GEN'].par["bus"]])

    fig, axes = plt.subplots(1, len(mode_idx), figsize=(4.5 * len(mode_idx), 4.8),
                             subplot_kw={'projection': 'polar'})
    fig.suptitle("Mode shapes (generator speed) of the least damped electromechanical modes")
    for ax_, i, ms in zip(np.atleast_1d(axes), mode_idx, mode_shape.T):
        ms = ms / np.max(np.abs(ms))
        for region, (codes, color) in regions.items():
            for m in ms[np.isin(gen_area, list(codes)) & (np.abs(ms) > 0.1)]:
                ax_.annotate("", xy=(np.angle(m), np.abs(m)), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
            ax_.plot([], [], color=color, label=region)
        ax_.set_ylim(0, 1.05)
        ax_.set_yticklabels([])
        ax_.set_xticklabels([])
        ax_.set_title(f"{freq[i]:.2f} Hz, damping {100 * damp[i]:.1f} %", pad=14)
    np.atleast_1d(axes)[-1].legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()

    plt.show()
