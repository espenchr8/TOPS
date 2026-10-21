"""Small-signal stability (eigenvalues) of N45 with the central AGC (aFRR).

Based on the TOPS modal-analysis example for N45. The only addition is the
AGC: it is added to the model and connected exactly as in n45_2025_agc.py.

All AGC settings (AGC_ENABLED, T_I, T_F, K_P, HVDC_SHARE, ...) are read from
n45_2025_agc.py, so this analysis always matches the simulated case.
Changes in ../common/agc.py are also included automatically.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import tops.ps_models.n45_2025 as model_data

# AGC model and the settings of the simulation file (same folder)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
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
        print(f"\nCase: {sim.AGC_TEXT}, K_i = {K_I:.1f} MW/(Hz s)")
    else:
        # AGC off: plain N45 model, exactly as in the TOPS example
        ps = dps.PowerSystemModel(model=model)
        ps.init_dyn_sim()
        print(f"\nCase: {sim.AGC_TEXT}")

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
    # PLOTS (TOPS)
    # =========================================================================

    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(np.atleast_1d(ax), mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

    plt.show()
