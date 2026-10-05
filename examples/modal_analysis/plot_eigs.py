import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import tops.ps_models.n45_2025 as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    eigs, damp, freq = ps_lin.eigs, ps_lin.damping, ps_lin.freq
    osc = freq > 0.005
    idx = np.where(osc)[0][np.argsort(damp[osc])][:10]
    for i in idx:
        print(f'{eigs[i].real:8.3f} {eigs[i].imag:8.3f}  f={freq[i]:.3f} Hz  zeta={100*damp[i]:.1f} %')
    print('Re > 1e-6:', np.sum(eigs.real > 1e-6), '| tilnærmet null:', np.sum(abs(eigs) < 1e-6))

    # Plot eigenvalues
    dps_plt.plot_eigs(ps_lin.eigs)

    # Get mode shape for electromechanical modes
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=0.3)
    rev = ps_lin.rev
    mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]

    # Plot mode shape
    fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
    for ax_, ms in zip(ax, mode_shape.T):
        dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

    plt.show()
