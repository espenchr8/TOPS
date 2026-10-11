"""Nonlinear TOPS simulation with AGC, and the matching linear simulation."""

import numpy as np
from scipy.linalg import expm
import tops.solvers as dps_sol

from linear_kundur import AREAS, LinearModel, build_case

DT = 0.005


def simulate_nonlinear(transfer=0.0, outage=None, pss=True, k_i=0.0,
                       t_end=60.0, t_event=1.0, step=1.0, bias_factor=1.0):
    """TOPS RMS simulation. AGC integrators are stepped outside the solver.

    The AGC acts on the TGOV1 bias (governor reference) of each unit, which
    is constant during each 5 ms step. AGC is slow, so this is accurate.
    """
    ps = build_case(transfer, outage, pss)
    lm = LinearModel(ps)  # Reused for indices, beta and the B9 admittance.
    gen = ps.gen["GEN"]
    H = gen.par["H"].astype(float)
    beta = lm.area_frequency_response()
    tie0 = lm.y0[4]
    z = np.zeros(2)
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic,
                                   0, ps.x_0.copy(), t_end, max_step=DT)
    out = []
    n_steps = round(t_end / DT)
    for k in range(n_steps + 1):
        d = step if sol.t >= t_event - 1e-9 else 0.0
        b9 = lm.b9
        if ps.y_bus_red_mod[b9, b9] != d * lm.y_step:
            ps.y_bus_red_mod[b9, b9] = d * lm.y_step
            sol.v[:] = ps.solve_algebraic(sol.t, sol.y)
        speed = gen.speed(sol.y, sol.v)
        tie = lm.tie_flow(sol.y, sol.v) - tie0
        ace = []
        for j, (a, gs) in enumerate(AREAS.items()):
            idx = [lm.names.index(g) for g in gs]
            df = np.average(speed[idx], weights=H[idx])
            ace.append((tie if a == 1 else -tie) + bias_factor * beta[a] * df)
        ace = np.array(ace)
        u = np.zeros(4)
        for j, gs in enumerate(AREAS.values()):
            for g in gs:
                u[lm.names.index(g)] = 0.5 * z[j]
        bias = lm.bias0.copy()
        bias[lm.gov_of_gen] += lm.R[lm.gov_of_gen] * u
        lm.gov.int_par["bias"][:] = bias
        out.append(np.r_[sol.t, speed, tie, ace, z])
        if k < n_steps:
            sol.step()
            z += -k_i * ace * DT
    return np.array(out), lm


def simulate_linear(lm, k_i=0.0, t_end=60.0, t_event=1.0, step=1.0,
                    bias_factor=1.0):
    """Exact discretization of the linear closed loop for a load step."""
    A, E, C, _ = lm.closed_loop(k_i, bias_factor)
    n = A.shape[0]
    M = expm(np.block([[A, E], [np.zeros((1, n + 1))]]) * DT)
    Ad, Ed = M[:n, :n], M[:n, n:]
    t = np.arange(0, t_end + DT / 2, DT)
    x = np.zeros(n)
    y = []
    for tk in t:
        d = step if tk >= t_event - 1e-9 else 0.0
        yk = C @ x
        yk[4] += lm.D_tie * d
        y.append(yk)
        x = Ad @ x + Ed[:, 0] * d
    return t, np.array(y)
