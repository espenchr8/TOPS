"""Part 2: linear model of the TOPS Kundur system, with AGC added.

The TOPS model is linearized numerically around a power-flow operating point:

    dx/dt = A x + B u + E d,      y = C x

x  all TOPS states (machines, governors, AVRs, PSSs)
u  governor reference of each generator, pu of its rating
   (TOPS TGOV1: droop input = -speed + bias, so bias = R * u)
d  load step at B9, 1.0 = 100 MW at the pre-event voltage (fixed admittance)
y  speed of each generator and the corridor flow B7->B8 (pu on 900 MVA)

AGC adds one integrator per area:  dz_a/dt = -K_I * ACE_a,
ACE_a = dP_tie,a + B_a * df_a,  and z_a is shared equally by the two units.
"""

import copy

import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data

AREAS = {1: ("G1", "G2"), 2: ("G3", "G4")}
TIE_LINES = ("L7-8-1", "L7-8-2")
LOAD_STEP_MW = 100.0


def build_case(transfer=0.0, outage=None, pss=True):
    """Same generation shift and outage as the Part 1 static study."""
    data = copy.deepcopy(model_data.load())
    gens = data["generators"]["GEN"]
    p_col = gens[0].index("P")
    for row in gens[1:]:
        if row[0] in ("G1", "G2"):
            row[p_col] += transfer / 2
        elif row[0] == "G4":
            row[p_col] -= transfer / 2
    if outage:
        data["lines"] = [r for r in data["lines"] if r[0] != outage]
    if not pss:
        data.pop("pss")
    ps = dps.PowerSystemModel(model=data)
    ps.perform_kron_reduction = False
    ps.pf_max_it = 30
    ps.init_dyn_sim()
    if not ps.power_flow_ready:
        raise RuntimeError("Power flow did not converge")
    return ps


class LinearModel:
    def __init__(self, ps, eps=1e-6):
        self.ps = ps
        self.gen = ps.gen["GEN"]
        self.gov = ps.gov["TGOV1"]
        self.names = list(self.gen.par["name"])
        self.n = ps.n_states
        self.x0 = ps.x_0.copy()
        self.bias0 = self.gov.int_par["bias"].copy()
        self.R = self.gov.par["R"].astype(float)
        # Governor order may differ from generator order.
        self.gov_of_gen = [list(self.gov.par["gen"]).index(g) for g in self.names]

        load = ps.loads["Load"]
        i = np.flatnonzero(load.par["bus"] == "B9")[0]
        self.b9 = load.bus_idx_red["terminal"][i]
        v9 = abs(ps.v_0[self.b9])
        p9 = v9**2 * load.y_load[i].real * ps.s_n
        self.y_step = LOAD_STEP_MW / p9 * load.y_load[i]

        lines = ps.lines["Line"]
        self.tie_idx = [list(lines.par["name"]).index(n)
                        for n in TIE_LINES if n in lines.par["name"]]

        self.speed_idx = self.gen.state_idx_global["speed"]
        self.f0 = self.rhs(self.x0, np.zeros(4), 0.0)
        self.y0 = self.output(self.x0)
        self.A, self.B, self.E, self.C = self._linearize(eps)

    # Nonlinear model as used by the time-domain solver -----------------------
    def set_inputs(self, u, d):
        bias = self.bias0.copy()
        bias[self.gov_of_gen] += self.R[self.gov_of_gen] * u
        self.gov.int_par["bias"][:] = bias
        b = self.b9
        self.ps.y_bus_red_mod[b, b] = d * self.y_step

    def rhs(self, x, u, d):
        self.set_inputs(u, d)
        v = self.ps.solve_algebraic(0, x)
        dx = self.ps.state_derivatives(0, x, v)
        self.set_inputs(np.zeros(4), 0.0)
        return dx

    def tie_flow(self, x, v):
        sf = self.ps.lines["Line"].s_from(x, v)
        return sum(sf[i].real for i in self.tie_idx)  # pu on 900 MVA

    def output(self, x, d=0.0):
        self.set_inputs(np.zeros(4), d)
        v = self.ps.solve_algebraic(0, x)
        y = np.r_[x[self.speed_idx], self.tie_flow(x, v)]
        self.set_inputs(np.zeros(4), 0.0)
        return y

    # Central differences ----------------------------------------------------
    def _linearize(self, eps):
        n = self.n
        A = np.zeros((n, n))
        C = np.zeros((5, n))
        u0 = np.zeros(4)
        for k in range(n):
            xp, xm = self.x0.copy(), self.x0.copy()
            xp[k] += eps
            xm[k] -= eps
            A[:, k] = (self.rhs(xp, u0, 0) - self.rhs(xm, u0, 0)) / (2 * eps)
            C[:, k] = (self.output(xp) - self.output(xm)) / (2 * eps)
        B = np.zeros((n, 4))
        for k in range(4):
            up, um = u0.copy(), u0.copy()
            up[k] += eps
            um[k] -= eps
            B[:, k] = (self.rhs(self.x0, up, 0) - self.rhs(self.x0, um, 0)) / (2 * eps)
        E = (self.rhs(self.x0, u0, eps) - self.rhs(self.x0, u0, -eps)) / (2 * eps)
        # The tie-flow output also reacts directly to the load step.
        self.D_tie = (self.output(self.x0, eps)[4] - self.output(self.x0, -eps)[4]) / (2 * eps)
        return A, B, E[:, None], C

    # AGC --------------------------------------------------------------------
    def area_frequency_response(self):
        """beta_a in pu power per pu speed, system base (TGOV1 droop + D_t)."""
        s_n = self.gen.par["S_n"].astype(float)
        dt = self.gov.par["D_t"].astype(float)
        k = (1 / self.R[self.gov_of_gen] + dt[self.gov_of_gen]) * s_n / self.ps.s_n
        return {a: sum(k[self.names.index(g)] for g in gs) for a, gs in AREAS.items()}

    def closed_loop(self, k_i, bias_factor=1.0):
        """State matrix with one AGC integrator per area.

        ACE_1 = dP_tie + B_1 df_1,  ACE_2 = -dP_tie + B_2 df_2
        df_a is the inertia-weighted mean speed of the two units in area a.
        """
        beta = self.area_frequency_response()
        H = self.gen.par["H"].astype(float)
        c_f = np.zeros((2, 5))
        for j, (a, gs) in enumerate(AREAS.items()):
            idx = [self.names.index(g) for g in gs]
            c_f[j, idx] = H[idx] / H[idx].sum()
        c_ace = np.zeros((2, 5))
        c_ace[0, 4], c_ace[1, 4] = 1.0, -1.0
        for j, a in enumerate(AREAS):
            c_ace[j] += bias_factor * beta[a] * c_f[j]
        # z_a (pu on system base) shared equally. u is pu of 900 MVA rating.
        share = np.zeros((4, 2))
        s_n = self.gen.par["S_n"].astype(float)
        for j, gs in enumerate(AREAS.values()):
            for g in gs:
                i = self.names.index(g)
                share[i, j] = 0.5 * self.ps.s_n / s_n[i]
        Cace = c_ace @ self.C
        Dace = c_ace[:, 4:5] * self.D_tie
        A_cl = np.block([[self.A, self.B @ share],
                         [-k_i * Cace, np.zeros((2, 2))]])
        E_cl = np.r_[self.E, -k_i * Dace]
        C_cl = np.c_[self.C, np.zeros((5, 2))]
        return A_cl, E_cl, C_cl, (c_ace, share, beta)


def modes(A, f_min=0.05, f_max=3.0):
    """Oscillatory eigenvalues (one of each pair) with damping ratio."""
    lam, vec = np.linalg.eig(A)
    keep = (lam.imag > 2 * np.pi * f_min) & (lam.imag < 2 * np.pi * f_max)
    out = []
    for l, v in zip(lam[keep], vec[:, keep].T):
        out.append((l, l.imag / (2 * np.pi), -l.real / abs(l), v))
    return sorted(out, key=lambda m: m[1])


def participation(A):
    lam, R = np.linalg.eig(A)
    L = np.linalg.inv(R)
    return lam, R, (R * L.T)
