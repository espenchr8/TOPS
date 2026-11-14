"""Part 2: eigenvalues of the Kundur model with AGC (base case).

Built on agc_kundur_course.py: same model, power flow and AGC equation (4).
The TOPS model is linearized around the base-case operating point:

    dx/dt = A x + B_u dp + E d
    y     = C x + D d

x    the TOPS states (machines, governors, AVRs, PSSs)
dp   change in each governor reference [MW]
d    the B9 load step, 1 = 100 MW at the pre-event voltage (as in Part 1)
y    generator speeds [pu] and the B7->B8 corridor flow [MW]

The derivatives are found by central differences: each state is moved a
little up and down, the network is solved, and the change in dx/dt is
divided by the step. The two AGC integrators are then added to A:

    dz_a/dt = -K_I * ACE_a,   ACE_a = dP_tie,a + B_a * w_a

Checks: the eigenvalues do not depend on the step size, and a 10 MW load
step in the linear model gives the same frequency as TOPS with AGC.

Run from the TOPS folder, next to agc_kundur_course.py.
"""

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from scipy.linalg import expm
from tops.ps_models import k2a as model_data

import agc_kundur_course as agc
from agc_kundur_course import AREAS, K_I, tie_flow
from dynamic_kundur_course import DT, LOAD_STEP_MW, T_EVENT

EPS = 1e-6                                   # Step for central differences
K_SWEEP = np.logspace(-3, 1, 400)            # AGC gains to test [1/s]


class LinearModel:
    """Linear model of the same system as agc_kundur_course.py."""

    def __init__(self, eps=EPS):
        # Same setup as simulate() in agc_kundur_course.py
        ps = dps.PowerSystemModel(model=model_data.load())
        ps.perform_kron_reduction = False
        ps.pf_max_it = 30
        ps.init_dyn_sim()
        if not ps.power_flow_ready:
            raise RuntimeError("Initial power flow did not converge")
        self.ps = ps
        gen = ps.gen["GEN"]
        gov = ps.gov["TGOV1"]
        load = ps.loads["Load"]
        names = list(gen.par["name"])

        # Same AGC data as agc_kundur_course.py
        self.area_idx = [[names.index(g) for g in units] for units in AREAS.values()]
        self.gov_idx = [list(gov.par["gen"]).index(g) for g in names]
        self.R = gov.par["R"][self.gov_idx]
        self.S_n = gen.par["S_n"]
        self.H = gen.par["H"]
        unit_k = self.S_n * (1 / self.R + gov.par["D_t"][self.gov_idx])
        self.B = np.array([unit_k[i].sum() for i in self.area_idx])
        self.bias0 = gov.int_par["bias"].copy()
        self.gov = gov

        # Same load step as Part 1: 100 MW at the pre-event B9 voltage
        load_idx = np.flatnonzero(load.par["bus"] == "B9")[0]
        self.b9 = load.bus_idx_red["terminal"][load_idx]
        pre_event_p = abs(ps.v_0[self.b9])**2 * load.y_load[load_idx].real * ps.s_n
        self.y_step = (LOAD_STEP_MW / pre_event_p) * load.y_load[load_idx]

        self.names = names
        self.speed_idx = gen.state_idx_global["speed"]
        self.x0 = ps.x_0.copy()
        self.n = len(self.x0)
        self.linearize(eps)

    def set_inputs(self, dp, d):
        """dp: reference change per unit [MW]. d: load step (1 = 100 MW)."""
        bias = self.bias0.copy()
        bias[self.gov_idx] += self.R * dp / self.S_n   # as in agc_kundur_course.py
        self.gov.int_par["bias"][:] = bias
        self.ps.y_bus_red_mod[self.b9, self.b9] = d * self.y_step

    def f(self, x, dp=None, d=0.0):
        """State derivatives dx/dt from TOPS, with the network solved."""
        self.set_inputs(np.zeros(4) if dp is None else dp, d)
        v = self.ps.solve_algebraic(0, x)
        dx = self.ps.state_derivatives(0, x, v)
        self.set_inputs(np.zeros(4), 0.0)
        return dx

    def y(self, x, d=0.0):
        """Outputs: four generator speeds [pu] and corridor flow [MW]."""
        self.set_inputs(np.zeros(4), d)
        v = self.ps.solve_algebraic(0, x)
        out = np.r_[x[self.speed_idx], tie_flow(self.ps, x, v)]
        self.set_inputs(np.zeros(4), 0.0)
        return out

    def linearize(self, eps):
        n = self.n
        self.A = np.zeros((n, n))
        self.C = np.zeros((5, n))
        for k in range(n):
            up, down = self.x0.copy(), self.x0.copy()
            up[k] += eps
            down[k] -= eps
            self.A[:, k] = (self.f(up) - self.f(down)) / (2 * eps)
            self.C[:, k] = (self.y(up) - self.y(down)) / (2 * eps)
        self.Bu = np.zeros((n, 4))
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = eps
            self.Bu[:, k] = (self.f(self.x0, dp) - self.f(self.x0, -dp)) / (2 * eps)
        self.E = (self.f(self.x0, d=eps) - self.f(self.x0, d=-eps)) / (2 * eps)
        self.D = (self.y(self.x0, eps) - self.y(self.x0, -eps)) / (2 * eps)

    def with_agc(self, k_i):
        """Add the two AGC integrators z_a [MW] of equation (4)."""
        # ACE_a = export of area a + B_a * (H-weighted speed of area a)
        c_ace = np.zeros((2, 5))
        c_ace[0, 4], c_ace[1, 4] = 1.0, -1.0
        for j, i in enumerate(self.area_idx):
            c_ace[j, i] = self.B[j] * self.H[i] / self.H[i].sum()
        # z_a is split equally between the two governors in area a
        share = np.zeros((4, 2))
        for j, i in enumerate(self.area_idx):
            share[i, j] = 1 / len(i)
        A = np.block([[self.A, self.Bu @ share],
                      [-k_i * c_ace @ self.C, np.zeros((2, 2))]])
        E = np.r_[self.E, -k_i * c_ace @ self.D]
        C = np.c_[self.C, np.zeros((5, 2))]
        return A, E, C

    def coi_step_response(self, k_i, step, t_end):
        """COI frequency [Hz] for a load step `step` (1 = 100 MW) at T_EVENT."""
        A, E, C = self.with_agc(k_i)
        n = len(A)
        M = expm(np.block([[A, E[:, None]], [np.zeros((1, n + 1))]]) * DT)
        Ad, Ed = M[:n, :n], M[:n, n]
        x = np.zeros(n)
        t = np.arange(0, t_end + DT / 2, DT)
        speeds = []
        for tk in t:
            d = step if tk >= T_EVENT - 1e-9 else 0.0
            speeds.append(C[:4] @ x + self.D[:4] * d)
            x = Ad @ x + Ed * d
        weights = self.H * self.S_n
        return t, 50 * (1 + np.average(np.array(speeds), axis=1, weights=weights))

    def mode_shape(self, A, k):
        """Speed of each generator in mode k, relative to the system COI."""
        lam, vec = np.linalg.eig(A)
        s = vec[self.speed_idx, k]
        s = s - np.average(s, weights=self.H)
        return s / s[np.argmax(abs(s))]


def rotor_modes(lm, A):
    """Oscillating modes 0.3-2 Hz where the rotor speeds take part.

    Returns (index, eigenvalue, speed shape, rotor share). Rotor share is the
    part of the participation factors that lies in rotor speed and angle.
    A high share means an electromechanical mode, a low share a control mode.
    """
    lam, vec = np.linalg.eig(A)
    part = abs(vec * np.linalg.inv(vec).T)              # participation factors
    states = list(lm.ps.state_desc[:, 1]) + ["agc"] * (len(A) - lm.n)
    rotor = np.array([s in ("speed", "angle") for s in states])
    modes = []
    for k in np.argsort(lam.imag):
        l = lam[k]
        if not 0.3 * 2 * np.pi < l.imag < 2 * 2 * np.pi:
            continue
        s = vec[lm.speed_idx, k]
        s = s - np.average(s, weights=lm.H)
        if abs(s).max() < 1e-6 * abs(vec[:, k]).max():
            continue
        s = s / s[np.argmax(abs(s))]
        modes.append((k, l, s, part[rotor, k].sum() / part[:, k].sum()))
    return modes


def inter_area(lm, A):
    """Electromechanical mode where G1/G2 swing against G3/G4."""
    best = None
    for k, l, s, share in rotor_modes(lm, A):
        a1, a2 = s[lm.area_idx[0]].mean(), s[lm.area_idx[1]].mean()
        score = abs(a1 - a2) * min(abs(s)) * share
        if best is None or score > best[0]:
            best = (score, l)
    return best[1]


def check_linear_model(lm):
    print("=== 1. CHECK OF THE LINEAR MODEL ===")
    print(f"States: {lm.n} from TOPS, plus 2 AGC integrators.")
    lam = np.sort_complex(np.linalg.eigvals(lm.A))
    print(f"Largest real part without AGC: {lam.real.max():.1e} "
          "(the zero is the angle reference, not an instability)")
    for eps in (EPS * 10, EPS / 10):
        lam2 = np.sort_complex(np.linalg.eigvals(LinearModel(eps).A))
        print(f"Eigenvalue change with step {eps:g} instead of {EPS:g}: "
              f"{np.max(abs(lam2 - lam)):.1e}")

    # Same AGC run in TOPS, with a 10 MW step (small, so it is nearly linear).
    agc.LOAD_STEP_MW = 10.0
    with contextlib.redirect_stdout(io.StringIO()):
        t, _, coi, _, _ = agc.simulate(K_I)
    agc.LOAD_STEP_MW = LOAD_STEP_MW
    _, coi_lin = lm.coi_step_response(K_I, 10.0 / LOAD_STEP_MW, t[-1])
    err = abs(coi - coi_lin).max() / abs(coi - 50).max()
    print(f"10 MW step with AGC: linear model vs TOPS, largest frequency "
          f"difference {100 * err:.1f}% of the largest deviation")


def show_modes(lm):
    print("\n=== 2. ROTOR MODES WITHOUT AGC (base case) ===")
    print("Speed of each generator relative to the COI: size and angle [deg].")
    print("Rotor share: part of the participation in rotor speed and angle.")
    print("Frequency  Damping  Rotor share   " + "   ".join(f"{g:>10s}" for g in lm.names))
    for k, l, s, share in rotor_modes(lm, lm.A):
        shape = "   ".join(f"{abs(v):4.2f} {np.angle(v, deg=True):+5.0f}" for v in s)
        print(f"{l.imag / (2 * np.pi):6.3f} Hz  {-l.real / abs(l):6.3f}  {share:8.2f}      {shape}")
    ia = inter_area(lm, lm.A)
    print(f"Inter-area mode: {ia.imag / (2 * np.pi):.3f} Hz. G1/G2 swing against "
          "G3/G4 (about 180 deg apart),")
    print("and it has the larger rotor share of the two modes with this pattern.")
    print("The other one is mostly AVR, PSS and field states (a control mode).")


def gain_sweep(lm):
    print("\n=== 3. INTER-AREA MODE VERSUS AGC GAIN ===")
    A0, _, _ = lm.with_agc(0.0)
    prev = inter_area(lm, A0)
    print(f"Without AGC: {prev.imag / (2 * np.pi):.3f} Hz, "
          f"damping ratio {-prev.real / abs(prev):.3f}")
    track, limit = [], None
    for k in K_SWEEP:
        A, _, _ = lm.with_agc(k)
        lam = np.linalg.eigvals(A)
        prev = lam[np.argmin(abs(lam - prev))]          # follow the same mode
        track.append(prev)
        if limit is None and lam.real.max() > 1e-6:
            limit = (k, lam[np.argmax(lam.real)])
    track = np.array(track)
    zeta = -track.real / abs(track)
    print("K_I [1/s]   T_i [s]   Frequency   Damping ratio")
    for k in (0.01, K_I, 0.5, 1.0, 2.0):
        i = np.argmin(abs(K_SWEEP - k))
        print(f"{k:8.2f}   {1 / k:7.1f}   {track[i].imag / (2 * np.pi):6.3f} Hz   {zeta[i]:8.3f}")
    k, l = limit
    print(f"Unstable from K_I = {k:.1f} 1/s (T_i = {1 / k:.2f} s): the "
          f"{l.imag / (2 * np.pi):.2f} Hz mode crosses the imaginary axis.")
    print(f"That is {k / K_I:.0f} times the gain used (K_I = {K_I:g} 1/s).")
    A, _, _ = lm.with_agc(K_I)
    lam = np.linalg.eigvals(A)
    slow = lam[(abs(lam) < 0.08) & (abs(lam) > 1e-6) & (lam.imag >= 0)]
    print(f"Slowest modes with K_I = {K_I:g}: " + ", ".join(
        f"{s.real:.3f}{s.imag:+.3f}j" for s in np.sort_complex(slow)))
    return zeta, k


def plot(zeta, k_limit):
    """Column-width figure, same size and font as the other figures."""
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    ax.semilogx(K_SWEEP, zeta, lw=1.0, label="Inter-area mode")
    ax.axvline(K_I, color="gray", ls=":", lw=0.8,
               label=f"$K_I$ = {K_I:g} s$^{{-1}}$ (used)")
    ax.plot(k_limit, 0, "rx", ms=4, mew=1.2,
            label=f"Unstable from {k_limit:.1f} s$^{{-1}}$")
    ax.axhline(0, color="gray", ls="--", lw=0.6)
    ax.set_xlabel("AGC gain $K_I$ [s$^{-1}$]")
    ax.set_ylabel("Damping ratio $\\zeta$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5.5, ncol=1, loc="lower left", handlelength=1.5,
              borderpad=0.3)
    fig.tight_layout(pad=0.3)

    output_file = Path(__file__).resolve().parent / "agc_eigen.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    # Vector version for LaTeX: sharp text at any zoom.
    fig.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=0.02)
    plt.show()


def main():
    print("Kundur data from k2a.py. Linearized around the base case.")
    print(f"Same AGC as agc_kundur_course.py. Step for central differences: {EPS:g}")
    lm = LinearModel()
    check_linear_model(lm)
    show_modes(lm)
    zeta, k_limit = gain_sweep(lm)
    plot(zeta, k_limit)


if __name__ == "__main__":
    main()
