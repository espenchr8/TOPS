"""Kundur transfer scan (nose curve) from area 1 to area 2.

Run from the TOPS folder with k2a_course.py installed, like static_kundur_course.py.

What it does
- The B9 load (area 2) increases in steps at constant power factor.
- G1 and G2 (area 1) each cover half of the increase. G3 (slack) covers losses.
  The extra power therefore has to cross the B7-B8-B9 tie corridor.
- Each step uses its own Newton-Raphson solver with a warm start from the
  previous step, so the scan can get close to the nose tip.
- Optional generator Q-limits: a PV generator whose Q would exceed its limit
  is switched to PQ with Q fixed at the limit (PV->PQ switching).
- The step size is halved when the power flow fails, until it is below 1 MW.
  The last converged point is reported as the maximum transfer.

The network (Y-bus, shunts, lines) is taken from TOPS. Only the solver is new.
validate() checks that this solver gives the same base case as TOPS.
"""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data

STEP_MW = 25.0          # Initial step in B9 load increase
MIN_STEP_MW = 1.0       # Stop refining the nose tip below this step
TIE_LINES = ("L7-8-1", "L7-8-2")
V_MIN = 0.90            # Lower voltage limit used for the margin (SO GL range)
OUTAGE = "L7-8-1"


def build(data, outage=None):
    """TOPS model for one topology. Only the Y-bus and bookkeeping are used."""
    case = copy.deepcopy(data)
    if outage:
        case["lines"] = [r for r in case["lines"] if r[0] != outage]
    ps = dps.PowerSystemModel(model=case)
    ps.setup()
    ps.build_y_bus_lf()
    return ps


def q_limit(p_mw, s_n=900.0):
    """Armature-current limit: Q_max = sqrt(S_n^2 - P^2) in MVAr."""
    return np.sqrt(np.maximum(s_n**2 - p_mw**2, 0.0))


def newton(Y, s_spec, v_set, types, v_start, tol=1e-8, max_it=20):
    """Polar NR with analytic Jacobian. types: 'SL', 'PV' or 'PQ' per bus.

    s_spec: specified net injection (generation - load) in pu.
    v_start: complex voltage used as the starting point (warm start).
    """
    pv = np.flatnonzero(types == "PV")
    pq = np.flatnonzero(types == "PQ")
    pvpq = np.r_[pv, pq]
    va = np.angle(v_start).copy()
    vm = np.abs(v_start).copy()
    vm[types != "PQ"] = v_set[types != "PQ"]

    for _ in range(max_it):
        V = vm * np.exp(1j * va)
        I = Y @ V
        mis = V * np.conj(I) - s_spec
        f = np.r_[mis.real[pvpq], mis.imag[pq]]
        if np.max(abs(f)) < tol:
            return V, True
        dV = np.diag(V)
        dI = np.diag(I)
        dVn = np.diag(V / abs(V))
        dS_dVm = dV @ np.conj(Y @ dVn) + np.conj(dI) @ dVn
        dS_dVa = 1j * dV @ np.conj(dI - Y @ dV)
        J = np.block([
            [dS_dVa.real[np.ix_(pvpq, pvpq)], dS_dVm.real[np.ix_(pvpq, pq)]],
            [dS_dVa.imag[np.ix_(pq, pvpq)], dS_dVm.imag[np.ix_(pq, pq)]],
        ])
        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            return V, False
        va[pvpq] += dx[:len(pvpq)]
        vm[pq] += dx[len(pvpq):]
        if np.any(vm <= 0.3) or not np.all(np.isfinite(vm)):
            return V, False
    return V, False


class Case:
    """Bus data for one topology and one load increase."""

    def __init__(self, ps, data):
        self.ps = ps
        self.Y = np.asarray(ps.y_bus_lf)
        self.names = ps.buses["name"]
        self.n = len(self.names)
        self.idx = {b: i for i, b in enumerate(self.names)}
        self.base = ps.s_n
        gens = data["generators"]["GEN"]
        h = gens[0]
        self.gen = [dict(zip(h, r)) for r in gens[1:]]
        lh = data["loads"][0]
        self.loads = [dict(zip(lh, r)) for r in data["loads"][1:]]
        self.slack = self.idx[data["slack_bus"]]

    def spec(self, delta_mw, q_fixed):
        """Injections for a B9 load increase of delta_mw.

        q_fixed: {gen name: Q in MVAr} for generators switched to PQ.
        """
        s = np.zeros(self.n, complex)
        v_set = np.ones(self.n)
        types = np.array(["PQ"] * self.n, dtype="<U2")
        for ld in self.loads:
            p, q = ld["P"], ld["Q"]
            if ld["bus"] == "B9":
                scale = (p + delta_mw) / p
                p, q = p * scale, q * scale  # constant power factor
            s[self.idx[ld["bus"]]] -= (p + 1j * q) / self.base
        for g in self.gen:
            i = self.idx[g["bus"]]
            p = g["P"] + (delta_mw / 2 if g["name"] in ("G1", "G2") else 0.0)
            s[i] += p / self.base
            if g["name"] in q_fixed:
                s[i] += 1j * q_fixed[g["name"]] / self.base
            else:
                types[i] = "PV"
                v_set[i] = g["V"]
        types[self.slack] = "SL"
        return s, v_set, types

    def gen_pq(self, V, delta_mw):
        """Generator P and Q in MW/MVAr at the solution."""
        s_bus = V * np.conj(self.Y @ V) * self.base
        out = {}
        for g in self.gen:
            i = self.idx[g["bus"]]
            out[g["name"]] = s_bus[i]  # no load at generator buses
        return out

    def solve(self, delta_mw, v_start, use_q_limits):
        """Power flow with optional PV->PQ switching. Returns (V, ok, limited)."""
        q_fixed = {}
        for _ in range(len(self.gen) + 1):
            s, v_set, types = self.spec(delta_mw, q_fixed)
            V, ok = newton(self.Y, s, v_set, types, v_start)
            if not ok:
                return V, False, q_fixed
            if not use_q_limits:
                return V, True, q_fixed
            pq = self.gen_pq(V, delta_mw)
            new = {}
            for g in self.gen:
                name = g["name"]
                if name in q_fixed or self.idx[g["bus"]] == self.slack:
                    continue
                qmax = q_limit(pq[name].real)
                if pq[name].imag > qmax:
                    new[name] = qmax
            if not new:
                return V, True, q_fixed
            q_fixed.update(new)
            v_start = V
        return V, False, q_fixed

    def tie_transfer(self, V):
        lines = self.ps.lines["Line"]
        sf = lines.s_from(None, V) * self.base
        return sum(sf[k].real for k, n in enumerate(lines.par["name"])
                   if n in TIE_LINES)


def validate(data):
    """Base case from this solver vs TOPS power flow (same network)."""
    ps = dps.PowerSystemModel(model=copy.deepcopy(data))
    ps.pf_max_it = 30
    ps.power_flow()
    c = Case(build(data), data)
    s, v_set, types = c.spec(0.0, {})
    V, ok = newton(c.Y, s, v_set, types, np.ones(c.n, complex))
    err = np.max(abs(V - ps.v_0))
    print(f"Validation against TOPS NR (base case): max |dV| = {err:.2e} pu")
    return err


def scan(data, outage, use_q_limits):
    c = Case(build(data, outage), data)
    b8, b9 = c.idx["B8"], c.idx["B9"]
    V = np.ones(c.n, complex)
    delta, step = 0.0, STEP_MW
    rows, events, seen = [], [], set()
    while step >= MIN_STEP_MW:
        V_new, ok, limited = c.solve(delta, V, use_q_limits)
        if not ok:
            if not rows:
                raise RuntimeError("Base case did not converge")
            delta -= step       # back to the last converged point
            step /= 2
            delta += step
            continue
        V = V_new
        rows.append((delta, c.tie_transfer(V), abs(V[b8]), abs(V[b9])))
        for name in limited:
            if name not in seen:
                seen.add(name)
                events.append((name, rows[-1]))
        delta += step
    return np.array(rows), events


def first_below(rows, v_min, col=2):
    """Interpolated tie transfer where the bus voltage first falls below v_min."""
    v = rows[:, col]
    k = np.flatnonzero(v < v_min)
    if len(k) == 0 or k[0] == 0:
        return None
    i = k[0]
    t = (v[i - 1] - v_min) / (v[i - 1] - v[i])
    return rows[i - 1, 1] + t * (rows[i, 1] - rows[i - 1, 1])


def main():
    data = model_data.load()
    print("Kundur transfer scan: B9 load up, G1/G2 cover it, G3 slack covers losses.")
    validate(data)

    runs = [
        ("Intact, no Q-limits", None, False, "tab:blue", "--"),
        ("Intact, Q-limits", None, True, "tab:blue", "-"),
        (f"N-1 {OUTAGE}, no Q-limits", OUTAGE, False, "tab:red", "--"),
        (f"N-1 {OUTAGE}, Q-limits", OUTAGE, True, "tab:red", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base_transfer = None
    print(f"\n{'Case':32s} {'Base tie':>9s} {'Max tie':>9s} {'Margin':>8s}"
          f" {'B8 at tip':>9s} {f'Tie at V_B8={V_MIN}':>17s}")
    for label, outage, ql, color, ls in runs:
        rows, events = scan(data, outage, ql)
        tie0, tie_max = rows[0, 1], rows[-1, 1]
        if base_transfer is None:
            base_transfer = tie0
        at_vmin = first_below(rows, V_MIN)
        if at_vmin is not None:
            vmin_txt = f"{at_vmin:.0f} MW"
        elif rows[0, 2] < V_MIN:
            vmin_txt = "below at base"
        else:
            vmin_txt = "never below"
        print(f"{label:32s} {tie0:7.0f}MW {tie_max:7.0f}MW {tie_max-base_transfer:6.0f}MW"
              f" {rows[-1, 2]:9.3f} {vmin_txt:>17s}")
        for name, r in events:
            print(f"    {name} reaches Q-limit at tie transfer {r[1]:.0f} MW"
                  f" (B9 load +{r[0]:.0f} MW), V_B8 = {r[2]:.3f} pu")
        ax.plot(rows[:, 1], rows[:, 2], ls, color=color, label=label)
        ax.plot(rows[-1, 1], rows[-1, 2], "o", color=color, ms=4)
        for name, r in events:
            ax.plot(r[1], r[2], "v", color=color, ms=6)
            ax.annotate(name, (r[1], r[2]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=color)
    ax.axvline(base_transfer, color="gray", ls=":", label="Base-case transfer")
    ax.axhline(V_MIN, color="gray", ls="--", lw=0.8, label=f"{V_MIN} pu")
    ax.set_xlabel("Active power B7 → B8 (both tie circuits) [MW]")
    ax.set_ylabel("B8 voltage [pu]")
    ax.set_title("Nose curves for area 1 → area 2 transfer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "static_transfer_nose.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved: {out.name}")
    print("Dots mark the last converged point (nose tip within 1 MW).")
    print("Triangles mark where a generator first reaches its Q-limit.")


if __name__ == "__main__":
    main()
