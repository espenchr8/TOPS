"""Kundur static study. Run from the TOPS folder with k2a_course.py installed.

Part 1: contingency analysis with TOPS power flow
    Base case, higher transfer, N-1 (L7-8-1) and N-1 with changed generation.
    Voltages and line loading are compared with chosen limits.

Part 2: nose curves for area 1 -> area 2 transfer
    The B9 load increases in steps; G1/G2 cover the increase, so the extra
    power must cross the tie corridor. A Newton-Raphson solver with warm start
    and generator Q-limits (PV->PQ switching) follows each curve to the nose tip.
    The network (Y-bus, lines) comes from TOPS; validate() compares the solver
    with the TOPS power flow.

Output: one combined figure (a) line loading, (b) nose curves, sized for a
two-column IEEE figure (figure*).
"""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data


# Chosen limits for comparison.
SHOW_DETAILS = False  # True prints all buses and lines for each case.
V_MIN = 0.90
TRANSFER_CHANGE = 100.0  # MW shifted between areas, not extra total demand.
LINE_LIMITS = dict(zip(
    ("L5-6", "L6-7", "L7-8-1", "L7-8-2", "L8-9-1", "L8-9-2", "L9-10", "L10-11"),
    (900, 1800, 500, 500, 500, 500, 1800, 900)))
# Line S_n=100 in k2a_course is an impedance base, not a thermal rating.

OUTAGE = "L7-8-1"
TIE_LINES = ("L7-8-1", "L7-8-2")
STEP_MW = 10.0       # Initial step in B9 load increase for the nose curves
MIN_STEP_MW = 1.0    # Stop refining the nose tip below this step

COLORS = {"Base": "tab:blue", "Higher transfer": "tab:orange",
          "N-1": "tab:green", "N-1 with changed generation": "tab:red"}


# =============================================================================
# PART 1: CONTINGENCY ANALYSIS (TOPS POWER FLOW)
# =============================================================================

def solve(data, outage=None, transfer=0.0):
    """Build a fresh model and run TOPS AC power flow (Newton-Raphson)."""
    case = copy.deepcopy(data)
    # Positive transfer: G1 and G2 each increase by half the chosen MW.
    # G4 decreases by half. Slack G3 supplies the remaining decrease
    # plus changed losses. Negative transfer reverses this redispatch.
    generators = case["generators"]["GEN"]
    p_col = generators[0].index("P")
    for row in generators[1:]:
        if row[0] in ("G1", "G2"):
            row[p_col] += transfer / 2
        elif row[0] == "G4":
            row[p_col] -= transfer / 2
    if outage:
        case["lines"] = [row for row in case["lines"] if row[0] != outage]
    ps = dps.PowerSystemModel(model=case)
    ps.pf_max_it = 30
    ps.power_flow()
    return ps if ps.power_flow_ready else None


def flows(ps, v=None):
    """Line power at both ends, converted from pu to MVA."""
    v = ps.v_0 if v is None else v
    lines = ps.lines["Line"]
    return (lines.par["name"], lines.s_from(None, v)*ps.s_n,
            lines.s_to(None, v)*ps.s_n)


def show_case(title, ps):
    names, sf, st = flows(ps)
    print(f"\n=== {title} ===")
    print(f"Bus voltages. Our chosen lower threshold is {V_MIN:.2f} pu")
    for bus, v in zip(ps.buses["name"], ps.v_0):
        mark = "  below chosen threshold" if abs(v) < V_MIN else ""
        print(f"{bus:4s} {abs(v):.4f} pu{mark}")

    print("\nLines: MW at the from end, and highest MVA at either end")
    print("Line limits below are assumptions for this study.")
    print("Line       P from       Highest S     Assumed limit   Used")
    for name, a, b in zip(names, sf, st):
        s = max(abs(a), abs(b))
        limit = LINE_LIMITS[name]
        print(f"{name:8s} {a.real:7.1f} MW  {s:8.1f} MVA  "
              f"{limit:7.0f} MVA     {100*s/limit:5.1f}%")

    machines = ps.gen["GEN"]
    powers = ps.load_flow_soln[machines]
    print("Generator output after solving the power flow")
    for name, power, rating in zip(machines.par["name"], powers, machines.par["S_n"]):
        mark = "  above MVA rating" if abs(power) > rating else ""
        print(f"{name}: {power.real:.1f} MW, {abs(power):.1f} MVA{mark}")


def show_summary(cases):
    """One row per case keeps the changes and results together."""
    print("\n=== 1. PRODUCTION CHANGES AND ONE LINE OUTAGE ===")
    print(f"Loads stay unchanged. N-1 means {OUTAGE} is disconnected.")
    print(f"Higher transfer: G1/G2 each +{TRANSFER_CHANGE/2:.0f} MW, G4 -{TRANSFER_CHANGE/2:.0f} MW.")
    print("N-1 with changed generation reverses that production change.")
    print("G3 balances the remaining demand and losses in every case.")
    print("\nCase                         B8 [pu]  B7->B8 [MW]  Losses [MW]  Highest line use")
    errors = []
    for title, ps in cases:
        names, sf, st = flows(ps)
        tie = sum(a.real for n, a in zip(names, sf) if n in TIE_LINES)
        losses = np.sum((sf + st).real) + sum(
            np.sum(t.p_line(None, ps.v_0))*ps.s_n for t in ps.trafos.values())
        generation = sum(np.sum(v.real) for v in ps.load_flow_soln.values())
        load = sum(np.sum(m.par["P"]) for m in ps.loads.values())
        errors.append(abs(generation - load - losses))
        ratios = [(n, 100*max(abs(a), abs(b))/LINE_LIMITS[n])
                  for n, a, b in zip(names, sf, st)]
        line, pct = max(ratios, key=lambda item: item[1])
        b8 = np.flatnonzero(ps.buses["name"] == "B8")[0]
        print(f"{title:28s} {abs(ps.v_0[b8]):7.4f} {tie:12.1f}"
              f" {losses:12.2f}  {line} {pct:.1f}%")
        low = [n for n, v in zip(ps.buses["name"], ps.v_0) if abs(v) < V_MIN]
        over = [n for n, ratio in ratios if ratio > 100]
        print(f"  Below {V_MIN} pu: " + (", ".join(low) or "none")
              + " | Above assumed line limit: " + (", ".join(over) or "none"))
        gen = ps.gen["GEN"]
        if np.any(abs(ps.load_flow_soln[gen]) > gen.par["S_n"]):
            print("  A generator exceeds its MVA rating.")
    slack = cases[0][1].load_flow_soln[cases[0][1].gen["GEN"]][2].real
    print(f"\nLargest power balance error across these cases: {max(errors):.3e} MW")
    print(f"Base-case slack (G3) output: {slack:.2f} MW")


# =============================================================================
# PART 2: NOSE CURVES (OWN NEWTON-RAPHSON WITH WARM START AND Q-LIMITS)
# =============================================================================

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
    """Bus data for one topology. The Y-bus is taken from TOPS."""

    def __init__(self, data, outage=None):
        case = copy.deepcopy(data)
        if outage:
            case["lines"] = [r for r in case["lines"] if r[0] != outage]
        self.ps = dps.PowerSystemModel(model=case)
        self.ps.setup()
        self.ps.build_y_bus_lf()
        self.Y = np.asarray(self.ps.y_bus_lf)
        self.idx = {b: i for i, b in enumerate(self.ps.buses["name"])}
        self.n = len(self.idx)
        self.base = self.ps.s_n
        gens = data["generators"]["GEN"]
        self.gen = [dict(zip(gens[0], r)) for r in gens[1:]]
        self.loads = [dict(zip(data["loads"][0], r)) for r in data["loads"][1:]]
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

    def solve(self, delta_mw, v_start, use_q_limits):
        """Power flow with optional PV->PQ switching. Returns (V, ok, limited)."""
        q_fixed = {}
        for _ in range(len(self.gen) + 1):
            s, v_set, types = self.spec(delta_mw, q_fixed)
            V, ok = newton(self.Y, s, v_set, types, v_start)
            if not ok or not use_q_limits:
                return V, ok, q_fixed
            s_bus = V * np.conj(self.Y @ V) * self.base  # no load at gen buses
            new = {}
            for g in self.gen:
                i = self.idx[g["bus"]]
                if g["name"] in q_fixed or i == self.slack:
                    continue
                if s_bus[i].imag > q_limit(s_bus[i].real):
                    new[g["name"]] = q_limit(s_bus[i].real)
            if not new:
                return V, True, q_fixed
            q_fixed.update(new)
            v_start = V
        return V, False, q_fixed

    def tie_transfer(self, V):
        names, sf, _ = flows(self.ps, V)
        return sum(a.real for n, a in zip(names, sf) if n in TIE_LINES)


def validate(data, ps_base):
    """Base case from the own solver vs the TOPS power flow (same network)."""
    c = Case(data)
    s, v_set, types = c.spec(0.0, {})
    V, _ = newton(c.Y, s, v_set, types, np.ones(c.n, complex))
    err = np.max(abs(V - ps_base.v_0))
    print(f"\nOwn NR solver vs TOPS power flow (base case): max |dV| = {err:.2e} pu")


def nose_curve(data, outage, use_q_limits):
    """Increase the B9 load until the power flow has no solution."""
    c = Case(data, outage)
    b8 = c.idx["B8"]
    V = np.ones(c.n, complex)
    delta, step = 0.0, STEP_MW
    rows, events, last_fail_limited = [], [], {}
    while step >= MIN_STEP_MW:
        V_new, ok, limited = c.solve(delta, V, use_q_limits)
        if not ok:
            if not rows:
                raise RuntimeError("Base case did not converge")
            last_fail_limited = limited
            delta -= step       # back to the last converged point
            step /= 2           # and approach the nose tip with smaller steps
            delta += step
            continue
        V = V_new
        rows.append((delta, c.tie_transfer(V), abs(V[b8])))
        for name in limited:
            if name not in [e[0] for e in events]:
                events.append((name, rows[-1]))
        delta += step
    # If the step just beyond the tip hit a Q-limit, that limit triggers the
    # collapse; mark it at the tip (within MIN_STEP_MW).
    for name in last_fail_limited:
        if name not in [e[0] for e in events]:
            events.append((name, rows[-1]))
    return np.array(rows), events


def show_nose_summary(results):
    print("\n=== 2. NOSE CURVES: AREA 1 -> AREA 2 TRANSFER ===")
    print("B9 load increases at constant power factor. G1/G2 each cover half.")
    print("G3 (slack) covers changed losses. Q-limit: Q_max = sqrt(S_n^2 - P^2).")
    print(f"\n{'Case':30s} {'Base tie':>9s} {'Max tie':>9s} {'Margin':>8s} {'B8 at tip':>10s}")
    base_tie = results[0][2][0, 1]
    for label, _, rows, events in results:
        print(f"{label:30s} {rows[0, 1]:7.0f}MW {rows[-1, 1]:7.0f}MW"
              f" {rows[-1, 1] - base_tie:6.0f}MW {rows[-1, 2]:10.3f}")
        for name, r in events:
            print(f"    {name} reaches its Q-limit at {r[1]:.0f} MW tie transfer,"
                  f" B8 = {r[2]:.3f} pu")
    print("Max tie is the last converged point (nose tip within 1 MW).")


# =============================================================================
# COMBINED FIGURE (IEEE figure*, full page width)
# =============================================================================

def plot_static(cases, results):
    plt.rcParams.update({"font.size": 8})
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) Line loading for base case, N-1 and redispatch
    group = [c for c in cases if c[0] != "Higher transfer"]
    line_names = list(LINE_LIMITS)
    positions = np.arange(len(line_names))
    width = 0.8 / len(group)
    for i, (label, ps) in enumerate(group):
        names, sf, st = flows(ps)
        loading = {n: 100 * max(abs(a), abs(b)) / LINE_LIMITS[n]
                   for n, a, b in zip(names, sf, st)}
        values = [loading.get(n, np.nan) for n in line_names]  # no bar if out
        offset = (i - (len(group) - 1) / 2) * width
        short = {"N-1 with changed generation": "N-1 + redispatch"}
        ax_a.bar(positions + offset, values, width, color=COLORS[label],
                 label=short.get(label, label))
    ax_a.axhline(100, color="gray", ls="--", lw=0.8)
    ax_a.set_xticks(positions, line_names, rotation=30)
    ax_a.set_ylabel("Use of assumed MVA limit [%]")
    ax_a.set_ylim(0, 128)
    ax_a.set_title("(a) Line loading, base case and N-1")
    ax_a.legend(fontsize=6.5, loc="upper center", ncol=3, framealpha=0.9)
    ax_a.grid(True, axis="y", alpha=0.3)

    # (b) Nose curves, same colours as (a): blue intact, green N-1
    for label, outage, rows, events in results:
        color = COLORS["N-1"] if outage else COLORS["Base"]
        ls = "-" if "with" in label else "--"
        ax_b.plot(rows[:, 1], rows[:, 2], ls, color=color, lw=1.2, label=label)
        ax_b.plot(rows[-1, 1], rows[-1, 2], "o", color=color, ms=3)
        for name, r in events:
            ax_b.plot(r[1], r[2], "v", color=color, ms=4)
            ax_b.annotate(name, (r[1], r[2]), textcoords="offset points",
                          xytext=(3, 3), fontsize=6.5, color=color)
    ax_b.axvline(results[0][2][0, 1], color="gray", ls=":", lw=0.8)
    ax_b.axhline(V_MIN, color="gray", ls="--", lw=0.8)
    ax_b.set_xlabel("Transfer B7 → B8 [MW]")
    ax_b.set_ylabel("B8 voltage [pu]")
    ax_b.set_title("(b) Nose curves, area 1 → area 2")
    ax_b.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.6, 0.0))
    ax_b.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=1.5)
    # Comment out fig.savefig(...) to disable PNG saving.
    output_file = Path(__file__).resolve().parent / "static_combined.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved: {output_file.name}")


def main():
    data = model_data.load()  # Same data loading as the dynamic study.
    print("Kundur data from k2a_course.py.")
    print(f"System base {data['base_mva']} MVA. Slack bus {data['slack_bus']}.")
    print(f"The {V_MIN} pu limit is from SO GL. Line limits are assumptions.")

    cases = []
    for title, removed, transfer in (
            ("Base", None, 0),
            ("Higher transfer", None, TRANSFER_CHANGE),
            ("N-1", OUTAGE, 0),
            ("N-1 with changed generation", OUTAGE, -TRANSFER_CHANGE)):
        ps = solve(data, removed, transfer)
        if ps is None:
            raise RuntimeError(f"Power flow failed for {title}")
        if SHOW_DETAILS:
            show_case(title, ps)
        cases.append((title, ps))
    show_summary(cases)

    validate(data, cases[0][1])
    results = []
    for label, outage, use_q in (
            ("Intact, no Q-limits", None, False),
            ("Intact, with Q-limits", None, True),
            (f"N-1, no Q-limits", OUTAGE, False),
            (f"N-1, with Q-limits", OUTAGE, True)):
        rows, events = nose_curve(data, outage, use_q)
        results.append((label, outage, rows, events))
    show_nose_summary(results)

    plot_static(cases, results)
    plt.show()


if __name__ == "__main__":
    main()
