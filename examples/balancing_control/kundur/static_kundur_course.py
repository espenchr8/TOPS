"""Part 1 static: AC load flow, a line outage and stepwise P-V scans.

Run from TOPS root: python examples/balancing_control/kundur/static_kundur_course.py
Requires the separate TOPS input file tops/ps_models/k2a_course.py.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data

# Chosen study criteria, NOT documented grid-code or equipment ratings.
V_MIN = 0.95
LINE_LIMITS_MVA = {
    "L5-6": 900, "L6-7": 1500, "L7-8-1": 500, "L7-8-2": 500,
    "L8-9-1": 500, "L8-9-2": 500, "L9-10": 1500, "L10-11": 900,
}
# The line data field S_n=100 is TOPS's impedance base, NOT a thermal limit.


def solve(data, outage=None):
    """Make a fresh network and run TOPS's Newton-Raphson AC power flow."""
    case = copy.deepcopy(data)
    if outage:
        # The header row's first element is 'name', so it stays in the table.
        case["lines"] = [row for row in case["lines"] if row[0] != outage]
    ps = dps.PowerSystemModel(model=case)
    ps.pf_max_it = 30
    ps.power_flow()  # TOPS builds Ybus and finds complex bus voltages.
    return ps if ps.power_flow_ready else None


def line_powers(ps):
    """TOPS returns S=V*conj(I) in pu; multiply by system MVA base."""
    line = ps.lines["Line"]
    return (line.par["name"],
            line.s_from(None, ps.v_0) * ps.s_n,
            line.s_to(None, ps.v_0) * ps.s_n)


def show_results(title, ps):
    """Show the base/N-1 state and verify its active-power balance."""
    names, from_s, to_s = line_powers(ps)
    print(f"\n=== {title} ===")
    print("AC power flow converged.")
    print(f"Bus voltages in pu. Chosen lower voltage threshold: {V_MIN:.2f} pu")
    for bus, v in zip(ps.buses["name"], ps.v_0):
        print(f"  {bus:4s} {abs(v):.4f}" +
              ("  Below chosen threshold" if abs(v) < V_MIN else ""))

    print("\nLine flows and illustrative loading thresholds")
    print("The threshold values below are study assumptions, not verified thermal ratings.")
    print("Line      P at from end    Highest |S|     Assumed threshold   Usage")
    for name, sf, st in zip(names, from_s, to_s):
        value = max(abs(sf), abs(st))
        limit = LINE_LIMITS_MVA.get(name)
        if limit:
            print(f"{name:8s} {sf.real:9.1f} MW  {value:10.1f} MVA"
                  f"  {limit:9.0f} MVA      {100*value/limit:5.1f}%")
        else:
            print(f"{name:8s} {sf.real:9.1f} MW  {value:10.1f} MVA"
                  "  unknown")

    # Loads and G1/G2/G4 P are specified; slack generator G3 balances losses.
    generation = sum(np.sum(s.real) for s in ps.load_flow_soln.values())
    load = sum(np.sum(m.par["P"]) for m in ps.loads.values())
    line_losses = np.sum((from_s + to_s).real)
    trafo_losses = sum(np.sum(m.p_line(None, ps.v_0)) * ps.s_n
                       for m in ps.trafos.values())
    losses = line_losses + trafo_losses
    print("\nCheck of active-power balance")
    print(f"Generation: {generation:.2f} MW")
    print(f"Load:       {load:.2f} MW")
    print(f"Grid losses:{losses:9.2f} MW")
    print(f"Generation minus load and losses: {generation-load-losses:.6f} MW")


def load_scan(data, outage, title):
    """Repeated independent load flows: upper P-V branch, not a true nose curve."""
    print(f"\n=== LOAD SCAN: {title} ===")
    print("Factor 1.00 means the original loads at B7 and B9.")
    print("P and Q increase together. G1, G2, G3 and G4 share the added load.")
    print("Line usage is compared with assumed study thresholds.")
    print("Factor   B8 voltage    G3 output    Highest gen |S|    Highest line usage")
    base_load = sum(row[2] for row in data["loads"][1:])
    factors, voltages = [], []
    first_thermal = None

    for step in range(26):  # 1.00 to 1.50 in steps of 0.02.
        factor = 1 + 0.02 * step
        case = copy.deepcopy(data)
        for row in case["loads"][1:]:
            row[2] *= factor  # Active load P, MW.
            row[3] *= factor  # Reactive load Q, MVAr; Q/P stays constant.
        # Generator table row 0 is a header. G1/G2/G4 each get 1/4 of
        # extra load; G3 (slack) gets 1/4 plus the change in network losses.
        for index in (1, 2, 4):
            case["generators"]["GEN"][index][4] += (factor - 1)*base_load/4

        ps = solve(case, outage)
        if ps is None:
            print(f"At factor {factor:.2f}, the power flow did not converge"
                  " within 30 iterations.")
            print("This does not establish the exact voltage-collapse point.")
            break

        gen = ps.gen["GEN"]
        gen_s = ps.load_flow_soln[gen]  # MW + j MVAr.
        if np.any(abs(gen_s) > gen.par["S_n"]):
            print(f"At factor {factor:.2f}, a generator exceeds its"
                  " 900 MVA nameplate rating.")
            print("The point is excluded, and the scan stops.")
            break

        names, from_s, to_s = line_powers(ps)
        loading = [(name, 100*max(abs(sf), abs(st))/LINE_LIMITS_MVA[name])
                   for name, sf, st in zip(names, from_s, to_s)
                   if name in LINE_LIMITS_MVA]
        worst_name, worst_pct = max(loading, key=lambda pair: pair[1])
        if first_thermal is None and worst_pct > 100:
            first_thermal = factor
            print(f"At factor {factor:.2f}, {worst_name} first exceeds its"
                  f" assumed line threshold ({worst_pct:.1f}% usage).")

        b8 = np.flatnonzero(ps.buses["name"] == "B8")[0]
        v8 = abs(ps.v_0[b8])
        print(f"{factor:5.2f}    {v8:7.4f} pu  {gen_s[2].real:7.1f} MW"
              f"       {max(abs(gen_s)):7.1f} MVA      {worst_name} {worst_pct:5.1f}%")
        factors.append(factor)
        voltages.append(v8)

    if first_thermal is not None:
        print(f"Chosen line threshold was first crossed at factor {first_thermal:.2f}.")
        print("Subsequent curve points show mathematical power-flow solutions"
              " outside this assumed operating limit.")
    return factors, voltages


def main():
    # Same model_data.load() -> PowerSystemModel pattern as the master baseline.
    data = model_data.load()  # Buses, lines, trafos, loads, shunts, generators.
    print(f"Kundur data: k2a_course.py")
    print(f"System base: {data['base_mva']} MVA")
    print(f"Slack bus: {data['slack_bus']}")
    print("Method: TOPS AC Newton-Raphson power flow")
    print("The 0.95 pu threshold and line MVA limits are study assumptions.")

    base = solve(data)
    if base is None:
        raise RuntimeError("Base power flow did not converge")
    show_results("BASE CASE: all lines connected", base)

    outage = solve(data, "L7-8-1")
    if outage is None:
        raise RuntimeError("N-1 power flow did not converge")
    show_results("N-1: L7-8-1 removed; L7-8-2 remains", outage)

    for removed, label in ((None, "all lines"), ("L7-8-1", "N-1 L7-8-1")):
        factors, voltages = load_scan(data, removed, label)
        plt.plot(factors, voltages, "o-", label=label)
    plt.axhline(V_MIN, color="gray", linestyle="--", label="chosen 0.95 pu")
    plt.xlabel("Load factor at B7 and B9")
    plt.ylabel("Voltage at B8 [pu]")
    plt.title("Stepwise P-V scan (converged points)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
