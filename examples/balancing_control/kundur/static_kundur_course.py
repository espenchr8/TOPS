"""Kundur static study. Run from the TOPS folder with k2a_course.py installed."""

import copy
import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data

# Chosen limits for comparison, not documented operating limits.
V_MIN = 0.95
TRANSFER_CHANGE = 100.0  # MW shifted between areas, not extra total demand.
LINE_LIMITS = dict(zip(
    ("L5-6", "L6-7", "L7-8-1", "L7-8-2", "L8-9-1", "L8-9-2", "L9-10", "L10-11"),
    (900, 1500, 500, 500, 500, 500, 1500, 900)))
# Line S_n=100 in k2a_course is an impedance base, not a thermal rating.


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


def flows(ps):
    """Line power at both ends, converted from pu to MVA."""
    lines = ps.lines["Line"]
    return (lines.par["name"], lines.s_from(None, ps.v_0)*ps.s_n,
            lines.s_to(None, ps.v_0)*ps.s_n)


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

    # Check: slack generator G3 covers remaining demand and network losses.
    gen = sum(np.sum(s.real) for s in ps.load_flow_soln.values())
    load = sum(np.sum(m.par["P"]) for m in ps.loads.values())
    losses = np.sum((sf + st).real) + sum(
        np.sum(t.p_line(None, ps.v_0))*ps.s_n for t in ps.trafos.values())
    print(f"\nPower balance: {gen:.2f} MW generation = "
          f"{load:.2f} MW load + {losses:.2f} MW losses")
    print(f"Balance error: {gen-load-losses:.6f} MW (should be close to zero)")
    machines = ps.gen["GEN"]
    powers = ps.load_flow_soln[machines]
    print("Generator output after solving the power flow")
    for name, power, rating in zip(machines.par["name"], powers, machines.par["S_n"]):
        mark = "  above MVA rating" if abs(power) > rating else ""
        print(f"{name}: {power.real:.1f} MW, {abs(power):.1f} MVA{mark}")
    tie = sum(a.real for name, a in zip(names, sf)
              if name in ("L7-8-1", "L7-8-2"))
    print(f"B7 to B8 transfer: {tie:.1f} MW (positive means towards area 2)")


def compare_cases(cases):
    """Compare the four operating points, including every remaining line."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    line_names = list(LINE_LIMITS)
    positions = np.arange(len(line_names))
    width = 0.2
    for i, (label, ps) in enumerate(cases):
        axes[0].plot(ps.buses["name"], abs(ps.v_0), "o-", label=label)
        names, sf, st = flows(ps)
        loading = {name: 100 * max(abs(a), abs(b)) / LINE_LIMITS[name]
                   for name, a, b in zip(names, sf, st)}
        # A missing bar means the line is disconnected, not lightly loaded.
        values = [loading.get(name, np.nan) for name in line_names]
        axes[1].bar(positions + (i - 1.5) * width, values, width, label=label)
    axes[0].axhline(V_MIN, color="gray", ls="--", label="Chosen 0.95 pu")
    axes[0].set_ylabel("Bus voltage [pu]")
    axes[0].set_title("Voltage before and after changes in operation")
    axes[1].axhline(100, color="gray", ls="--")
    axes[1].set_xticks(positions, line_names, rotation=30)
    axes[1].set_ylabel("Use of assumed MVA limit [%]")
    axes[1].set_title("Line loading (missing bar = disconnected line)")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    # Compare the corrective action with N-1, rather than with the base case.
    before, after = cases[2][1], cases[3][1]
    print("\n=== DID THE N-1 ACTION HELP? ===")
    for bus in ("B7", "B8", "B9"):
        idx = np.flatnonzero(before.buses["name"] == bus)[0]
        print(f"{bus}: {abs(before.v_0[idx]):.4f} -> {abs(after.v_0[idx]):.4f} pu")
    old_names, old_sf, old_st = flows(before)
    new_names, new_sf, new_st = flows(after)
    old = {n: max(abs(a), abs(b)) for n, a, b in zip(old_names, old_sf, old_st)}
    print("Line MVA changes after the action (negative means lower loading)")
    for name, a, b in zip(new_names, new_sf, new_st):
        print(f"{name}: {max(abs(a), abs(b)) - old[name]:+.1f} MVA")
    print("Check both voltages and line loading. Improvement in one does not guarantee both.")


def scan(data, outage, title):
    """Increase load and solve a NEW power flow at each point, not full CPF."""
    print(f"\n=== LOAD SCAN: {title} ===")
    print("Factor 1.00 is the original load. Factor 1.10 means 10% more P and Q.")
    print("G1, G2 and G4 each take 1/4 of the added load.")
    print("G3 takes the rest and the change in losses.")
    print("Factor  B8 [pu]  G3 [MW]  Largest generator [MVA]  Most loaded line")
    base_p = sum(row[2] for row in data["loads"][1:])
    factors, voltages, first_limit = [], [], None

    for step in range(26):  # 1.00, 1.02, ... 1.50
        factor = 1 + step*0.02
        case = copy.deepcopy(data)
        for load in case["loads"][1:]:
            load[2] *= factor  # P in MW
            load[3] *= factor  # Q in MVAr, same power factor
        for i in (1, 2, 4):  # G1, G2, G4. Row 0 is the header.
            case["generators"]["GEN"][i][4] += (factor-1)*base_p/4

        ps = solve(case, outage)
        if ps is None:
            print(f"Stop at {factor:.2f}: no solution found in 30 iterations.")
            print("This is not a calculated voltage-collapse point.")
            break
        gen = ps.gen["GEN"]
        gen_s = ps.load_flow_soln[gen]
        if np.any(abs(gen_s) > gen.par["S_n"]):
            print(f"Stop at {factor:.2f}: a generator exceeds its MVA rating.")
            break

        names, sf, st = flows(ps)
        ratios = [(name, 100*max(abs(a), abs(b))/LINE_LIMITS[name])
                  for name, a, b in zip(names, sf, st)]
        name, pct = max(ratios, key=lambda pair: pair[1])
        if pct > 100 and first_limit is None:
            first_limit = factor
            print(f"At {factor:.2f}: {name} exceeds our assumed line limit.")
        b8 = np.flatnonzero(ps.buses["name"] == "B8")[0]
        v8 = abs(ps.v_0[b8])
        print(f"{factor:5.2f}    {v8:7.4f}  {gen_s[2].real:7.1f}"
              f"          {max(abs(gen_s)):7.1f}          {name} {pct:5.1f}%")
        factors.append(factor)
        voltages.append(v8)

    if first_limit is not None:
        print(f"From factor {first_limit:.2f}, later curve points exceed our"
              " assumed line limit.")
    return factors, voltages


def main():
    data = model_data.load()  # Same data loading as your dynamic baseline.
    print("Kundur data from k2a_course.py. TOPS AC Newton-Raphson power flow.")
    print(f"System base {data['base_mva']} MVA. Slack bus {data['slack_bus']}.")
    print("The 0.95 pu threshold and line limits are assumptions, not equipment data.")
    print(f"Transfer test: shift {TRANSFER_CHANGE:.0f} MW of production to area 1.")
    print("N-1 action: shift the same amount to area 2 instead, relative to the base case.")
    print("Loads stay unchanged in these four cases. G3 balances changes in losses.")
    cases = []
    for title, removed, transfer in (
            ("Base", None, 0),
            ("Higher transfer", None, TRANSFER_CHANGE),
            ("N-1", "L7-8-1", 0),
            ("N-1 with redispatch", "L7-8-1", -TRANSFER_CHANGE)):
        ps = solve(data, removed, transfer)
        if ps is None:
            raise RuntimeError(f"Power flow failed for {title}")
        show_case(title, ps)
        cases.append((title, ps))
    compare_cases(cases)

    plt.figure(figsize=(7, 4))
    for title, outage in (("all lines", None), ("N-1 L7-8-1", "L7-8-1")):
        factors, voltages = scan(data, outage, title)
        plt.plot(factors, voltages, "o-", label=title)
    plt.axhline(V_MIN, color="gray", ls="--", label="chosen 0.95 pu")
    plt.xlabel("Load factor at B7 and B9")
    plt.ylabel("B8 voltage [pu]")
    plt.title("Stepwise P-V scan, not a full nose curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
