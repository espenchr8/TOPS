"""Kundur static study. Run from the TOPS folder with k2a_course.py installed."""

import copy
import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data

from pathlib import Path



# Chosen limits for comparison, not documented operating limits.
SHOW_DETAILS = False  # True prints all buses, lines and scan points.
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
    # Two comparisons keep the effect of each change easy to see.
    groups = (("Line outage and changed generation",
               [cases[0], cases[2], cases[3]]),)
    colors = {"Base": "tab:blue", "Higher transfer": "tab:orange",
              "N-1": "tab:green", "N-1 with changed generation": "tab:red"}
    buses = ("B7", "B8", "B9")
    line_names = list(LINE_LIMITS)
    positions = np.arange(len(line_names))
    for title, group in groups:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        width = 0.8 / len(group)
        for i, (label, ps) in enumerate(group):
            indices = [np.flatnonzero(ps.buses["name"] == bus)[0] for bus in buses]
            axes[0].plot(buses, abs(ps.v_0[indices]), "o-",
                         color=colors[label], label=label)
            names, sf, st = flows(ps)
            loading = {name: 100 * max(abs(a), abs(b)) / LINE_LIMITS[name]
                       for name, a, b in zip(names, sf, st)}
            # No bar is drawn for the disconnected line.
            values = [loading.get(name, np.nan) for name in line_names]
            offset = (i - (len(group) - 1) / 2) * width
            axes[1].bar(positions + offset, values, width,
                        color=colors[label], label=label)
        axes[0].axhline(V_MIN, color="gray", ls="--", label="Chosen 0.95 pu")
        axes[0].set_ylabel("Bus voltage [pu]")
        axes[0].set_title("Voltages at B7, B8 and B9")
        axes[1].axhline(100, color="gray", ls="--", label="Assumed limit")
        axes[1].set_xticks(positions, line_names, rotation=30)
        axes[1].set_ylabel("Use of assumed MVA limit [%]")
        axes[1].set_title("Line loading (missing bar = disconnected line)")
        for ax in axes:
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
        fig.suptitle(title)
        fig.tight_layout()

        # Comment out fig.savefig(...) to disable PNG saving.
        output_file = Path(__file__).resolve().parent / "static_cases.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")

def show_summary(cases):
    """One row per case keeps the changes and results together."""
    print("\n=== 1. PRODUCTION CHANGES AND ONE LINE OUTAGE ===")
    print("Loads stay unchanged. N-1 means L7-8-1 is disconnected.")
    print(f"Higher transfer: G1/G2 each +{TRANSFER_CHANGE/2:.0f} MW, G4 -{TRANSFER_CHANGE/2:.0f} MW.")
    print("N-1 with changed generation reverses that production change.")
    print("G3 balances the remaining demand and losses in every case.")
    print("\nCase                         B8 [pu]  B7->B8 [MW]  Losses [MW]  Highest line use")
    errors = []
    for title, ps in cases:
        names, sf, st = flows(ps)
        tie = sum(a.real for n, a in zip(names, sf) if n in ("L7-8-1", "L7-8-2"))
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
        print("  Below 0.95 pu: " + (", ".join(low) or "none")
              + " | Above assumed line limit: " + (", ".join(over) or "none"))
        gen = ps.gen["GEN"]
        if np.any(abs(ps.load_flow_soln[gen]) > gen.par["S_n"]):
            print("  A generator exceeds its MVA rating.")
    print(f"\nLargest power balance error across these cases: {max(errors):.3e} MW")
    print("The figure compares Base, N-1 and the changed generation after N-1.")



def scan(data, outage, title):
    """Increase load and solve a NEW power flow at each point, not full CPF."""
    print(f"\n=== LOAD SCAN: {title} ===")
    print("Factor 1.00 is the original load. Factor 1.10 means 10% more P and Q.")
    print("G1, G2 and G4 each take 1/4 of the added load.")
    print("G3 takes the rest and the change in losses.")
    if SHOW_DETAILS:
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
        # Apparent power includes both active and reactive power.
        overloaded = np.flatnonzero(abs(gen_s) > gen.par["S_n"])
        if len(overloaded):
            added_load = (factor - 1) * base_p
            print("\nScan stops because a generator exceeds its MVA rating.")
            print(f"Total load increase: {added_load:.1f} MW "
                  f"({100 * (factor - 1):.0f}%)")
            for i in overloaded:
                print(f"{gen.par['name'][i]}: P = {gen_s[i].real:.1f} MW, "
                      f"Q = {gen_s[i].imag:.1f} MVAr")
                print(f"  Apparent power = {abs(gen_s[i]):.1f} MVA")
                print(f"  Generator rating = {gen.par['S_n'][i]:.0f} MVA")
            # Show the first sampled point above the rating as a separate cross.
            # The exact crossing lies between this and the previous scan point.
            b8 = np.flatnonzero(ps.buses["name"] == "B8")[0]
            voltage = abs(ps.v_0[b8])
            plt.plot(added_load, voltage, "rx", ms=8, mew=2,
                     label="First point above generator MVA rating" if not outage else "_nolegend_")
            print("This point is shown as a cross, outside the curve.")
            print("This is a capacity check, not a voltage-collapse limit.")
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
        if SHOW_DETAILS:
            print(f"{factor:5.2f}    {v8:7.4f}  {gen_s[2].real:7.1f}"
                  f"          {max(abs(gen_s)):7.1f}          {name} {pct:5.1f}%")
        factors.append(factor)
        voltages.append(v8)

    if first_limit is not None:
        print(f"From factor {first_limit:.2f}, later curve points exceed our"
              " assumed line limit.")
    if factors:
        print(f"Last plotted factor: {factors[-1]:.2f}, B8 voltage: {voltages[-1]:.4f} pu")
    return factors, voltages


def main():
    data = model_data.load()  # Same data loading as your dynamic baseline.
    print("Kundur data from k2a_course.py. TOPS AC Newton-Raphson power flow.")
    print(f"System base {data['base_mva']} MVA. Slack bus {data['slack_bus']}.")
    print("The 0.95 pu threshold and line limits are assumptions, not equipment data.")

    cases = []
    for title, removed, transfer in (
            ("Base", None, 0),
            ("Higher transfer", None, TRANSFER_CHANGE),
            ("N-1", "L7-8-1", 0),
            ("N-1 with changed generation", "L7-8-1", -TRANSFER_CHANGE)):
        ps = solve(data, removed, transfer)
        if ps is None:
            raise RuntimeError(f"Power flow failed for {title}")
        if SHOW_DETAILS:
            show_case(title, ps)
        cases.append((title, ps))
    show_summary(cases)
    compare_cases(cases)
    print("\n=== 2. SEPARATE LOAD SCAN ===")
    print("Both loads now increase. Production sharing stays the same as before.")
    print("This scan starts from the original data, without the production shift.")

    plt.figure(figsize=(7, 4.5))
    for title, outage in (("all lines", None), ("N-1 L7-8-1", "L7-8-1")):
        factors, voltages = scan(data, outage, title)
        # Convert the load factors to total added active load in MW.
        base_load = sum(row[2] for row in data["loads"][1:])
        added_load = (np.asarray(factors) - 1) * base_load
        plt.plot(added_load, voltages, "o-", label=title)
    plt.axhline(V_MIN, color="gray", ls="--", label="chosen 0.95 pu")
    plt.xlabel("Total load increase at B7 and B9 [MW]")
    plt.ylabel("B8 voltage [pu]")
    plt.title("Stepwise P–V scan at B8")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Comment out plt.gcf().savefig(...) to disable PNG saving.
    output_file = Path(__file__).resolve().parent / "static_load_scan.png"
    plt.gcf().savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


if __name__ == "__main__":
    main()
