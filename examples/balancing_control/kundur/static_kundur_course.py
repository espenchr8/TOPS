"""Static Kundur study: base case and one line outage."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data

V_MIN = 0.95  # Study screening criterion, not a documented Kundur grid-code limit.
# Illustrative screening limits for this test case, NOT sourced equipment ratings.
# Chosen above intact flows to test N-1 loading; replace for a real thermal study.
# The data field S_n=100 is an impedance base in TOPS, not a thermal rating.
LINE_LIMITS_MVA = {
    "L5-6": 900, "L6-7": 1500,
    "L7-8-1": 500, "L7-8-2": 500,
    "L8-9-1": 500, "L8-9-2": 500,
    "L9-10": 1500, "L10-11": 900,
}


def show_results(name, ps):
    """Print the quantities used to assess and check the load flow."""
    v = ps.v_0  # Complex bus voltages from TOPS power flow, in pu.
    lines = ps.lines["Line"]
    s_from = lines.s_from(None, v) * ps.s_n  # MVA at the from end.
    s_to = lines.s_to(None, v) * ps.s_n      # MVA at the to end.

    print(f"\n{name}")
    for bus, voltage in zip(ps.buses["name"], v):
        print(f"{bus}: |V| = {abs(voltage):.4f} pu")
    low = [(bus, abs(voltage)) for bus, voltage in zip(ps.buses["name"], v)
           if abs(voltage) < V_MIN]
    print(f"Buses below chosen {V_MIN:.2f} pu criterion: " +
          (", ".join(f"{bus} ({value:.4f})" for bus, value in low) if low else "none"))
    for line, power in zip(lines.par["name"], s_from):
        print(f"{line}: P_from = {power.real:.1f} MW, |S_from| = {abs(power):.1f} MVA")
    for line, power_from, power_to in zip(lines.par["name"], s_from, s_to):
        loading = max(abs(power_from), abs(power_to))
        if line in LINE_LIMITS_MVA:
            limit = LINE_LIMITS_MVA[line]
            print(f"{line}: assumed limit, max |S| {loading:.1f}/{limit:.0f} MVA "
                  f"({100 * loading / limit:.1f}%)")
        else:
            print(f"{line}: max |S| {loading:.1f} MVA; thermal rating not provided")

    # Active power balance: generator output = load + line/transformer losses.
    generation = sum(np.sum(s.real) for s in ps.load_flow_soln.values())
    load = sum(np.sum(m.par["P"]) for m in ps.loads.values())
    line_losses = np.sum((s_from + s_to).real)
    trafo_losses = sum(np.sum(m.p_line(None, v)) * ps.s_n for m in ps.trafos.values())
    losses = line_losses + trafo_losses
    print(f"Balance: {generation:.2f} MW generation = {load:.2f} MW load + {losses:.2f} MW losses")
    print(f"Balance error: {generation - load - losses:.6f} MW")


def main():
    # Same TOPS model and data-loading pattern as in the dynamic baseline.
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.power_flow()
    if not ps.power_flow_ready:
        raise RuntimeError("Base-case power flow did not converge")
    show_results("BASE CASE", ps)

    # A new model is needed because the network changes after a line outage.
    outage_model = copy.deepcopy(model_data.load())
    outage_model["lines"] = [
        line for line in outage_model["lines"]
        if line[0] != "L7-8-1"  # Keep the header and all other lines.
    ]
    ps_outage = dps.PowerSystemModel(model=outage_model)
    ps_outage.power_flow()
    if not ps_outage.power_flow_ready:
        raise RuntimeError("Outage power flow did not converge")
    show_results("N-1: L7-8-1 removed", ps_outage)

    # P-V scan: increase the two loads, shared across four generators.
    # G3 (slack) takes its share plus changes in losses.
    bus8 = np.flatnonzero(ps.buses["name"] == "B8")[0]
    base_load = sum(row[2] for row in model_data.load()["loads"][1:])
    for name, removed in (("All lines in service", None), ("L7-8-1 outage", "L7-8-1")):
        factors, voltages = [], []
        print(f"\nLOAD SCAN: {name}")
        print("Factor   B8 [pu]  G3 [MW]  B7-B8 [MW]  max |S_gen| [MVA]")
        for factor in np.arange(1.0, 1.51, 0.02):
            scan_model = copy.deepcopy(model_data.load())
            for load in scan_model["loads"][1:]:
                load[2] *= factor  # P in MW
                load[3] *= factor  # Q in MVAr, keeping Q/P constant
            # Three specified generators each supply 1/4 of the extra load.
            # The slack generator G3 supplies the remaining 1/4 and changed losses.
            generators = scan_model["generators"]["GEN"]
            for index in (1, 2, 4):  # Rows for G1, G2, G4; row 0 is header.
                generators[index][4] += (factor - 1) * base_load / 4
            if removed is not None:
                scan_model["lines"] = [line for line in scan_model["lines"]
                                       if line[0] != removed]

            scan_ps = dps.PowerSystemModel(model=scan_model)
            scan_ps.pf_max_it = 30
            scan_ps.power_flow()
            if not scan_ps.power_flow_ready:
                print(f"No convergence at {factor:.2f}; exact limit unknown.")
                break

            gen = scan_ps.gen["GEN"]
            gen_s = scan_ps.load_flow_soln[gen]  # MW + j MVAr
            if np.any(abs(gen_s) > gen.par["S_n"] + 1e-6):
                print(f"Generator MVA rating exceeded at {factor:.2f}; scan stops.")
                break

            line = scan_ps.lines["Line"]
            from_s = line.s_from(None, scan_ps.v_0) * scan_ps.s_n
            to_s = line.s_to(None, scan_ps.v_0) * scan_ps.s_n
            if any(max(abs(sf), abs(st)) > LINE_LIMITS_MVA[line_name]
                   for line_name, sf, st in zip(line.par["name"], from_s, to_s)
                   if line_name in LINE_LIMITS_MVA):
                print(f"An assumed line limit exceeded at {factor:.2f}; scan stops.")
                break
            flow = line.s_from(None, scan_ps.v_0).real * scan_ps.s_n
            tie = sum(p for line_name, p in zip(line.par["name"], flow)
                      if line_name in ("L7-8-1", "L7-8-2"))
            voltage = abs(scan_ps.v_0[bus8])
            print(f"{factor:5.2f}    {voltage:7.4f}  {gen_s[2].real:7.1f}"
                  f"      {tie:7.1f}          {max(abs(gen_s)):7.1f}")
            factors.append(factor)
            voltages.append(voltage)
        if factors:
            plt.plot(factors, voltages, "o-", label=name)

    plt.xlabel("Load factor at B7 and B9")
    plt.ylabel("Voltage at B8 [pu]")
    plt.title("Stepwise P-V scan")
    plt.axhline(V_MIN, color="gray", linestyle="--", label="Chosen 0.95 pu criterion")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
