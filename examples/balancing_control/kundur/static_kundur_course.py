"""Static Kundur study: base case and one line outage."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data


def show_results(name, ps):
    """Print the quantities used to assess and check the load flow."""
    v = ps.v_0  # Complex bus voltages from TOPS power flow, in pu.
    lines = ps.lines["Line"]
    s_from = lines.s_from(None, v) * ps.s_n  # MVA at the from end.
    s_to = lines.s_to(None, v) * ps.s_n      # MVA at the to end.

    print(f"\n{name}")
    for bus, voltage in zip(ps.buses["name"], v):
        print(f"{bus}: |V| = {abs(voltage):.4f} pu")
    for line, power in zip(lines.par["name"], s_from):
        print(f"{line}: P_from = {power.real:.1f} MW, |S_from| = {abs(power):.1f} MVA")

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

    # P-V scan: increase both loads with the same factor (P and Q).
    # G1, G2 and G4 keep their P setpoints; slack generator G3 balances the change.
    factors, b8_voltages = [], []
    bus8 = np.flatnonzero(ps.buses["name"] == "B8")[0]
    print("\nLOAD SCAN (intact network)")
    print("Factor    B8 [pu]    G3 [MW]    B7-B8 total [MW]")
    for factor in np.arange(1.0, 1.51, 0.02):
        scan_model = copy.deepcopy(model_data.load())
        for load in scan_model["loads"][1:]:
            load[2] *= factor  # P in MW
            load[3] *= factor  # Q in MVAr: same power factor

        scan_ps = dps.PowerSystemModel(model=scan_model)
        scan_ps.pf_max_it = 30  # More iterations near a difficult operating point.
        scan_ps.power_flow()
        if not scan_ps.power_flow_ready:
            print(f"No convergence at factor {factor:.2f}; the limit is not established.")
            break

        voltage = abs(scan_ps.v_0[bus8])
        gen_power = scan_ps.load_flow_soln[scan_ps.gen["GEN"]].real
        g3 = gen_power[2]  # G3 is connected to slack bus B3.
        line = scan_ps.lines["Line"]
        flow = line.s_from(None, scan_ps.v_0).real * scan_ps.s_n
        tie = sum(p for name, p in zip(line.par["name"], flow)
                  if name in ("L7-8-1", "L7-8-2"))
        print(f"{factor:5.2f}     {voltage:7.4f}    {g3:8.1f}         {tie:8.1f}")
        factors.append(factor)
        b8_voltages.append(voltage)

    plt.plot(factors, b8_voltages, "o-")
    plt.xlabel("Load factor at B7 and B9")
    plt.ylabel("Voltage at B8 [pu]")
    plt.title("Stepwise P-V scan (intact network)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
