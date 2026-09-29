"""Static Kundur study: base case and one line outage."""

import copy

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


if __name__ == "__main__":
    main()
