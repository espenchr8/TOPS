"""Part 1: static load flow, one line outage, and a simple P-V scan."""

import copy

import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
from tops.ps_models import k2a_course


def solve(load_scale=1.0, outaged_line=None):
    # Each case starts from untouched Kundur data.
    data = copy.deepcopy(k2a_course.load())
    for load in data["loads"][1:]:
        load[2] *= load_scale  # P [MW]
        load[3] *= load_scale  # Q [MVAr], constant power factor
    if outaged_line is not None:
        data["lines"] = [data["lines"][0]] + [
            line for line in data["lines"][1:] if line[0] != outaged_line
        ]

    ps = dps.PowerSystemModel(model=data)
    ps.power_flow()
    if not ps.power_flow_ready:
        raise RuntimeError(f"Power flow failed: scale={load_scale}, outage={outaged_line}")
    return ps


def report(label, ps):
    print(f"\n{label}")
    print("Bus     |V| [pu]   angle [deg]")
    for bus, v in zip(ps.buses["name"], ps.v_0):
        print(f"{bus:5s}   {abs(v):7.4f}      {np.angle(v, deg=True):8.2f}")

    # TOPS returns branch power in pu on the system base (900 MVA).
    lines = ps.lines["Line"]
    s_from = lines.s_from(None, ps.v_0) * ps.s_n
    s_to = lines.s_to(None, ps.v_0) * ps.s_n
    print("Line       P_from [MW]  |S_from| [MVA]  |S_to| [MVA]")
    for name, sf, st in zip(lines.par["name"], s_from, s_to):
        print(f"{name:9s}  {sf.real:10.1f}      {abs(sf):10.1f}      {abs(st):10.1f}")
    print(f"Lowest voltage: {min(abs(ps.v_0)):.4f} pu")


def main():
    base = solve()
    report("BASE CASE", base)

    # One of the parallel tie lines is removed; B7-B8 remains connected.
    outage = solve(outaged_line="L7-8-1")
    report("N-1: L7-8-1 disconnected", outage)

    # Repeated load flows trace the upper branch of a P-V characteristic.
    scales, voltages = [], []
    bus8 = np.flatnonzero(base.buses["name"] == "B8")[0]
    for scale in np.arange(1.0, 2.01, 0.05):
        try:
            ps = solve(load_scale=float(scale))
        except RuntimeError:
            print(f"Last successful scan point before scale {scale:.2f}")
            break
        scales.append(scale)
        voltages.append(abs(ps.v_0[bus8]))

    plt.plot(scales, voltages, "o-")
    plt.xlabel("Load multiplier (P and Q at B7 and B9)")
    plt.ylabel("Voltage at B8 [pu]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()
