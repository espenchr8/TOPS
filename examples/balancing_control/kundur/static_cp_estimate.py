"""Steady-state estimate with constant-power loads: +100 MW at B9 (same Q/P),
all four generators share the increase (droop, equal ratings).
Delta f = -(total generation increase)/K * f_n."""
import copy
import tops.dynamic as dps
from tops.ps_models import k2a_course as model_data

def solve(step):
    d = copy.deepcopy(model_data.load())
    L = d["loads"]
    h = L[0]
    for r in L[1:]:
        if r[h.index("bus")] == "B9":
            q_ratio = r[h.index("Q")] / r[h.index("P")]
            r[h.index("P")] += step
            r[h.index("Q")] += step * q_ratio
    G = d["generators"]["GEN"]
    pc = G[0].index("P")
    for r in G[1:]:
        if r[0] != "G3":
            r[pc] += step / 4  # G3 (slack) takes its quarter plus extra losses
    ps = dps.PowerSystemModel(model=d)
    ps.power_flow()
    # Total generator injection in MW (B1-B4 hold only generators).
    s_gen = sum(ps.s_0[i] for i in ps.gen["GEN"].bus_idx["terminal"]) * ps.s_n
    return s_gen.real

p0, p1 = solve(0), solve(100)
dpg = p1 - p0
K = 4 * 900 * (1 / 0.05 + 0.02)  # 72 072 MW/pu, as in the report
print(f"Generation increase: {dpg:.1f} MW (100 MW load + {dpg-100:.1f} MW losses)")
print(f"Droop offset: {-dpg/K*50*1000:.1f} mHz, versus -29.3 mHz with constant impedance "
      f"({dpg/42.25:.2f} times larger)")
