"""Central frequency-based AGC (aFRR) for TOPS.

Signal chain:  COI frequency deviation -> low-pass filter (T_f)
               -> PI regulator (K_p, K_i) -> total command dP_AGC (MW)
               -> participation factors -> governor bias / HVDC p_ref

The AGC is a TOPS model (DAEModel), so its states are part of the system
state vector and are included in the linearization / modal analysis.
"""

import types

import numpy as np

from tops.dyn_models.blocks import PIRegulator, TimeConstant
from tops.dyn_models.utils import DAEModel


# =============================================================================
# AGC CONTROLLER
# =============================================================================

class AGC(DAEModel):
    """Low-pass filter (T_f) followed by a PI regulator (K_p, K_i).
    Input: frequency deviation (Hz). Output: dP_AGC (MW), positive = more generation.
    """

    def add_blocks(self):
        p = self.par
        # Filter removes fast oscillations, so the AGC follows the slow trend.
        self.filter = TimeConstant(T=p["T_f"])
        self.filter.input = lambda x, v: self.input(x, v)
        # Control error e = 0 - df: low frequency gives more generation.
        self.pi = PIRegulator(K_p=p["K_p"], K_i=p["K_i"])
        self.pi.input = lambda x, v: -self.filter.output(x, v)
        self.output = self.pi.output


# =============================================================================
# MODEL SETUP
# =============================================================================

def user_model_library():
    """Make the AGC class visible to TOPS as model category 'agc'."""
    return types.SimpleNamespace(agc=types.SimpleNamespace(AGC=AGC))


def add_agc_to_model(model, T_f, K_p, K_i, name="AGC1"):
    """Add one AGC unit to the model dictionary (before PowerSystemModel)."""
    model["agc"] = {"AGC": [["name", "T_f", "K_p", "K_i"], [name, T_f, K_p, K_i]]}


def regulating_strength_mw_per_hz(model):
    """Droop contribution P_n / (R f0) per governor type (MW/Hz)."""
    f0 = float(model["f"])
    gen_table = model["generators"]["GEN"]
    h = list(gen_table[0])
    rating = {row[h.index("name")]: float(row[h.index("S_n")]) for row in gen_table[1:]}

    strength = {}
    for gov_type, table in model["gov"].items():
        g = list(table[0])
        strength[gov_type] = sum(
            rating[row[g.index("gen")]] / (float(row[g.index("R")]) * f0) for row in table[1:]
        )
    return strength


# =============================================================================
# CONNECTION OF MEASUREMENT AND SET-POINTS
# =============================================================================

def connect_agc(agc, ps, hydro_units=None, thermal_units=(),
                hvdc_share=0.0, hvdc_link_split=None):
    """Connect the AGC input (COI frequency) and the unit set-points.

    hydro_units:     HYGOV generators in the AGC (None = all)
    thermal_units:   TGOV1 generators in the AGC (empty = none)
    hvdc_share:      share of dP_AGC sent to HVDC (0 to 1)
    hvdc_link_split: split of the HVDC share between links (sum = 1)
    """
    gen = ps.gen["GEN"]
    gen_names = np.asarray(gen.par["name"], dtype=str)
    gen_rating = gen.par["S_n"] * gen.par["N_par"]
    gen_p_base = gen_rating * gen.par["PF_n"]
    gen_idx = {name: i for i, name in enumerate(gen_names)}

    # Participation factors: generator share split in proportion to rating.
    if hydro_units is None:
        hydro_units = list(np.asarray(ps.gov["HYGOV"].par["gen"], dtype=str))
    units = list(hydro_units) + list(thermal_units)
    total_rating = sum(gen_rating[gen_idx[g]] for g in units)
    participation_gen = {g: (1.0 - hvdc_share) * gen_rating[gen_idx[g]] / total_rating for g in units}
    participation_hvdc = {k: hvdc_share * s for k, s in (hvdc_link_split or {}).items()}

    # Measurement: COI frequency deviation in Hz.
    weights = gen.par["H"] * gen_rating
    agc.input = lambda x, v: np.array([ps.f_n * np.average(gen.speed(x, v), weights=weights)])

    # Governors: d_bias = R * dP / p_base (TGOV1), R * dP / (A_t * p_base) (HYGOV).
    for gov_type in ("HYGOV", "TGOV1"):
        gov = ps.gov[gov_type]
        names = np.asarray(gov.par["gen"], dtype=str)
        alpha = np.array([participation_gen.get(n, 0.0) for n in names])
        p_base = gen_p_base[[gen_idx[n] for n in names]]
        R = gov.par["R"]

        if gov_type == "HYGOV":
            scale = R * alpha / (gov.par["A_t"] * p_base)
            # Original TOPS input: -speed + bias - R * c
            gov.time_constant_1.input = lambda x, v, gov=gov, R=R, scale=scale: (
                -gov.input(x, v) + gov.int_par["bias"] + scale * agc.output(x, v) - R * gov.c(x, v)
            )
        else:
            scale = R * alpha / p_base
            # Original TOPS input: -speed + bias
            gov.droop.input = lambda x, v, gov=gov, scale=scale: (
                -gov.input(x, v) + gov.int_par["bias"] + scale * agc.output(x, v)
            )

    # HVDC: p_ref = scheduled p_ref + alpha * dP / S_n (positive = more import).
    # The scheduled value is read every call, so vsc.set_input still works.
    vsc = ps.vsc["VSC_SI"]
    alpha = np.array([participation_hvdc.get(n, 0.0) for n in np.asarray(vsc.par["name"], dtype=str)])
    vsc.p_ref = lambda x, v: vsc._input_values["p_ref"] + alpha * agc.output(x, v) / vsc.par["S_n"]

    return participation_gen, participation_hvdc
