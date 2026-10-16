"""Central frequency-based AGC (aFRR) for TOPS.

Signal chain:

    measured frequency deviation (Hz)
        -> low-pass filter            (TimeConstant, T_f)
        -> control error e = 0 - df   (reference is nominal frequency)
        -> PI regulator               (PIRegulator, K_p and K_i)
        -> total aFRR command dP_AGC  (MW)
        -> participation factors      (share per unit, sum = 1)
        -> set-point of each unit     (governor bias or HVDC p_ref)

The AGC is a TOPS model (DAEModel). Its states are therefore part of the
system state vector, integrated by the same solver as the rest of the
power system and included in the linearization / modal analysis.

Usage in a simulation file:

    model = model_data.load()
    B = sum(regulating_strength_mw_per_hz(model).values())
    add_agc_to_model(model, T_f=2.0, K_p=0.0, K_i=B / T_i)
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_model_library())
    ps.init_dyn_sim()
    agc = ps.agc["AGC"]
    connect_agc(agc, ps, hydro_units=None, thermal_units=[],
                hvdc_share=0.0, hvdc_link_split={"NO_2-GB": 1.0})
"""

import types

import numpy as np

from tops.dyn_models.blocks import PIRegulator, TimeConstant
from tops.dyn_models.utils import DAEModel


# =============================================================================
# AGC CONTROLLER
# =============================================================================

class AGC(DAEModel):
    """Low-pass filter followed by a PI regulator.

    Parameters (from the model data):
        T_f  filter time constant (s)
        K_p  proportional gain (MW/Hz)
        K_i  integral gain (MW/(Hz s))

    Input:  frequency deviation (Hz), connected after initialization.
    Output: total aFRR command dP_AGC (MW). Positive = more generation.
    """

    def add_blocks(self):
        p = self.par

        # Low-pass filter: removes fast oscillations (e.g. the 0.74 Hz mode)
        # so that the AGC only reacts to the slow frequency trend.
        self.filter = TimeConstant(T=p["T_f"])
        self.filter.input = lambda x, v: self.input(x, v)

        # PI regulator acting on the control error e = 0 - df.
        # Low frequency (df < 0) gives a positive error and more generation.
        self.pi = PIRegulator(K_p=p["K_p"], K_i=p["K_i"])
        self.pi.input = lambda x, v: -self.filter.output(x, v)

        self.output = self.pi.output


# =============================================================================
# MODEL SETUP HELPERS
# =============================================================================

def user_model_library():
    """Make the AGC class visible to TOPS as model category 'agc'."""
    return types.SimpleNamespace(agc=types.SimpleNamespace(AGC=AGC))


def add_agc_to_model(model, T_f, K_p, K_i, name="AGC1"):
    """Add one AGC unit to the model dictionary (before PowerSystemModel)."""
    model["agc"] = {
        "AGC": [
            ["name", "T_f", "K_p", "K_i"],
            [name, T_f, K_p, K_i],
        ]
    }


# =============================================================================
# MEASUREMENT
# =============================================================================

def connect_coi_frequency(agc, ps):
    """AGC input = centre-of-inertia frequency deviation (Hz)."""
    gen = ps.gen["GEN"]
    weights = gen.par["H"] * gen.par["S_n"] * gen.par["N_par"]
    f_n = ps.f_n

    agc.input = lambda x, v: np.array(
        [f_n * np.average(gen.speed(x, v), weights=weights)]
    )


# =============================================================================
# SET-POINT CONNECTIONS
# =============================================================================

def _participation_array(gov, participation):
    """Participation factor per governor unit (0 for units not listed)."""
    gen_names = np.asarray(gov.par["gen"], dtype=str)
    alpha = np.array([participation.get(name, 0.0) for name in gen_names])
    return alpha


def _mechanical_base_mw(ps, gov):
    """Active-power base (MW) of the generator behind each governor."""
    gen = ps.gen["GEN"]
    gen_index = {name: i for i, name in enumerate(np.asarray(gen.par["name"], dtype=str))}
    idx = np.array([gen_index[str(name)] for name in gov.par["gen"]])
    return (gen.par["S_n"] * gen.par["N_par"] * gen.par["PF_n"])[idx]


def connect_tgov1(agc, ps, participation):
    """Add the AGC command to the TGOV1 reference (bias).

    TGOV1 steady state: dP = d_bias / R  ->  d_bias = R * dP (p.u.).
    """
    gov = ps.gov["TGOV1"]
    alpha = _participation_array(gov, participation)
    p_base = _mechanical_base_mw(ps, gov)
    R = gov.par["R"]

    def agc_bias(x, v):
        return R * alpha * agc.output(x, v) / p_base

    # Original TOPS input: -speed + bias
    gov.droop.input = lambda x, v: (
        -gov.input(x, v) + gov.int_par["bias"] + agc_bias(x, v)
    )


def connect_hygov(agc, ps, participation):
    """Add the AGC command to the HYGOV reference (bias).

    HYGOV steady state: dP = A_t * d_bias / R  ->  d_bias = R * dP / A_t.
    """
    gov = ps.gov["HYGOV"]
    alpha = _participation_array(gov, participation)
    p_base = _mechanical_base_mw(ps, gov)
    R = gov.par["R"]
    A_t = gov.par["A_t"]

    def agc_bias(x, v):
        return R * alpha * agc.output(x, v) / (A_t * p_base)

    # Original TOPS input: -speed + bias - R * c
    gov.time_constant_1.input = lambda x, v: (
        -gov.input(x, v) + gov.int_par["bias"] + agc_bias(x, v) - R * gov.c(x, v)
    )


def connect_hvdc(agc, ps, participation):
    """Add the AGC command to the active-power reference of VSC units.

    p_ref = p_ref_set + alpha * dP_AGC / S_n   (positive = more import)

    p_ref_set is the scheduled set-point stored in the VSC input values.
    It is read at every call, so a scheduled change (e.g. a reduced HVDC
    import used as a disturbance) can still be applied with
    vsc.set_input("p_ref", value, idx=i).
    """
    vsc = ps.vsc["VSC_SI"]
    names = np.asarray(vsc.par["name"], dtype=str)
    alpha = np.array([participation.get(name, 0.0) for name in names])
    S_n = vsc.par["S_n"]

    vsc.p_ref = lambda x, v: vsc._input_values["p_ref"] + alpha * agc.output(x, v) / S_n


# =============================================================================
# REGULATING STRENGTH AND COMPLETE CONNECTION
# =============================================================================

def regulating_strength_mw_per_hz(model):
    """Droop contribution P_n / (R f0) summed per governor type (MW/Hz).

    Only governors contribute. Wind and HVDC (VSC) have no droop, and the
    loads have no frequency dependence, so they do not add to the result.
    """
    f0 = float(model["f"])
    gen_table = model["generators"]["GEN"]
    header = list(gen_table[0])
    gen_rating = {
        row[header.index("name")]: float(row[header.index("S_n")])
        for row in gen_table[1:]
    }

    strength = {}
    for gov_type, table in model["gov"].items():
        gov_header = list(table[0])
        strength[gov_type] = sum(
            gen_rating[row[gov_header.index("gen")]]
            / (float(row[gov_header.index("R")]) * f0)
            for row in table[1:]
        )
    return strength


def connect_agc(agc, ps, hydro_units=None, thermal_units=(),
                hvdc_share=0.0, hvdc_link_split=None):
    """Connect measurement and all set-points in one call.

    hydro_units:     generator names with HYGOV in the AGC (None = all HYGOV)
    thermal_units:   generator names with TGOV1 in the AGC (empty = none)
    hvdc_share:      share of the AGC command sent to HVDC (0 to 1)
    hvdc_link_split: split of the HVDC share between links, sum = 1

    The generator share (1 - hvdc_share) is split between the listed
    generators in proportion to their rating. Returns the participation
    factors so they can be printed or checked.
    """
    gen = ps.gen["GEN"]
    gen_rating = dict(zip(
        np.asarray(gen.par["name"], dtype=str),
        gen.par["S_n"] * gen.par["N_par"],
    ))

    if hydro_units is None:
        hydro_units = list(np.asarray(ps.gov["HYGOV"].par["gen"], dtype=str))
    units = list(hydro_units) + list(thermal_units)
    total_rating = sum(gen_rating[g] for g in units)

    participation_gen = {
        g: (1.0 - hvdc_share) * gen_rating[g] / total_rating for g in units
    }
    participation_hvdc = {
        link: hvdc_share * split for link, split in (hvdc_link_split or {}).items()
    }

    connect_coi_frequency(agc, ps)
    connect_hygov(agc, ps, participation_gen)
    connect_tgov1(agc, ps, participation_gen)
    connect_hvdc(agc, ps, participation_hvdc)

    return participation_gen, participation_hvdc
