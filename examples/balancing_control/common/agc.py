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
    add_agc_to_model(model, T_f=2.0, K_p=0.0, K_i=B / T_i)
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_model_library())
    ps.init_dyn_sim()
    agc = ps.agc["AGC"]
    connect_coi_frequency(agc, ps)
    connect_tgov1(agc, ps, {"G1": 0.5, "G2": 0.5})
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
