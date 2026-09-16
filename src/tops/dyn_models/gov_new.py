import numpy as np

from tops.dyn_models.blocks_new import *
from tops.dyn_models.utils import auto_init
import tops.utility_functions as dps_uf


class GOV:
    def input_list(self):
        return ["input", "P_n_gen"]

    def connections(self):
        return [
            {
                "input": "input",
                "source": {
                    "container": "gen",
                    "mdl": "*",
                    "id": self.par["gen"],
                },
                "output": "speed",
            },
            {
                "input": "P_n_gen",
                "source": {
                    "container": "gen",
                    "mdl": "*",
                    "id": self.par["gen"],
                },
                "output": "P_nom",
            },
            {
                "output": "output",
                "destination": {
                    "container": "gen",
                    "mdl": "*",
                    "id": self.par["gen"],
                },
                "input": "P_m",
            },
        ]


class TGOV1(GOV, DAEModel):
    def add_blocks(self):
        p = self.par

        self.droop = Gain(K=1.0 / p["R"])
        self.time_constant_lim = TimeConstantLims(
            T=p["T_1"],
            V_min=p["V_min"],
            V_max=p["V_max"],
        )
        self.lead_lag = LeadLag(
            T_1=p["T_2"],
            T_2=p["T_3"],
        )
        self.damping_gain = Gain(K=p["D_t"])

        self.droop.input = lambda x, v: (
            -self.input(x, v)
            + self.int_par["bias"]
        )
        self.time_constant_lim.input = self.droop.output
        self.lead_lag.input = self.time_constant_lim.output
        self.damping_gain.input = lambda x, v: self.input(x, v)

        self.output = lambda x, v: (
            self.lead_lag.output(x, v)
            - self.damping_gain.output(x, v)
        )

    def int_par_list(self):
        return ["bias"]

    def init_from_connections(self, x0, v0, output_0):
        self.int_par["bias"] = self.droop.initialize(
            x0,
            v0,
            self.time_constant_lim.initialize(
                x0,
                v0,
                self.lead_lag.initialize(
                    x0,
                    v0,
                    output_0["output"],
                ),
            ),
        )


class HYGOV(GOV, DAEModel):
    """Hydro governor with gate/rate limits and nonlinear water column."""

    _GATE_EPS = 1e-6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Normalize optional HYGOV parameters. This retains compatibility
        # with older data sets that used lower-case gate-limit names.
        optional_parameters = {
            "G_min": ("g_min", 0.0),
            "G_max": ("g_max", 1.0),
            "V_elm": (None, np.inf),
            "D_turb": (None, 0.0),
            "P_N": (None, 0.0),
        }

        for target, (alias, default) in optional_parameters.items():
            if target in self.par.dtype.names:
                continue

            values = np.full(self.n_units, default, dtype=float)
            if alias is not None and alias in self.par.dtype.names:
                values[:] = self.par[alias]

            new_field = np.zeros(
                self.n_units,
                dtype=[(target, float)],
            )
            new_field[target] = values
            self.par = dps_uf.combine_recarrays(
                self.par,
                new_field,
            )

    def int_par_list(self):
        return ["bias"]

    def add_blocks(self):
        p = self.par

        self.time_constant_1 = TimeConstant(T=p["T_f"])
        self.pi_reg = PIRegulator2(
            T_1=p["T_r"],
            T_2=p["T_r"] * p["r"],
        )
        self.time_constant_2 = TimeConstantLimsRate(
            T=p["T_g"],
            V_min=p["G_min"],
            V_max=p["G_max"],
            Rate=p["V_elm"],
        )
        self.integrator = Integrator2(T=p["T_w"])

        self.time_constant_1.input = lambda x, v: (
            -self.input(x, v)
            + self.int_par["bias"]
            - p["R"] * self.c(x, v)
        )
        self.pi_reg.input = self.time_constant_1.output
        self.c = self.pi_reg.output
        self.time_constant_2.input = self.c

        if "backlash" in p.dtype.names:
            self.backlash = Backlash(db=p["backlash"])
            self.backlash.input = self.time_constant_2.output
            self.g = self.backlash.output
        else:
            self.g = self.time_constant_2.output

        self.q = self.integrator.output

        def flow_ratio(x, v):
            gate = np.maximum(self.g(x, v), self._GATE_EPS)
            return self.q(x, v) / gate

        self.div = flow_ratio
        self.h = lambda x, v: self.div(x, v) ** 2
        self.integrator.input = lambda x, v: 1.0 - self.h(x, v)

        def turbine_power(x, v):
            hydraulic_power = p["A_t"] * (
                self.q(x, v) - p["q_nl"]
            ) * self.h(x, v)

            damping_power = (
                p["D_turb"]
                * self.g(x, v)
                * self.input(x, v)
            )

            unscaled_power = hydraulic_power - damping_power

            scale = np.ones(self.n_units)
            use_turbine_rating = p["P_N"] > 0
            np.divide(
                p["P_N"],
                self.P_n_gen(x, v),
                out=scale,
                where=use_turbine_rating,
            )

            return unscaled_power * scale

        self.output = turbine_power

    def init_from_connections(self, x0, v0, output_0):
        """Initialize the positive steady-state HYGOV solution."""

        p = self.par
        desired_output = np.asarray(
            output_0["output"],
            dtype=float,
        )

        scale = np.ones(self.n_units)
        use_turbine_rating = p["P_N"] > 0
        np.divide(
            p["P_N"],
            self.P_n_gen(x0, v0),
            out=scale,
            where=use_turbine_rating,
        )

        turbine_output = desired_output / scale

        q0 = (
            p["q_nl"]
            + turbine_output / p["A_t"]
        )

        outside_limits = (
            (q0 < p["G_min"] - 1e-9)
            | (q0 > p["G_max"] + 1e-9)
        )

        if np.any(outside_limits):
            names = ", ".join(
                map(str, p["name"][outside_limits])
            )
            raise ValueError(
                "HYGOV steady-state gate is outside "
                "[G_min, G_max] for: "
                + names
            )

        g0 = q0.copy()

        self.time_constant_1.initialize(
            x0,
            v0,
            np.zeros(self.n_units),
        )
        self.pi_reg.initialize(x0, v0, g0)
        self.time_constant_2.initialize(x0, v0, g0)
        self.integrator.initialize(x0, v0, q0)

        if hasattr(self, "backlash"):
            self.backlash.initialize(x0, v0, g0)

        self.int_par["bias"][:] = p["R"] * g0


class IEESGO(GOV, DAEModel):
    def int_par_list(self):
        return ["bias"]

    def add_blocks(self):
        p = self.par

        self.lead_lag = LeadLag(
            T_1=p["T_2"],
            T_2=p["T_1"],
        )
        self.time_constant_gain_k1_t3 = TimeConstantGain(
            K=p["K_1"],
            T=p["T_3"],
        )
        self.limiter = Limiter(
            Min=p["P_min"],
            Max=p["P_max"],
        )
        self.time_constant_2 = TimeConstant(T=p["T_4"])
        self.gain_1_minus_k2 = Gain(K=1.0 - p["K_2"])
        self.time_constant_gain_k2_t5 = TimeConstantGain(
            K=p["K_2"],
            T=p["T_5"],
        )
        self.gain_1_minus_k3 = Gain(K=1.0 - p["K_3"])
        self.time_constant_gain_k3_t6 = TimeConstantGain(
            K=p["K_3"],
            T=p["T_6"],
        )

        self.lead_lag.input = lambda x, v: self.input(x, v)
        self.time_constant_gain_k1_t3.input = self.lead_lag.output
        self.limiter.input = lambda x, v: (
            -self.time_constant_gain_k1_t3.output(x, v)
            + self.int_par["bias"]
        )
        self.time_constant_2.input = self.limiter.output
        self.gain_1_minus_k2.input = self.time_constant_2.output
        self.time_constant_gain_k2_t5.input = self.time_constant_2.output
        self.gain_1_minus_k3.input = self.time_constant_gain_k2_t5.output
        self.time_constant_gain_k3_t6.input = (
            self.time_constant_gain_k2_t5.output
        )

        def output_sum(x, v):
            return (
                self.gain_1_minus_k2.output(x, v)
                + self.gain_1_minus_k3.output(x, v)
                + self.time_constant_gain_k3_t6.output(x, v)
            )

        P_n = p["P_N"]
        self.output = lambda x, v: output_sum(x, v) * (
            P_n / self.P_n_gen(x, v)
            + 1.0 * (P_n == 0)
        )

    def init_from_connections(self, x0, v0, output_0):
        auto_init(
            self,
            x0,
            v0,
            output_0["output"],
        )
