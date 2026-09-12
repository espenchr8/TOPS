from tops.dyn_models.blocks import *
from .pll import PLL1

class VSC(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=self.par['T_pll'], bus=p['bus'])

        self.pi_p = PIRegulator(K_p=p['P_K_p'], K_i=p['P_K_i'])
        self.pi_p.input = lambda x, v: self.P_setp(x, v) - self.P(x, v)

        self.pi_q = PIRegulator(K_p=p['Q_K_p'], K_i=p['Q_K_i'])
        self.pi_q.input = lambda x, v: self.Q_setp(x, v) - self.Q(x, v)

        self.lag_p = TimeConstant(T=p['T_i'])
        self.lag_p.input = self.pi_p.output
        self.lag_q = TimeConstant(T=p['T_i'])
        self.lag_q.input = self.pi_q.output

        self.I_d = self.lag_p.output
        self.I_q = self.lag_q.output

    def I_inj(self, x, v):
        return (self.I_d(x, v) - 1j*self.I_q(x, v))*np.exp(1j*self.pll.output(x, v))

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_d(x, v)

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_q(x, v)

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['P_setp'], -self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        V_0 = v_0[self.bus_idx_red['terminal']]*v_n

        I_d_0 = self.par['P_setp']/(abs(V_0)*np.sqrt(3))
        I_q_0 = self.par['Q_setp']/(abs(V_0)*np.sqrt(3))

        self.pi_p.initialize(
            x_0, v_0, self.lag_p.initialize(x_0, v_0, I_d_0)
        )

        self.pi_q.initialize(
            x_0, v_0, self.lag_q.initialize(x_0, v_0, I_q_0)
        )

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        # self.P(x, v)
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]


class VSC_PQ(DAEModel):
    """
    Grid-following VSC with active- and reactive-power control.

    Expected model-data columns:

    [
        "name",
        "bus",
        "S_n",
        "p_ref",
        "q_ref",
        "k_p",
        "k_q",
        "T_p",
        "T_q",
        "k_pll",
        "T_pll",
        "T_i",
        "i_max",
    ]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        bus_index_dtype = [
            (key, int)
            for key in self.bus_ref_spec()
        ]

        self.bus_idx = np.zeros(
            self.n_units,
            dtype=bus_index_dtype,
        )

        self.bus_idx_red = np.zeros(
            self.n_units,
            dtype=bus_index_dtype,
        )

    def bus_ref_spec(self):
        return {
            "terminal": self.par["bus"],
        }

    def load_flow_pq(self):
        return (
            self.bus_idx["terminal"],
            -self.par["p_ref"] * self.par["S_n"],
            -self.par["q_ref"] * self.par["S_n"],
        )

    def state_list(self):
        return [
            "i_d",
            "i_q",
            "x_p",
            "x_q",
            "x_pll",
            "angle",
        ]

    def input_list(self):
        return [
            "p_ref",
            "q_ref",
        ]

    def int_par_list(self):
        return ["f"]

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        active_power_error = (
            self.p_ref(x, v)
            - self.p_e(x, v)
        )

        reactive_power_error = (
            -self.q_ref(x, v)
            + self.q_e(x, v)
        )

        # Limited local values are used without overwriting the solver state.
        x_p_limited = np.clip(
            X["x_p"],
            -p["i_max"],
            p["i_max"],
        )

        x_q_limited = np.clip(
            X["x_q"],
            -p["i_max"],
            p["i_max"],
        )

        i_d_reference = (
            p["k_p"] * active_power_error
            + x_p_limited
        )

        i_q_reference = (
            p["k_q"] * reactive_power_error
            + x_q_limited
        )

        # Circular converter-current limit.
        current_reference = (
            i_d_reference
            + 1j * i_q_reference
        )

        current_reference_magnitude = np.abs(
            current_reference
        )

        current_reference = (
            current_reference
            * p["i_max"]
            / np.maximum(
                p["i_max"],
                current_reference_magnitude,
            )
        )

        dx_p_unlimited = (
            p["k_p"]
            / p["T_p"]
            * active_power_error
        )

        dx_q_unlimited = (
            p["k_q"]
            / p["T_q"]
            * reactive_power_error
        )

        # Conditional integration provides simple anti-windup.
        block_active_integrator = (
            (
                (X["x_p"] >= p["i_max"])
                & (dx_p_unlimited > 0.0)
            )
            | (
                (X["x_p"] <= -p["i_max"])
                & (dx_p_unlimited < 0.0)
            )
        )

        block_reactive_integrator = (
            (
                (X["x_q"] >= p["i_max"])
                & (dx_q_unlimited > 0.0)
            )
            | (
                (X["x_q"] <= -p["i_max"])
                & (dx_q_unlimited < 0.0)
            )
        )

        dX["i_d"][:] = (
            current_reference.real
            - X["i_d"]
        ) / p["T_i"]

        dX["i_q"][:] = (
            current_reference.imag
            - X["i_q"]
        ) / p["T_i"]

        dX["x_p"][:] = np.where(
            block_active_integrator,
            0.0,
            dx_p_unlimited,
        )

        dX["x_q"][:] = np.where(
            block_reactive_integrator,
            0.0,
            dx_q_unlimited,
        )

        pll_voltage_error = self.v_q(x, v)

        dX["x_pll"][:] = (
            p["k_pll"]
            / p["T_pll"]
            * pll_voltage_error
        )

        dX["angle"][:] = (
            X["x_pll"]
            + p["k_pll"] * pll_voltage_error
        )

    def init_from_load_flow(self, x_0, v_0, S):
        X = self.local_view(x_0)

        self._input_values["p_ref"] = self.par["p_ref"]
        self._input_values["q_ref"] = self.par["q_ref"]

        terminal_voltage = v_0[
            self.bus_idx_red["terminal"]
        ]

        terminal_voltage_magnitude = np.abs(
            terminal_voltage
        )

        X["i_d"][:] = (
            self.par["p_ref"]
            / terminal_voltage_magnitude
        )

        X["i_q"][:] = (
            self.par["q_ref"]
            / terminal_voltage_magnitude
        )

        X["x_p"][:] = X["i_d"]
        X["x_q"][:] = X["i_q"]
        X["x_pll"][:] = 0.0
        X["angle"][:] = np.angle(terminal_voltage)

    def current_injections(self, x, v):
        converter_to_system_base = (
            self.par["S_n"]
            / self.sys_par["s_n"]
        )

        return (
            self.bus_idx_red["terminal"],
            self.i_inj(x, v)
            * converter_to_system_base,
        )

    def i_inj(self, x, v):
        X = self.local_view(x)

        converter_current_dq = (
            X["i_d"]
            + 1j * X["i_q"]
        )

        return (
            converter_current_dq
            * np.exp(1j * X["angle"])
        )

    def v_t(self, x, v):
        return v[
            self.bus_idx_red["terminal"]
        ]

    def s_e(self, x, v):
        return (
            self.v_t(x, v)
            * np.conj(self.i_inj(x, v))
        )

    def p_e(self, x, v):
        return self.s_e(x, v).real

    def q_e(self, x, v):
        return self.s_e(x, v).imag

    def v_q(self, x, v):
        converter_angle = self.local_view(x)["angle"]

        voltage_dq = (
            self.v_t(x, v)
            * np.exp(-1j * converter_angle)
        )

        return voltage_dq.imag


class VSC_SI(VSC_PQ):
    """
    N45-compatible VSC model with synthetic inertia disabled.

    The N45 data contain the ordinary VSC_PQ parameters followed by
    K_SI, T_SI and P_SI_max.

    In the current N45 data sets K_SI is zero. Therefore the converter
    response is identical to VSC_PQ, and the additional SI parameters
    are accepted but not used.

    This class must be extended before non-zero synthetic-inertia gains
    are used.
    """

    pass
