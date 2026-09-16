import numpy as np

from tops.dyn_models.blocks_new import (
    DAEModel,
    PIRegulator,
    TimeConstant,
)
from tops.dyn_models.pll import PLL1

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
    Instantiate:
    'vsc': {
            'VSC_PQ': [
                ['name', 'bus', 'S_n', 'p_ref', 'q_ref',  'k_p', 'k_q', 'T_p', 'T_q', 'k_pll','T_pll', 'T_i', 'i_max'],
                ['VSC1', 'B1',    50,     1,       0,       1,      1,    0.1,   0.1,     5,      1,      0.01,    1.2],
            ],
        }
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    # region Definitions

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], -self.par['q_ref']*self.par['S_n']

    def int_par_list(self):
        return ['f']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    # endregion

    def state_list(self):
        """
        All states in pu.

        i_d: d-axis current, first-order approximation of EM dynamics
        i_q: q-axis current -------------""--------------
        x_p: p-control integral
        x_q: q-control integral
        x_pll: pll q-axis integral
        angle: pll angle
        """
        return ['i_d', 'i_q', 'x_p', 'x_q', 'x_pll', 'angle']

    def input_list(self):
        """
        All outputs in pu.

        p_ref: outer loop active power setpoint
        q_ref: outer loop reactive power setpoint
        """
        return ['p_ref', 'q_ref']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        dp = self.p_ref(x, v) - self.p_e(x, v)
        dq = -self.q_ref(x, v) + self.q_e(x, v)

        # Use limited local copies of the integral states. Never overwrite X
        # inside the derivative function, since X is a view into the solver's
        # global state vector.
        x_p_limited = np.clip(X['x_p'], -par['i_max'], par['i_max'])
        x_q_limited = np.clip(X['x_q'], -par['i_max'], par['i_max'])

        i_d_ref = dp * par['k_p'] + x_p_limited
        i_q_ref = dq * par['k_q'] + x_q_limited

        # Limiters
        i_ref = i_d_ref + 1j * i_q_ref
        i_ref = (
            i_ref
            * par['i_max']
            / np.maximum(par['i_max'], np.abs(i_ref))
        )

        dx_p_unlimited = par['k_p'] / par['T_p'] * dp
        dx_q_unlimited = par['k_q'] / par['T_q'] * dq

        # Conditional integration prevents further windup when an integral
        # state is at a limit, while still allowing it to move back inward.
        block_x_p = (
            ((X['x_p'] >= par['i_max']) & (dx_p_unlimited > 0))
            | ((X['x_p'] <= -par['i_max']) & (dx_p_unlimited < 0))
        )
        block_x_q = (
            ((X['x_q'] >= par['i_max']) & (dx_q_unlimited > 0))
            | ((X['x_q'] <= -par['i_max']) & (dx_q_unlimited < 0))
        )

        dX['i_d'][:] = 1 / (par['T_i']) * (i_ref.real - X['i_d'])
        dX['i_q'][:] = 1 / (par['T_i']) * (i_ref.imag - X['i_q'])
        dX['x_p'][:] = np.where(block_x_p, 0.0, dx_p_unlimited)
        dX['x_q'][:] = np.where(block_x_q, 0.0, dx_q_unlimited)
        dX['x_pll'][:] = par['k_pll'] / (par['T_pll']) * (self.v_q(x,v))
        dX['angle'][:] = X['x_pll']+par['k_pll']*self.v_q(x,v)
        #dX['angle'][:] = 0
        return

    def init_from_load_flow(self, x_0, v_0, S):
        X = self.local_view(x_0)

        self._input_values['p_ref'] = self.par['p_ref']
        self._input_values['q_ref'] = self.par['q_ref']

        v0 = v_0[self.bus_idx_red['terminal']]

        X['i_d'] = self.par['p_ref']/abs(v0)
        X['i_q'] = self.par['q_ref']/abs(v0)
        X['x_p'] = X['i_d']
        X['x_q'] = X['i_q']
        X['x_pll'] = 0
        X['angle'] = np.angle(v0)

    def current_injections(self, x, v):
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r

    # region Utility methods
    def i_inj(self, x, v):
        X = self.local_view(x)
        #v_t = self.v_t(x,v)
        return (X['i_d'] + 1j * X['i_q']) * np.exp(1j*X['angle'])

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def s_e(self, x, v):
        # Apparent power in p.u. (generator base units)
        return self.v_t(x, v)*np.conj(self.i_inj(x, v))

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    def p_e(self, x, v):
        return self.s_e(x,v).real

    def q_e(self, x, v):
        return self.s_e(x,v).imag

    # endregion


class VSC_SI(VSC_PQ):
    """
    N45-compatible VSC model while synthetic inertia is disabled.

    The N45 data table contains the ordinary VSC_PQ parameters followed by
    K_SI, T_SI and P_SI_max.  In the supplied N45 cases K_SI is zero for every
    converter, so the physically correct behaviour is identical to VSC_PQ.

    This compatibility class deliberately does not activate synthetic inertia.
    A dedicated implementation is required before using non-zero K_SI values.
    """
    pass


class VSC_PV(DAEModel):
    """
    Instantiate:
    'vsc': {
            'VSC_PV': [
                ['name', 'bus', 'S_n', 'p_ref', 'V', 'k_p', 'k_v', 'T_p', 'T_v', 'k_pll', 'T_pll', 'T_i', 'i_max'],
                ['VSC1', 'B1',    50,   0.8,    0.93,    1,      1,   0.1,   0.1,     5,      1,      0.01,    1.2],
            ],
        }
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    # region Definitions

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], self.par['V']

    def int_par_list(self):
        return ['f']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    # endregion

    def state_list(self):
        """
        All states in pu.

        i_d: d-axis current, first-order approximation of EM dynamics
        i_q: q-axis current -------------""--------------
        x_p: p-control integral
        x_v: v-control integral
        x_pll: pll q-axis integral
        angle: pll angle
        """
        return ['i_d', 'i_q', 'x_p', 'x_v', 'x_pll', 'angle']

    def input_list(self):
        """
        All outputs in pu.

        p_ref: outer loop active power setpoint
        v_ref: outer loop voltage setpoint
        """
        return ['p_ref', 'v_ref']

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        dp = self.p_ref(x, v) - self.p_e(x, v)
        dv = self.v_ref(x, v) - np.abs(self.v_t(x, v))

        # Use limited local copies; do not mutate the solver state X here.
        x_p_limited = np.clip(X['x_p'], -par['i_max'], par['i_max'])
        x_v_limited = np.clip(X['x_v'], -par['i_max'], par['i_max'])

        i_d_ref = dp * par['k_p'] + x_p_limited
        i_q_ref = -dv * par['k_v'] - x_v_limited

        # Limiters
        i_ref = i_d_ref + 1j * i_q_ref
        i_ref = (
            i_ref
            * par['i_max']
            / np.maximum(par['i_max'], np.abs(i_ref))
        )

        dx_p_unlimited = par['k_p'] / par['T_p'] * dp
        dx_v_unlimited = par['k_v'] / par['T_v'] * dv

        block_x_p = (
            ((X['x_p'] >= par['i_max']) & (dx_p_unlimited > 0))
            | ((X['x_p'] <= -par['i_max']) & (dx_p_unlimited < 0))
        )
        block_x_v = (
            ((X['x_v'] >= par['i_max']) & (dx_v_unlimited > 0))
            | ((X['x_v'] <= -par['i_max']) & (dx_v_unlimited < 0))
        )

        dX['i_d'][:] = 1 / (par['T_i']) * (i_ref.real - X['i_d'])
        dX['i_q'][:] = 1 / (par['T_i']) * (i_ref.imag - X['i_q'])
        dX['x_p'][:] = np.where(block_x_p, 0.0, dx_p_unlimited)
        dX['x_v'][:] = np.where(block_x_v, 0.0, dx_v_unlimited)
        dX['x_pll'][:] = par['k_pll']/par['T_pll'] * (self.v_q(x,v))
        dX['angle'][:] = X['x_pll']+par['k_pll']*self.v_q(x,v)
        return

    def init_from_load_flow(self, x_0, v_0, S):
        X = self.local_view(x_0)

        self._input_values['p_ref'] = self.par['p_ref']
        self._input_values['v_ref'] = self.par['V']

        v0 = v_0[self.bus_idx_red['terminal']]
        s0 = S[self.bus_idx_red['terminal']]/self.par['S_n']

        X['x_p'] = s0.real/abs(v0)
        X['x_v'] = s0.imag/abs(v0)
        X['i_d'] = X['x_p']
        X['i_q'] = -X['x_v']
        X['x_pll'] = 0
        X['angle'] = np.angle(v0)

    def current_injections(self, x, v):
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r

    # region Utility methods
    def i_inj(self, x, v):
        X = self.local_view(x)
        #v_t = self.v_t(x,v)
        return (X['i_d'] + 1j * X['i_q']) * np.exp(1j*X['angle'])

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def s_e(self, x, v):
        # Apparent power in p.u. (generator base units)
        return self.v_t(x, v)*np.conj(self.i_inj(x, v))

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    def p_e(self, x, v):
        return self.s_e(x,v).real

    def q_e(self, x, v):
        return self.s_e(x,v).imag

    # endregion


class DROOP_VSC_GEN(DAEModel):
    """
    Instantiate:
    'vsc': {
            'DROOP_VSC_GEN': [
                ['name', 'bus', 'S_n', 'p_ref', 'V', 'k_p', 'k_v', 'T_p', 'T_v', 'k_pll', 'T_pll', 'T_i', 'i_max'],
                ['VSC1', 'B1',    50,   0.8,    0.93,    1,      1,   0.1,   0.1,     5,      1,      0.01,    1.2],
            ],
        }

    Replace all dynamic modelling aspects!
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    # region Definitions

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], self.par['V']

    def int_par_list(self):
        return ['f']

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    # endregion

    def state_list(self):
        """
        All states (in pu).
        """
        pass

    def input_list(self):
        """
        All inputs (in pu).
        """
        pass

    def state_derivatives(self, dx, x, v):
        pass

    def init_from_load_flow(self, x_0, v_0, S):
        pass

    def current_injections(self, x, v):
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r

    # region Utility methods
    def i_inj(self, x, v):
        pass

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def s_e(self, x, v):
        # Apparent power in p.u. (generator base units)
        return self.v_t(x, v)*np.conj(self.i_inj(x, v))

    def v_q(self,x,v):
        return (self.v_t(x,v)*np.exp(-1j*self.local_view(x)['angle'])).imag
    def p_e(self, x, v):
        return self.s_e(x,v).real

    def q_e(self, x, v):
        return self.s_e(x,v).imag

    # endregion
