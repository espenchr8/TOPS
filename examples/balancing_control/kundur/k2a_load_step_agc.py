"""K2A load step with conventional FCR and ACE-based AGC (aFRR)."""

import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.ps_models import k2a as model_data


# =============================================================================
# STUDY SETTINGS
# =============================================================================

EVENT_TIME = 1.0
LOAD_NAME = "L2"            # Load at bus B9, area 2
LOAD_STEP_MW = 100.0
LOAD_STEP_MVAR = 0.0
T_END = 300.0               # AGC acts over minutes, so simulate longer than FCR
MAX_STEP = 5e-3
FINAL_AVERAGING_WINDOW_S = 5.0
RESTORATION_BAND_HZ = 1e-3  # Frequency counted as restored within +-1 mHz


# =============================================================================
# AREA DEFINITION
# =============================================================================

# K2A has no 'Area' column, so the areas are defined from the topology.
AREA_BUSES = {
    1: {"B1", "B2", "B5", "B6", "B7"},
    2: {"B3", "B4", "B8", "B9", "B10", "B11"},
}
AREAS = [1, 2]

# Tie lines between the areas. Positive flow = area 1 to area 2 (at B7).
TIE_LINE_NAMES = ["L7-8-1", "L7-8-2"]


# =============================================================================
# AGC SETTINGS
# =============================================================================

AGC_ENABLED = True
AGC_UPDATE_INTERVAL_S = 1.0   # Discrete AGC cycle, as in a real LFC
AGC_K_P = 0.0                 # Proportional gain (MW/MW). 0 = pure integral
AGC_T_I_S = 60.0              # Integral time (s)
AGC_BIAS_SCALE = 1.0          # B = scale x droop stiffness of the area

# HVDC participation (prepared). Share of each area's AGC command sent to the
# HVDC VSC units in that area. Requires VSC units in the model; K2A has none,
# so the share is ignored until VSC units are added.
HVDC_SHARE = 0.0
HVDC_RAMP_MW_PER_S = 50.0


def run_simulation():

    # =========================================================================
    # MODEL LOADING AND INITIALIZATION
    # =========================================================================

    model = model_data.load()
    system_base_mva = float(model["base_mva"])
    nominal_frequency = float(model["f"])

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    gen = ps.gen["GEN"]
    load = ps.loads["Load"]
    line = ps.lines["Line"]
    tgov1 = ps.gov["TGOV1"]
    vsc = ps.vsc.get("VSC_SI") if hasattr(ps, "vsc") else None


    # =========================================================================
    # INITIAL EQUILIBRIUM CHECK
    # =========================================================================

    v_initial = ps.solve_algebraic(0.0, ps.x0)
    dx_initial = ps.state_derivatives(0.0, ps.x0, v_initial)
    max_initial_derivative = float(np.max(np.abs(dx_initial)))
    if not ps.power_flow_ready or max_initial_derivative > 1e-5:
        raise RuntimeError(
            "Invalid initial operating point: "
            f"power_flow_ready={ps.power_flow_ready}, "
            f"max |dx/dt|={max_initial_derivative:.3e}."
        )


    # =========================================================================
    # GENERATOR AND AREA DATA
    # =========================================================================

    bus_area = {
        bus: area for area, buses in AREA_BUSES.items() for bus in buses
    }

    gen_names = np.asarray(gen.par["name"], dtype=str)
    gen_buses = np.asarray(gen.par["bus"], dtype=str)
    rating_mva = (
        np.asarray(gen.par["S_n"], dtype=float)
        * np.asarray(gen.par["N_par"], dtype=float)
    )
    inertia_weights = np.asarray(gen.par["H"], dtype=float) * rating_mva
    mechanical_base_mw = rating_mva * np.asarray(gen.par["PF_n"], dtype=float)
    gen_area = np.asarray([bus_area[bus] for bus in gen_buses], dtype=int)
    gen_index = {name: i for i, name in enumerate(gen_names)}
    area_gen_idx = {a: np.where(gen_area == a)[0] for a in AREAS}

    initial_p_m = np.asarray(gen.P_m(ps.x0, v_initial), dtype=float).copy()


    # =========================================================================
    # GOVERNOR DATA FOR AGC
    # =========================================================================

    # Map each TGOV1 unit to its generator and area.
    tgov_gen_idx = np.asarray(
        [gen_index[str(name)] for name in tgov1.par["gen"]], dtype=int
    )
    tgov_area = gen_area[tgov_gen_idx]
    tgov_R = np.asarray(tgov1.par["R"], dtype=float)
    tgov_base_mw = mechanical_base_mw[tgov_gen_idx]
    tgov_v_max = np.asarray(tgov1.par["V_max"], dtype=float)
    tgov_v_min = np.asarray(tgov1.par["V_min"], dtype=float)

    # The reference (bias) that AGC moves. Steady state for TGOV1:
    # delta_bias = R * delta_P (p.u. of the generator's mechanical base).
    bias_0 = np.asarray(tgov1.int_par["bias"], dtype=float).copy()

    # Available up/down margin per unit (MW), limited by the valve limits.
    p_m0_tgov = initial_p_m[tgov_gen_idx]
    up_margin_mw = (tgov_v_max - p_m0_tgov) * tgov_base_mw
    down_margin_mw = (p_m0_tgov - tgov_v_min) * tgov_base_mw

    # Participation factors within each area, proportional to unit rating.
    tgov_part = np.zeros(len(tgov_gen_idx))
    for a in AREAS:
        in_area = tgov_area == a
        tgov_part[in_area] = tgov_base_mw[in_area] / np.sum(tgov_base_mw[in_area])

    # Frequency bias B (MW/Hz) = droop stiffness of each area:
    # droop gives delta_P = -(P_base / R) * delta_f / f_0.
    bias_B = np.asarray([
        AGC_BIAS_SCALE
        * np.sum(tgov_base_mw[tgov_area == a] / tgov_R[tgov_area == a])
        / nominal_frequency
        for a in AREAS
    ])

    area_up_margin = np.asarray([np.sum(up_margin_mw[tgov_area == a]) for a in AREAS])
    area_down_margin = np.asarray([np.sum(down_margin_mw[tgov_area == a]) for a in AREAS])


    # =========================================================================
    # HVDC DATA FOR AGC (PREPARED)
    # =========================================================================

    hvdc_idx = np.zeros(0, dtype=int)
    hvdc_names = np.zeros(0, dtype=str)
    hvdc_s_n = np.zeros(0)
    hvdc_area = np.zeros(0, dtype=int)
    hvdc_part = np.zeros(0)
    hvdc_p_ref_0 = np.zeros(0)

    if vsc is not None:
        vsc_names = np.asarray(vsc.par["name"], dtype=str)
        hvdc_idx = np.where(~np.char.startswith(vsc_names, "WG"))[0]
        hvdc_names = vsc_names[hvdc_idx]
        hvdc_s_n = np.asarray(vsc.par["S_n"], dtype=float)[hvdc_idx]
        hvdc_area = np.asarray(
            [bus_area[str(b)] for b in np.asarray(vsc.par["bus"], dtype=str)[hvdc_idx]],
            dtype=int,
        )
        hvdc_part = np.zeros(len(hvdc_idx))
        for a in AREAS:
            in_area = hvdc_area == a
            if np.any(in_area):
                hvdc_part[in_area] = hvdc_s_n[in_area] / np.sum(hvdc_s_n[in_area])
        hvdc_p_ref_0 = np.asarray(vsc._input_values["p_ref"], dtype=float)[hvdc_idx].copy()

    # The HVDC share only applies to areas that contain HVDC units.
    area_hvdc_share = np.asarray([
        HVDC_SHARE if np.any(hvdc_area == a) else 0.0 for a in AREAS
    ])

    hvdc_target_mw = np.zeros(len(hvdc_idx))
    hvdc_actual_mw = np.zeros(len(hvdc_idx))


    # =========================================================================
    # TIE-LINE MEASUREMENT
    # =========================================================================

    line_names = np.asarray(line.par["name"], dtype=str)
    tie_line_idx = np.asarray(
        [int(np.where(line_names == name)[0][0]) for name in TIE_LINE_NAMES],
        dtype=int,
    )

    def tie_line_power_mw(x, v):
        return float(np.sum(line.p_from(x, v)[tie_line_idx]) * system_base_mva)

    def area_frequency_deviation_hz(speed):
        return np.asarray([
            nominal_frequency
            * np.average(speed[area_gen_idx[a]], weights=inertia_weights[area_gen_idx[a]])
            for a in AREAS
        ])

    initial_tie_power = tie_line_power_mw(ps.x0, v_initial)


    # =========================================================================
    # LOAD DISTURBANCE
    # =========================================================================

    load_names = np.asarray(load.par["name"], dtype=str)
    matches = np.where(load_names == LOAD_NAME)[0]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one load named {LOAD_NAME}.")
    load_idx = int(matches[0])
    load_bus_name = str(load.par["bus"][load_idx])
    load_bus_idx = int(load.bus_idx_red["terminal"][load_idx])
    initial_load_voltage = complex(v_initial[load_bus_idx])

    # Constant-impedance load: delta_Y = conj(delta_S) / |V_0|^2.
    delta_s_pu = (LOAD_STEP_MW + 1j * LOAD_STEP_MVAR) / system_base_mva
    delta_y = np.conj(delta_s_pu) / abs(initial_load_voltage) ** 2
    initial_load_power_mw = (
        np.asarray(load.p(ps.x0, v_initial), dtype=float) * system_base_mva
    )


    # =========================================================================
    # MODEL AND AGC OVERVIEW
    # =========================================================================

    print("\nInitialized K2A model overview")
    print("------------------------------")
    print("Power flow ready:", ps.power_flow_ready)
    print("Synchronous generators:", gen.n_units)
    print("TGOV1 thermal units:", tgov1.n_units)
    print("HVDC VSC units:", len(hvdc_idx))
    print("Dynamic states:", ps.n_states)
    print("Maximum initial derivative:", f"{max_initial_derivative:.6e}")
    print(f"Initial tie-line flow (area 1 to 2): {initial_tie_power:.2f} MW")

    print("\nAGC settings")
    print("------------")
    print("AGC enabled:", AGC_ENABLED)
    print(f"Update interval: {AGC_UPDATE_INTERVAL_S:.1f} s, K_p = {AGC_K_P}, T_i = {AGC_T_I_S:.0f} s")
    for a, b, up, dn, sh in zip(AREAS, bias_B, area_up_margin, area_down_margin, area_hvdc_share):
        print(f"  Area {a}: B = {b:7.1f} MW/Hz, margin +{up:.0f}/-{dn:.0f} MW, HVDC share = {sh:.2f}")

    print(
        f"\nApplying nominal +{LOAD_STEP_MW:.1f} MW at {LOAD_NAME}, "
        f"bus {load_bus_name}, t={EVENT_TIME:.3f} s."
    )


    # =========================================================================
    # NUMERICAL SOLVER AND RESULT STORAGE
    # =========================================================================

    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0.0,
        ps.x0,
        T_END,
        max_step=MAX_STEP,
    )

    ace = np.zeros(len(AREAS))
    ace_integral = np.zeros(len(AREAS))
    agc_command_mw = np.zeros(len(AREAS))
    next_agc_time = 0.0

    initial_speed = np.asarray(gen.speed(ps.x0, v_initial), dtype=float).copy()
    time_values = [0.0]
    speed_values = [initial_speed]
    area_df_values = [area_frequency_deviation_hz(initial_speed)]
    tie_values = [initial_tie_power]
    ace_values = [ace.copy()]
    agc_values = [agc_command_mw.copy()]
    p_m_values = [initial_p_m.copy()]
    load_power_values = [initial_load_power_mw.copy()]
    hvdc_values = [hvdc_actual_mw.copy()]


    # =========================================================================
    # TIME-DOMAIN SIMULATION
    # =========================================================================

    event_applied = False
    next_progress = 10
    t_wall_start = time.perf_counter()

    while solver.t < T_END:
        # Apply the admittance step once, at the selected event time.
        if solver.t >= EVENT_TIME and not event_applied:
            load.y_load[load_idx] += delta_y
            ps.y_bus_red_mod[load_bus_idx, load_bus_idx] += delta_y
            solver.v[:] = ps.solve_algebraic(solver.t, solver.y)
            event_applied = True

            actual_step = (
                float(load.p(solver.y, solver.v)[load_idx]) * system_base_mva
                - initial_load_power_mw[load_idx]
            )
            print("Actual immediate load increase:", f"{actual_step:.3f} MW")

        # ---------------------------------------------------------------------
        # AGC cycle: measure, compute ACE, integrate, distribute to units.
        # ---------------------------------------------------------------------
        if AGC_ENABLED and solver.t >= next_agc_time - 1e-9:
            speed_now = np.asarray(gen.speed(solver.y, solver.v), dtype=float)
            df = area_frequency_deviation_hz(speed_now)
            d_tie = tie_line_power_mw(solver.y, solver.v) - initial_tie_power

            # ACE_1 = dP_tie + B_1 df_1, ACE_2 = -dP_tie + B_2 df_2 (MW).
            ace = np.asarray([d_tie, -d_tie]) + bias_B * df

            # PI regulator with anti-windup: the integrator is frozen in an
            # area whose command would exceed the available margin.
            ace_integral_new = ace_integral + ace * AGC_UPDATE_INTERVAL_S
            command_new = -(AGC_K_P * ace + ace_integral_new / AGC_T_I_S)
            command_limited = np.clip(command_new, -area_down_margin, area_up_margin)
            saturated = command_limited != command_new
            ace_integral = np.where(saturated, ace_integral, ace_integral_new)
            agc_command_mw = command_limited

            # Generator share -> TGOV1 reference (bias).
            for k in range(len(tgov_gen_idx)):
                a_pos = AREAS.index(int(tgov_area[k]))
                unit_mw = (1.0 - area_hvdc_share[a_pos]) * tgov_part[k] * agc_command_mw[a_pos]
                tgov1.int_par["bias"][k] = bias_0[k] + tgov_R[k] * unit_mw / tgov_base_mw[k]

            # HVDC share -> target for the ramp-limited p_ref below.
            for j in range(len(hvdc_idx)):
                a_pos = AREAS.index(int(hvdc_area[j]))
                hvdc_target_mw[j] = area_hvdc_share[a_pos] * hvdc_part[j] * agc_command_mw[a_pos]

            next_agc_time += AGC_UPDATE_INTERVAL_S

        # HVDC: move p_ref towards the AGC target with a ramp limit.
        if len(hvdc_idx) > 0:
            max_change = HVDC_RAMP_MW_PER_S * MAX_STEP
            hvdc_actual_mw += np.clip(hvdc_target_mw - hvdc_actual_mw, -max_change, max_change)
            vsc.set_input(
                "p_ref",
                hvdc_p_ref_0 + hvdc_actual_mw / hvdc_s_n,
                idx=hvdc_idx,
            )

        solver.step()

        speed_now = np.asarray(gen.speed(solver.y, solver.v), dtype=float).copy()
        time_values.append(float(solver.t))
        speed_values.append(speed_now)
        area_df_values.append(area_frequency_deviation_hz(speed_now))
        tie_values.append(tie_line_power_mw(solver.y, solver.v))
        ace_values.append(ace.copy())
        agc_values.append(agc_command_mw.copy())
        p_m_values.append(np.asarray(gen.P_m(solver.y, solver.v), dtype=float).copy())
        load_power_values.append(
            np.asarray(load.p(solver.y, solver.v), dtype=float) * system_base_mva
        )
        hvdc_values.append(hvdc_actual_mw.copy())

        # Terminal progress indicator; this does not affect the simulation.
        progress = int(100 * solver.t / T_END)
        if progress >= next_progress:
            print(f"Simulation progress: {min(progress, 100)}%")
            next_progress += 10

    wall_time = time.perf_counter() - t_wall_start


    # =========================================================================
    # RESULT PROCESSING
    # =========================================================================

    t = np.asarray(time_values)
    speed = np.asarray(speed_values)
    area_df = np.asarray(area_df_values)
    tie_deviation = np.asarray(tie_values) - initial_tie_power
    ace_hist = np.asarray(ace_values)
    agc_hist = np.asarray(agc_values)
    p_m = np.asarray(p_m_values)
    load_power_mw = np.asarray(load_power_values)
    hvdc_hist = np.asarray(hvdc_values).reshape(len(t), len(hvdc_idx))

    coi_frequency = nominal_frequency * (
        1.0 + np.average(speed, axis=1, weights=inertia_weights)
    )
    area_frequency = nominal_frequency + area_df

    # Mechanical power change per area (FCR + AGC), in MW.
    p_m_change_mw = (p_m - initial_p_m[np.newaxis, :]) * mechanical_base_mw[np.newaxis, :]
    area_mech_response = np.stack(
        [np.sum(p_m_change_mw[:, area_gen_idx[a]], axis=1) for a in AREAS], axis=1
    )

    actual_load_increase = load_power_mw[:, load_idx] - initial_load_power_mw[load_idx]


    # =========================================================================
    # RESPONSE METRICS AND TERMINAL SUMMARY
    # =========================================================================

    post_event = t >= EVENT_TIME
    post_idx = np.where(post_event)[0]
    nadir_idx = int(post_idx[np.argmin(coi_frequency[post_event])])

    outside_band = post_event & (np.abs(coi_frequency - nominal_frequency) > RESTORATION_BAND_HZ)
    if np.any(outside_band):
        restoration_time = float(t[np.where(outside_band)[0][-1]] - EVENT_TIME)
    else:
        restoration_time = 0.0

    final_window = t >= T_END - FINAL_AVERAGING_WINDOW_S

    print("\nResult")
    print("------")
    print("COI-frequency nadir:", f"{coi_frequency[nadir_idx]:.5f} Hz")
    print("Time to nadir:", f"{t[nadir_idx] - EVENT_TIME:.3f} s")
    print(
        "Mean COI frequency in final 5 s:",
        f"{np.mean(coi_frequency[final_window]):.5f} Hz",
    )
    print(
        f"Last time outside +-{RESTORATION_BAND_HZ*1e3:.0f} mHz:",
        f"{restoration_time:.1f} s after the event",
    )
    print(
        "Mean tie-line deviation in final 5 s:",
        f"{np.mean(tie_deviation[final_window]):+.2f} MW",
    )
    for k, a in enumerate(AREAS):
        print(
            f"  Area {a}: final ACE = {np.mean(ace_hist[final_window, k]):+.2f} MW, "
            f"AGC command = {np.mean(agc_hist[final_window, k]):+.2f} MW, "
            f"mechanical response = {np.mean(area_mech_response[final_window, k]):+.2f} MW"
        )
    for name, values in zip(hvdc_names, hvdc_hist.T):
        print(f"  {name}: final HVDC change = {np.mean(values[final_window]):+.2f} MW")

    print("\nRuntime")
    print("-------")
    print(f"Wall-clock time: {wall_time:.1f} s")
    print(f"Real-time factor: {wall_time / T_END:.2f} (wall time / simulated time)")

    return {
        "time": t,
        "nominal_frequency": nominal_frequency,
        "coi_frequency": coi_frequency,
        "area_frequency": area_frequency,
        "tie_deviation": tie_deviation,
        "ace": ace_hist,
        "agc_command": agc_hist,
        "area_mech_response": area_mech_response,
        "actual_load_increase": actual_load_increase,
        "hvdc_change": hvdc_hist,
        "hvdc_names": hvdc_names,
        "nadir": float(coi_frequency[nadir_idx]),
        "nadir_time": float(t[nadir_idx]),
    }


def plot_results(results):

    # =========================================================================
    # FIGURE SETUP
    # =========================================================================

    t = results["time"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 12))
    fig.suptitle(
        "Kundur two-area system – FCR and ACE-based AGC\n"
        f"Nominal +{LOAD_STEP_MW:.0f} MW load step at {LOAD_NAME} (area 2), "
        f"t = {EVENT_TIME:.1f} s, AGC T_i = {AGC_T_I_S:.0f} s",
        y=0.985,
    )

    # =========================================================================
    # SYSTEM AND AREA FREQUENCIES
    # =========================================================================

    axes[0].plot(t, results["coi_frequency"], "k", lw=2, label="System COI")
    for k, a in enumerate(AREAS):
        axes[0].plot(t, results["area_frequency"][:, k], lw=1.2, label=f"Area {a}")
    axes[0].axhline(results["nominal_frequency"], color="gray", ls=":", label="Nominal")
    axes[0].plot(
        results["nadir_time"],
        results["nadir"],
        "ko",
        label=f"Nadir {results['nadir']:.3f} Hz",
    )
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[0].legend(ncol=3)

    # =========================================================================
    # TIE-LINE FLOW
    # =========================================================================

    axes[1].plot(t, results["tie_deviation"], color="tab:red", label="Tie line area 1 → 2")
    axes[1].axhline(0.0, color="gray", ls=":")
    axes[1].set_ylabel("$\\Delta P_{tie}$ (MW)")
    axes[1].legend()

    # =========================================================================
    # AREA CONTROL ERROR
    # =========================================================================

    for k, a in enumerate(AREAS):
        axes[2].plot(t, results["ace"][:, k], label=f"ACE area {a}")
    axes[2].axhline(0.0, color="gray", ls=":")
    axes[2].set_ylabel("ACE (MW)")
    axes[2].legend()

    # =========================================================================
    # POWER RESPONSE AND AGC COMMAND
    # =========================================================================

    for k, a in enumerate(AREAS):
        line_obj, = axes[3].plot(
            t, results["area_mech_response"][:, k], label=f"Mechanical response area {a}"
        )
        axes[3].plot(
            t,
            results["agc_command"][:, k],
            ls="--",
            color=line_obj.get_color(),
            label=f"AGC command area {a}",
        )
    for k, name in enumerate(results["hvdc_names"]):
        axes[3].plot(t, results["hvdc_change"][:, k], label=f"HVDC {name}")
    axes[3].axhline(LOAD_STEP_MW, color="gray", ls=":", label="Nominal load step")
    axes[3].axhline(0.0, color="gray", ls=":")
    axes[3].set_ylabel("Power change (MW)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=2)

    # =========================================================================
    # FINAL FIGURE LAYOUT
    # =========================================================================

    for ax in axes:
        ax.axvline(EVENT_TIME, color="black", ls="--", lw=1)
        ax.set_xlim(0.0, T_END)
        ax.grid(True)

    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.065, top=0.91, hspace=0.30
    )
    fig.align_ylabels(axes)
    plt.show()


# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":
    plot_results(run_simulation())
