# Kundur Static and Dynamic Study — TET4575 Part 1

This project studies power flow and dynamic RMS response
in the Kundur two-area system using TOPS.

## Required software

Use a Python environment with TOPS, NumPy and Matplotlib installed.
The model file `k2a_course.py` must be available in `tops.ps_models`.

Both studies use the same model data:
11 buses, four generators, a 900 MVA system base and 50 Hz.

## Run the studies

Activate the Python environment used for TOPS, then run:

```bash
python static_kundur_course.py
python dynamic_kundur_course.py
```

## Static study

The script compares:
- Base case.
- Increased transfer from area 1.
- Outage of L7-8-1.
- Post-outage redispatch.

A separate scan increases both loads' active and reactive power
in steps of 2% of their original values, with and without the outage.

Terminal output reports bus voltages, corridor transfer,
network losses, line-limit violations, generator-rating violations
and the static active-power balance error.

The voltage threshold and line limits are study assumptions.
The load scan does not calculate a voltage-collapse point.

## Dynamic study

The script runs an undisturbed simulation and a B9 load addition
at 5 s. Both simulations last 60 s.

The added admittance represents 50 MW at the pre-event voltage.
Actual load power varies with voltage.

Terminal output reports initialization checks, frequency response,
mechanical-power changes, voltage changes, active-power balances
and the final generator speed-equation residual.

The reported frequency minimum is within the simulated interval.
It does not necessarily represent the final equilibrium.

## Time-step comparison

The default dynamic time step is `DT = 0.005` s.
To repeat the sensitivity check, change it to `DT = 0.0025` s
and rerun the dynamic script.

## Saved plots

The scripts display the plots and save PNG files beside the scripts:

- `static_cases.png`
- `static_load_scan.png`
- `dynamic_response.png`

Comment out the `savefig(...)` lines to disable PNG saving.
Existing PNG files with the same names are overwritten.

## Verification

The report summarizes the main findings.
Running the scripts reproduces the detailed terminal checks.
Small equation residuals check internal consistency;
the time-step comparison separately checks numerical sensitivity.
