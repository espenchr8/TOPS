# Static and Dynamic Study of the Kundur System

TET4575 Part 1. Simulation files for the static and dynamic study of the
Kundur two-area system, built in TOPS.

## Files
| File | What it does |
|---|---|
| `static_kundur_course.py` | Power flow for base case, N-1 (L7-8-1 out) and redispatch. Nose curves with an own Newton–Raphson solver and generator Q-limits. Saves `static_combined.png` and `.pdf`. |
| `dynamic_kundur_course.py` | Two 60 s RMS simulations: one without disturbance and one with a 100 MW load step at B9 at t = 5 s. Saves `dynamic_response.png` and `.pdf`. |
| `static_cp_estimate.py` | The same 100 MW step at B9 as a constant-power load in the power flow, shared equally by the four generators. Prints the generation increase (113.2 MW) and the droop offset (−78.5 mHz). |
| `k2a_course.py` | Kundur model data used by all scripts (TOPS format). |

## Requirements
- Python 3 with `numpy` and `matplotlib`
- TOPS: https://github.com/hallvar-h/TOPS
- `k2a_course.py` placed in `tops/ps_models/`

## How to run
Run from the TOPS folder:
```
python static_kundur_course.py
python dynamic_kundur_course.py
python static_cp_estimate.py
```
All scripts print the results to the terminal. The first two also save the figures.

## Main results
- No line is overloaded. Losing L7-8-1 drops B8 to 0.890 pu, below the 0.90 pu limit.
- Distance to voltage collapse: 152 MW intact, 70 MW after the outage. After the outage the operating point already violates 0.90 pu, so the 70 MW is not a usable margin.
- The 100 MW load step gives a frequency offset of −29.31 mHz, equal to the droop calculation.
- With constant-power loads instead of constant impedance, the same step gives −78.5 mHz, 2.7 times larger.

## Notes
- Line ratings and the 0.90 pu limit are assumptions, explained in the report.
- The static study uses constant-power loads. The dynamic study uses constant-impedance loads (TOPS converts the loads after the power flow).
- The scripts were written with AI assistance and checked against TOPS and hand calculations.
