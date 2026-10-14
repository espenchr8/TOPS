# Static and Dynamic Study of the Kundur System

TET4575 Part 1. Simulation files for the static and dynamic study of the
Kundur two-area system, built in TOPS.

## Files
| File | What it does |
|---|---|
| `static_kundur_course.py` | Power flow for base case, N-1 (L7-8-1 out) and redispatch. Nose curves with an own Newton–Raphson solver and generator Q-limits. Saves `static_combined.png`. |
| `dynamic_kundur_course.py` | Two 60 s RMS simulations: one without disturbance and one with a 100 MW load step at B9 at t = 5 s. Saves `dynamic_response.png`. |
| `k2a_course.py` | Kundur model data used by both scripts (TOPS format). |

## Requirements
- Python 3 with `numpy` and `matplotlib`
- TOPS: https://github.com/hallvar-h/TOPS
- `k2a_course.py` placed in `tops/ps_models/`

## How to run
Run from the TOPS folder:
```
python static_kundur_course.py
python dynamic_kundur_course.py
```
Both scripts print the results to the terminal and save the figures.

## Main results
- No line is overloaded. Losing L7-8-1 drops B8 to 0.890 pu, below the 0.90 pu limit.
- Voltage collapse margin: 152 MW intact, 70 MW after the outage.
- The 100 MW load step gives a frequency offset of −29.31 mHz, equal to the droop calculation.

## Notes
- Line ratings and the 0.90 pu limit are assumptions, explained in the report.
- The scripts were written with AI assistance and checked against TOPS and hand calculations.
