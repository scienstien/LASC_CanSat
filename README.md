# LASC_CanSat — CanSat Descent Analysis — Ground Station

Offline analysis of rocket-recovery descent data. Reads the flight CSV from the
recovered SD card and returns damping, drag area, and a performance assessment.

## Run

```bash
python -m descent.main flight.csv \
       --mass 0.35 --riser 0.50 --diameter 0.30 \
       --bias-file bias_calibration.csv \
       --design-cda 0.15 --out results

python -m descent.main --selftest            # synthetic data, known answers
python -m descent.main --selftest --coning
```

## Stage order

| # | module | does |
|---|---|---|
| 1 | `load.py` | CSV → typed DataFrame, integrity checks |
| 2 | `preprocess.py` | time base → gyro bias → zero-phase filter |
| 3 | `oscillation.py` | PCA → motion class → damping identification |
| 4 | `aerodynamics.py` | density → altitude → velocity → C_D·A → β |
| 5 | `assessment.py` | design intent / consistency / stability |
| 6 | `plot.py` | six figures |

`config.py` holds constants and the run configuration. `synth.py` generates
self-test data.

**Stage 4 consumes stage 3's output.** The C_D·A window is snapped to a whole
number of oscillation periods, and C_D·A is corrected for the oscillation bias
using the measured modulation. Running the two branches independently is
incorrect.

## Debugging

Each stage runs on its own:

```bash
python -m descent.load         flight.csv
python -m descent.preprocess   flight.csv --bias-file bias.csv --out pre.csv
python -m descent.oscillation  flight.csv --bias-file bias.csv
python -m descent.aerodynamics flight.csv --no-oscillation
```

`--deploy-ms <int>` overrides the deployment instant (a raw `t_ms` value) on
any of `main.py`, `preprocess.py`, `oscillation.py`, or `aerodynamics.py`,
instead of inferring it from the accelerometer. It must fall within the
file's `t_ms` range, snaps to the nearest logged sample, and warns (without
stopping) if it leaves under 20% of the record. How deployment was
determined -- inferred or manual, with any snap distance -- is recorded in
`deployment_method` in `results.json`.

Halt the pipeline mid-way, or keep every stage's output:

```bash
python -m descent.main flight.csv --stop-after preprocess
python -m descent.main flight.csv --checkpoint debug/
```

`--checkpoint` writes `stage_1_load.json` … `stage_5_assessment.json` plus the
preprocessed CSV, so a failure in a late stage leaves the earlier ones on disk.

## Input files

**Flight CSV** — required columns:
`t_ms, gx_dps, gy_dps, gz_dps, pressure_pa, temp_c, rh_pct`
optional: `ax_ms2, ay_ms2, az_ms2, gps_alt_m`

`t_ms` is milliseconds since system boot, not since deployment -- it is what
`--deploy-ms` is given in terms of.

Raw `pressure_pa` is required. Air density cannot be reconstructed from an
onboard-derived altitude, and the onboard conversion embeds a fixed
standard-atmosphere temperature that this pipeline replaces with the measured
one — worth ~8% in C_D·A.

**Bias calibration CSV** — a few seconds of gyro output with the vehicle still,
recorded before flight: `gx_dps, gy_dps, gz_dps, temp_c`

## Self-test

Generates a flight whose coefficients are chosen in advance and checks what the
pipeline returns:

| quantity | truth | recovered | error |
|---|---|---|---|
| damping ratio ζ | 0.0300 | 0.0309 | 2.9% |
| quadratic coefficient c | 0.2000 | 0.2144 | 7.2% |
| natural frequency | 0.6200 Hz | 0.6202 Hz | 0.04% |
| terminal speed | 6.2996 m/s | 6.2980 m/s | 0.03% |
| drag area C_D·A | 0.1500 m² | 0.1498 m² | 0.16% |

Run it after any change.

## Before real data

- `--design-cda` from the drogue sizing (criterion 1 is skipped without it)
- bias calibration recorded and its temperature noted
- `--mass`, `--riser`, `--diameter` measured on the recovered vehicle
- `--inertia` if rotational energy is wanted in joules rather than J/(kg·m²)
