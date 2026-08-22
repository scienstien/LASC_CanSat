"""
load.py -- STAGE 1: read the flight CSV off the recovered SD card.

Run this stage alone to check a file before anything else touches it:

    python -m descent.load flight.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Expected columns, with the units the avionics subsystem writes.
COLUMNS = {
    "t_ms":        "timestamp, milliseconds since system boot",
    "gx_dps":      "gyro X, degrees per second",
    "gy_dps":      "gyro Y, degrees per second",
    "gz_dps":      "gyro Z, degrees per second",
    "ax_ms2":      "accel X, m/s^2",
    "ay_ms2":      "accel Y, m/s^2",
    "az_ms2":      "accel Z, m/s^2",
    "pressure_pa": "static pressure, pascals  (RAW -- not onboard altitude)",
    "temp_c":      "temperature, degrees Celsius",
    "rh_pct":      "relative humidity, percent",
    "gps_alt_m":   "GPS altitude, metres (0.0 when no fix)",
}
REQUIRED = ["t_ms", "gx_dps", "gy_dps", "gz_dps",
            "pressure_pa", "temp_c", "rh_pct"]


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a boolean array."""
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best + 1 if best else 0


def load_csv(path: str) -> tuple[pd.DataFrame, dict]:
    """Read the flight CSV with dtypes forced, not inferred.

    CSV carries no type information, so one malformed row can silently turn a
    numeric column into strings and every number downstream becomes wrong
    without an error. Declaring dtypes makes that a failure instead.
    """
    df = pd.read_csv(path, dtype={c: "float64" for c in COLUMNS})

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}\n"
            f"Note: raw pressure_pa is required. Air density cannot be "
            f"reconstructed from an onboard-derived altitude, and the onboard "
            f"conversion embeds a fixed standard-atmosphere temperature."
        )
    if df.empty:
        raise ValueError("CSV contains no rows.")

    report: dict = {"n_rows_raw": int(len(df))}

    # --- timestamp ----------------------------------------------------------
    t = df["t_ms"].to_numpy(dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        warnings.warn("Non-monotonic timestamps -- sorting and de-duplicating.")
        df = (df.sort_values("t_ms").drop_duplicates(subset="t_ms")
                .reset_index(drop=True))
        t = df["t_ms"].to_numpy(dtype=np.float64)
    df["t_s"] = (t - t[0]) * 1e-3           # seconds since the first sample

    # --- unit conversion ----------------------------------------------------
    for ax in ("x", "y", "z"):
        df[f"g{ax}"] = np.radians(df[f"g{ax}_dps"].to_numpy(dtype=np.float64))

    # --- GPS: 0.0 means "no fix", not "sea level" ---------------------------
    if "gps_alt_m" in df:
        fix = df["gps_alt_m"].to_numpy() != 0.0
        df["gps_fix"] = fix
        report["gps_fix_fraction"] = float(fix.mean())

    # --- stale-sensor heuristic --------------------------------------------
    # Firmware re-logs the last valid reading when a read fails, so a dead
    # sensor is indistinguishable from a perfectly steady one. Runs of
    # bit-identical values are the only signature available. This is a
    # heuristic, not a guarantee -- a per-sample staleness flag from firmware
    # would remove the ambiguity.
    report["suspected_stale"] = {}
    for col in ("pressure_pa", "temp_c", "rh_pct"):
        v = df[col].to_numpy(dtype=np.float64)
        same = np.concatenate([[False], v[1:] == v[:-1]])
        n = longest_run(same)
        if n >= 100:
            warnings.warn(f"'{col}': {n} consecutive identical values "
                          f"-- possible stale sensor reads.")
        report["suspected_stale"][col] = int(n)

    # --- sampling -----------------------------------------------------------
    dt = np.diff(df["t_s"].to_numpy())
    report.update({
        "duration_s":   float(df["t_s"].iloc[-1]),
        "fs_median_hz": float(1.0 / np.median(dt)),
        "jitter_pct":   float(100.0 * np.std(dt) / np.median(dt)),
    })
    return df, report


def load_bias(path: str) -> tuple[np.ndarray, float | None]:
    """Pre-flight gyro stillness calibration.

    Measuring bias before flight avoids a circularity: every in-flight
    estimate assumes something about the oscillation -- that it has decayed by
    landing, or that it is zero-mean -- and that behaviour is exactly what the
    mission sets out to measure.

    Expected columns: gx_dps, gy_dps, gz_dps [, temp_c]
    """
    b = pd.read_csv(path)
    bias = np.array([np.radians(b[f"g{a}_dps"].mean()) for a in ("x", "y", "z")])
    temp = float(b["temp_c"].mean()) if "temp_c" in b else None
    return bias, temp


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Stage 1: load and check a flight CSV")
    ap.add_argument("csv")
    ap.add_argument("--bias-file", default=None)
    args = ap.parse_args()

    df, rep = load_csv(args.csv)
    print(json.dumps(rep, indent=2))
    print(f"\ncolumns: {list(df.columns)}")
    print(df.head())
    if args.bias_file:
        bias, temp = load_bias(args.bias_file)
        print(f"\ngyro bias (rad/s): {np.round(bias, 6).tolist()}   at {temp} C")
