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
    """Length of the longest run of True in a boolean array.

    Calibrated for a diff-encoded mask (like `same` below, which compares
    each sample to its predecessor with a leading False): the first sample of
    a run never appears as True in that encoding, so the +1 corrects for it.
    Do NOT call this on a raw True/False mask (e.g. np.isnan(x)) expecting the
    literal run length -- it will overcount by one. Use _run_lengths for that.
    """
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best + 1 if best else 0


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of every contiguous run of True in a raw boolean array."""
    if mask.size == 0:
        return np.array([], dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return ends - starts


def _fill_nan_gaps(df: pd.DataFrame, cols: list[str],
                    max_gap: int) -> tuple[pd.DataFrame, dict]:
    """Bridge short NaN runs; trim unrecoverable edges; reject long interior gaps.

    Stage 2's filtfilt is IIR (recursive): a single NaN anywhere in a channel
    poisons the ENTIRE output, not just the neighbourhood of the gap. NaNs
    have to be resolved here, before anything downstream touches the data --
    there is no fixing this after the filter has run.

    A short gap is safe to bridge with a straight line: the filter cutoff
    (Stage 2) throws away anything faster than a few Hz anyway, so a short
    linear fill is indistinguishable from noise once filtered. A gap longer
    than max_gap is left alone and raises -- drawing a long straight line
    across a real sensor outage would fabricate data, not recover it.
    Leading/trailing runs (nothing to interpolate from) are trimmed instead
    of guessed at.
    """
    df = df.copy()
    report: dict = {}
    for col in cols:
        isnan = np.isnan(df[col].to_numpy(dtype=np.float64))
        n_nan = int(isnan.sum())
        runs = _run_lengths(isnan)
        report[col] = {"n_nan": n_nan,
                        "longest_gap": int(runs.max()) if runs.size else 0,
                        "n_gaps": int(runs.size)}
        if n_nan:
            df[col] = df[col].interpolate(method="linear", limit=max_gap,
                                           limit_area="inside")

    still_bad = df[cols].isna().any(axis=1).to_numpy()
    report["rows_trimmed"] = 0
    if still_bad.any():
        valid = np.flatnonzero(~still_bad)
        if valid.size == 0:
            raise ValueError(f"Columns {cols}: every row has an unrecoverable NaN.")
        first, last = int(valid[0]), int(valid[-1])
        if still_bad[first:last + 1].any():
            offenders = {c: report[c]["longest_gap"] for c in cols
                         if report[c]["longest_gap"] > max_gap}
            raise ValueError(
                f"NaN gap(s) exceed max_nan_gap={max_gap} sample(s) and cannot "
                f"be bridged: {offenders} (column: longest gap in samples). "
                f"This looks like a real sensor outage, not a dropout -- verify "
                f"the record before raising --max-nan-gap to cover it."
            )
        n_trimmed = len(df) - (last - first + 1)
        if n_trimmed:
            warnings.warn(
                f"Trimming {n_trimmed} row(s) at the start/end of the record -- "
                f"unrecoverable NaNs with no valid sample to interpolate from."
            )
        df = df.iloc[first:last + 1].reset_index(drop=True)
        report["rows_trimmed"] = n_trimmed
    return df, report


def load_csv(path: str, max_nan_gap: int = 15) -> tuple[pd.DataFrame, dict]:
    """Read the flight CSV with dtypes forced, not inferred.

    CSV carries no type information, so one malformed row can silently turn a
    numeric column into strings and every number downstream becomes wrong
    without an error. Declaring dtypes makes that a failure instead.

    max_nan_gap: NaN runs up to this many samples (in any sensor column) are
    linearly interpolated -- see _fill_nan_gaps. Longer runs raise.
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

    # --- NaN gaps -------------------------------------------------------------
    # t_ms is not interpolable (it's the time axis itself); a NaN there is a
    # corrupt row, not a dropout.
    if df["t_ms"].isna().any():
        raise ValueError("t_ms contains NaN -- cannot repair the time axis itself.")
    sensor_cols = [c for c in
                   ("gx_dps", "gy_dps", "gz_dps", "pressure_pa", "temp_c",
                    "rh_pct", "ax_ms2", "ay_ms2", "az_ms2") if c in df.columns]
    df, nan_report = _fill_nan_gaps(df, sensor_cols, max_nan_gap)
    report["nan_gaps"] = nan_report
    report["n_rows_after_nan_trim"] = int(len(df))

    # --- timestamp ----------------------------------------------------------
    t = df["t_ms"].to_numpy(dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        warnings.warn("Non-monotonic timestamps -- sorting and de-duplicating.")
        n_before = len(df)
        df = (df.sort_values("t_ms").drop_duplicates(subset="t_ms")
                .reset_index(drop=True))
        # A handful of duplicate rows is normal SD-card behaviour. Losing most
        # of the record to dedup means most timestamps were equal, which is a
        # frozen/broken logger clock, not a few duplicate samples -- surface
        # that now rather than let it resurface downstream as an opaque
        # "not enough samples" error with no clue as to why they're gone.
        dropped_frac = 1.0 - len(df) / n_before
        if dropped_frac > 0.5:
            raise ValueError(
                f"De-duplicating t_ms dropped {100*dropped_frac:.1f}% of rows "
                f"({n_before} -> {len(df)}). Timestamps are mostly or entirely "
                f"identical -- this looks like a frozen/broken logger clock, "
                f"not routine duplicate samples. Check the raw file."
            )
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
    ap.add_argument("--max-nan-gap", type=int, default=15,
                    help="longest NaN run (samples) to bridge by linear "
                         "interpolation; longer runs raise (default: 15)")
    args = ap.parse_args()

    df, rep = load_csv(args.csv, max_nan_gap=args.max_nan_gap)
    print(json.dumps(rep, indent=2))
    print(f"\ncolumns: {list(df.columns)}")
    print(df.head())
    if args.bias_file:
        bias, temp = load_bias(args.bias_file)
        print(f"\ngyro bias (rad/s): {np.round(bias, 6).tolist()}   at {temp} C")
