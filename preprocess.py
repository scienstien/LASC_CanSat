"""
preprocess.py -- STAGE 2: time base -> gyro bias -> zero-phase filter.

The order matters: bias is a DC offset and comes off before filtering so it
cannot interact with filter edge transients.

Run this stage alone:

    python -m descent.preprocess flight.csv --bias-file bias.csv
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .config import ACC, GYRO, G, Config


def find_deployment(df: pd.DataFrame,
                     deploy_ms: int | None = None) -> tuple[float, str]:
    """Deployment instant, from the acceleration signature -- or a manual override.

    The logged timestamp is uptime since boot, so analysis times must be
    referenced to deployment rather than to power-up.

    deploy_ms, if given, is a raw t_ms value (e.g. from --deploy-ms) that
    overrides the accelerometer inference entirely. It must fall within the
    file's t_ms range -- outside it, the likeliest explanation is a value
    pasted from a different flight or a units mix-up, so that raises rather
    than silently clamping. It is snapped to the nearest logged sample.
    """
    if deploy_ms is not None:
        t_ms = df["t_ms"].to_numpy(dtype=np.float64)
        lo, hi = float(t_ms.min()), float(t_ms.max())
        if not (lo <= deploy_ms <= hi):
            raise ValueError(
                f"--deploy-ms={deploy_ms} is outside this file's t_ms range "
                f"[{lo:.0f}, {hi:.0f}]. Check for a value pasted from a "
                f"different flight, or a units mix-up (t_ms is milliseconds)."
            )
        idx = int(np.argmin(np.abs(t_ms - deploy_ms)))
        snap_ms = abs(t_ms[idx] - deploy_ms)
        t0 = float(df["t_s"].iloc[idx])

        total = float(df["t_s"].iloc[-1]) - float(df["t_s"].iloc[0])
        remaining = float(df["t_s"].iloc[-1]) - t0
        if total > 0 and remaining < 0.2 * total:
            warnings.warn(
                f"--deploy-ms={deploy_ms} leaves only "
                f"{100 * remaining / total:.1f}% of the record after it."
            )

        method = f"manual --deploy-ms={deploy_ms}"
        if snap_ms > 0:
            method += f" (snapped {snap_ms:.0f} ms)"
        return t0, method

    # --- accelerometer inference ---------------------------------------------
    # First crossing of 3g, deliberately not the largest |a| in the record:
    # the largest is often the landing impact, and picking that would discard
    # the whole descent. A detection that leaves under 20% of the record is
    # rejected as a likely misdetection (e.g. a bump near the very end) rather
    # than trusted -- unlike a manual --deploy-ms, which is the user's call.
    if not all(c in df for c in ACC):
        return float(df["t_s"].iloc[0]), "no accelerometer columns -- using record start"
    a = np.linalg.norm(df[ACC].to_numpy(dtype=np.float64), axis=1)
    crossings = np.flatnonzero(a >= 3.0 * G)
    if crossings.size == 0:
        return (float(df["t_s"].iloc[0]),
                "no shock above 3g in record -- using record start")

    idx = int(crossings[0])
    t0 = float(df["t_s"].iloc[idx])
    total = float(df["t_s"].iloc[-1]) - float(df["t_s"].iloc[0])
    remaining = float(df["t_s"].iloc[-1]) - t0
    if total > 0 and remaining < 0.2 * total:
        warnings.warn(
            f"3g crossing detected at t={t0:.2f}s, but that leaves only "
            f"{100 * remaining / total:.1f}% of the record after it -- "
            f"rejecting as a likely misdetection and using record start "
            f"instead. Use --deploy-ms to override explicitly."
        )
        return (float(df["t_s"].iloc[0]),
                "3g crossing rejected (<20% of record remained) -- using record start")
    return t0, "accelerometer inference (first 3g crossing)"


def run(df: pd.DataFrame, cfg: Config, bias: np.ndarray | None = None,
        deploy_ms: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (filtered, raw_debiased, info).

    raw_debiased is kept so the filter can be inspected against its input --
    the first thing to look at when the oscillation stage misbehaves.
    """
    info: dict = {}

    # --- 2.1 time base ------------------------------------------------------
    t0, deploy_method = find_deployment(df, deploy_ms)
    df = df[df["t_s"] >= t0].reset_index(drop=True)
    df["t"] = df["t_s"] - t0
    info["deployment_at_s"] = t0
    info["deployment_method"] = deploy_method

    t = df["t"].to_numpy()
    dt = np.diff(t)
    fs = 1.0 / np.median(dt)
    jitter = float(np.std(dt) / np.median(dt))
    info["fs_hz"], info["jitter_pct"] = float(fs), 100 * jitter

    # filtfilt has no notion of timestamps -- it assumes uniform spacing, so
    # jitter would make the effective cutoff vary through the record.
    if jitter > 0.01:
        warnings.warn(f"Sample jitter {100*jitter:.1f}% -- resampling to {fs:.1f} Hz.")
        t_new = np.arange(t[0], t[-1], 1.0 / fs)
        out = {"t": t_new}
        for c in df.columns:
            if c in ("t", "t_s", "t_ms", "gps_fix"):
                continue
            out[c] = np.interp(t_new, t, df[c].to_numpy(dtype=np.float64))
        if "gps_fix" in df:
            out["gps_fix"] = np.interp(t_new, t, df["gps_fix"].to_numpy(float)) > 0.5
        df = pd.DataFrame(out)
        info["resampled"] = True
    else:
        info["resampled"] = False

    # --- 2.2 gyro bias ------------------------------------------------------
    if bias is None:
        bias = np.asarray(cfg.bias_rads, dtype=np.float64)
        info["bias_source"] = "config constant"
    else:
        info["bias_source"] = "pre-flight stillness calibration"
    for i, c in enumerate(GYRO):
        df[c] = df[c].to_numpy(dtype=np.float64) - bias[i]
    info["bias_rads"] = np.asarray(bias).tolist()
    info["bias_temp_c"] = cfg.bias_temp_c

    # Bias affects the two strands unevenly: the damping results are largely
    # insensitive to it (PCA centres the data, the mode is detrended, and the
    # decrement is a ratio), but rotational energy depends on the ABSOLUTE
    # rate through 1/2 omega^2 and is affected directly.
    raw = df.copy()

    # --- 2.3 zero-phase filter ---------------------------------------------
    # Forward-backward filtering cancels phase shift exactly. That matters
    # because the damping analysis measures the TIMING of oscillation peaks; a
    # phase-shifted signal would bias the recovered period. It needs samples
    # from after each point, which is why this is a ground-station operation
    # and cannot run onboard.
    fc = cfg.cutoff_hz(fs)
    b, a = butter(cfg.filter_order, fc / (0.5 * fs), btype="low")
    padlen = 3 * max(len(a), len(b))
    if len(df) <= padlen:
        raise ValueError(f"Only {len(df)} samples; filtfilt needs more than {padlen}.")

    for c in GYRO + [c for c in ACC if c in df]:
        df[c] = filtfilt(b, a, df[c].to_numpy(dtype=np.float64))

    info.update({
        "filter_order": cfg.filter_order,
        "cutoff_hz": float(fc),
        "pendulum_estimate_hz": cfg.pendulum_frequency_hz(),
        "margin_over_estimate": float(fc / cfg.pendulum_frequency_hz()),
        "nyquist_hz": float(0.5 * fs),
    })
    return df, raw, info


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    from .load import load_bias, load_csv

    ap = argparse.ArgumentParser(description="Stage 2: preprocess a flight CSV")
    ap.add_argument("csv")
    ap.add_argument("--bias-file", default=None)
    ap.add_argument("--riser", type=float, default=0.50)
    ap.add_argument("--deploy-ms", type=int, default=None,
                    help="override deployment instant (raw t_ms), instead of "
                         "inferring it from the accelerometer")
    ap.add_argument("--out", default=None, help="write the preprocessed CSV here")
    args = ap.parse_args()

    cfg = Config(riser_length_m=args.riser)
    df, _ = load_csv(args.csv, max_nan_gap=cfg.max_nan_gap_samples)
    bias = None
    if args.bias_file:
        bias, cfg.bias_temp_c = load_bias(args.bias_file)
    out, raw, info = run(df, cfg, bias, deploy_ms=args.deploy_ms)
    print(json.dumps(info, indent=2))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(out)} rows)")
