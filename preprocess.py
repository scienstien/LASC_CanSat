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


def find_deployment(df: pd.DataFrame) -> float:
    """Deployment instant, from the acceleration signature.

    The logged timestamp is uptime since boot, so analysis times must be
    referenced to deployment rather than to power-up.
    """
    if not all(c in df for c in ACC):
        return float(df["t_s"].iloc[0])
    a = np.linalg.norm(df[ACC].to_numpy(dtype=np.float64), axis=1)
    if a.max() < 3.0 * G:
        return float(df["t_s"].iloc[0])          # no clear shock in the record
    return float(df["t_s"].iloc[int(np.argmax(a))])


def run(df: pd.DataFrame, cfg: Config,
        bias: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (filtered, raw_debiased, info).

    raw_debiased is kept so the filter can be inspected against its input --
    the first thing to look at when the oscillation stage misbehaves.
    """
    info: dict = {}

    # --- 2.1 time base ------------------------------------------------------
    t0 = find_deployment(df)
    df = df[df["t_s"] >= t0].reset_index(drop=True)
    df["t"] = df["t_s"] - t0
    info["deployment_at_s"] = t0

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
    ap.add_argument("--out", default=None, help="write the preprocessed CSV here")
    args = ap.parse_args()

    cfg = Config(riser_length_m=args.riser)
    df, _ = load_csv(args.csv)
    bias = None
    if args.bias_file:
        bias, cfg.bias_temp_c = load_bias(args.bias_file)
    out, raw, info = run(df, cfg, bias)
    print(json.dumps(info, indent=2))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(out)} rows)")
