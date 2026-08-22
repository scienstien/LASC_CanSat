"""
aerodynamics.py -- STAGE 4: density -> altitude -> velocity -> C_D*A -> beta.

CONSUMES the oscillation period from stage 3: the analysis window is snapped
to a whole number of oscillation periods, and C_D*A is corrected for the
oscillation bias. Passing period=None makes the stage runnable standalone for
debugging, at the cost of those two refinements.

    python -m descent.aerodynamics flight.csv --bias-file bias.csv --mass 0.35
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress

from .config import ACC, G, MU_AIR, R_DRY, R_VAPOUR, Config
from .load import longest_run


# =============================================================================
# 4.1  Air density
# =============================================================================
def saturation_vapour_pressure(temp_c: np.ndarray) -> np.ndarray:
    """Buck (1981), Pa."""
    return 611.21 * np.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))


def air_density(p: np.ndarray, temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Humid air density by Dalton partial pressures.

        rho = p_dry/(R_d T) + p_vapour/(R_v T)

    Water vapour is LIGHTER than dry air (R_v > R_d), so humid air is less
    dense at the same pressure and temperature -- the opposite of what most
    people expect. Neglecting humidity therefore overestimates rho and
    underestimates C_D*A, by roughly 0.6% at typical conditions.
    """
    T = temp_c + 273.15
    p_v = np.clip(rh, 0, 100) / 100.0 * saturation_vapour_pressure(temp_c)
    return (p - p_v) / (R_DRY * T) + p_v / (R_VAPOUR * T)


def virtual_temperature(p, temp_c, rh):
    """Temperature a dry parcel would need to match the moist density. Lets
    the hypsometric relation stay in its dry-air form."""
    T = temp_c + 273.15
    p_v = np.clip(rh, 0, 100) / 100.0 * saturation_vapour_pressure(temp_c)
    return T / (1.0 - (p_v / p) * (1.0 - R_DRY / R_VAPOUR))


# =============================================================================
# 4.2  Altitude
# =============================================================================
def altitude_hypsometric(p, temp_c, rh) -> np.ndarray:
    """Height above the landing level, using the MEASURED temperature.

    From hydrostatic balance with rho = P/(R T):

        d(ln P) = -g dz / (R T)     =>     dz = -(R T / g) d(ln P)

    so the height of a level above the reference is

        h = (R/g) * INTEGRAL from ln(P) to ln(P_ref) of  Tv  d(ln P)

    Two things this deliberately does NOT do:

    1. It does not use h = 44330[1-(P/P0)^0.190263]. Those constants embed a
       288.15 K sea-level assumption (44330 = T0/lapse), i.e. they convert
       pressure to height using a temperature nobody measured, while a
       thermometer is sitting on the board. Since h scales with T, the error
       is the ratio of assumed to actual temperature -- roughly 4% in altitude
       and 8% in C_D*A at ~24 C.

    2. It does not collapse the profile to a single layer with a
       height-averaged temperature. The relation needs the temperature
       averaged with respect to ln P, not height. The two differ by ~0.7% when
       T varies through the descent, and velocity error enters C_D*A squared.
    """
    p = np.asarray(p, dtype=np.float64)
    Tv = virtual_temperature(p, temp_c, rh)
    lnp = np.log(p)

    order = np.argsort(lnp)                 # make the integration variable monotonic
    lp, Tv_s = lnp[order], Tv[order]
    seg = np.diff(lp) * 0.5 * (Tv_s[1:] + Tv_s[:-1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    h_sorted = -(R_DRY / G) * cum
    h = np.empty_like(h_sorted)
    h[order] = h_sorted
    return h - h[int(np.argmax(p))]         # zero at the landing level


# =============================================================================
# 4.3  Velocity
# =============================================================================
def mad_mask(x: np.ndarray, threshold: float = 5.0, window: int = 21) -> np.ndarray:
    """Rolling median-absolute-deviation filter. True = keep.

    Targets GPS jumps and deployment-shock pressure spikes: short, large
    excursions that would otherwise drag the velocity fit.
    """
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    resid = (s - med).abs()
    mad = resid.rolling(window, center=True, min_periods=1).median()
    return (resid / (1.4826 * mad.replace(0, np.nan))).fillna(0.0).to_numpy() < threshold


def descent_velocity(h: np.ndarray, fs: float, window_s: float) -> dict:
    """v = dh/dt.

    Savitzky-Golay differentiates and smooths in one pass; raw finite
    differencing of a noisy altitude trace amplifies noise badly (in testing,
    velocity standard deviation went from 0.26 to 18.8 m/s without it).

    The accelerometer is NOT integrated for absolute velocity: a constant bias
    integrates into a linearly growing error which, over a full descent,
    dominates the drag-area measurement entirely. Its role is the short-window
    quasi-steady check below, where drift has not yet accumulated.
    """
    w = int(window_s * fs)
    w = max(5, w + 1 - w % 2)
    w = min(w, (len(h) - 1) | 1)
    return {"altitude_smooth": savgol_filter(h, w, 3),
            "descent_speed": -savgol_filter(h, w, 3, deriv=1, delta=1.0 / fs),
            "window_samples": int(w), "window_s": float(w / fs)}


def snap_window(t, start, end, period, min_periods) -> np.ndarray:
    """Trim the analysis window to a whole number of oscillation periods.

    A window ending mid-swing weights one phase of the oscillation more than
    the other. Needs the period from stage 3 -- which is why that stage runs
    first.
    """
    if period is None or period <= 0:
        return (t >= start) & (t <= end)
    n = int((end - start) // period)
    if n < min_periods:
        warnings.warn(f"Steady window spans only {n} oscillation periods "
                      f"(want >= {min_periods}); using the full window.")
        return (t >= start) & (t <= end)
    return (t >= start) & (t <= start + n * period)


# =============================================================================
def analyse(df: pd.DataFrame, cfg: Config, fs: float,
            period: float | None = None) -> dict:
    t = df["t"].to_numpy()
    p = df["pressure_pa"].to_numpy(dtype=np.float64)
    temp = df["temp_c"].to_numpy(dtype=np.float64)
    rh = df["rh_pct"].to_numpy(dtype=np.float64)

    # Reject pressure spikes before anything downstream consumes them.
    ok = mad_mask(p, cfg.mad_threshold)
    n_out = int((~ok).sum())
    if n_out:
        p = (pd.Series(np.where(ok, p, np.nan))
             .interpolate(limit_direction="both").to_numpy())

    rho = air_density(p, temp, rh)
    h = altitude_hypsometric(p, temp, rh)
    vel = descent_velocity(h, fs, cfg.velocity_window_s)
    v = vel["descent_speed"]

    # --- quasi-steady window -----------------------------------------------
    tail = t >= t[0] + 0.5 * (t[-1] - t[0])
    v_t = float(np.median(v[tail]))
    t_skip = cfg.terminal_skip_s(v_t)
    steady = snap_window(t, t[0] + t_skip, t[-1] - cfg.skip_end_s,
                         period, cfg.min_periods_in_window)
    if steady.sum() < 20:
        raise ValueError("Quasi-steady window too short after trimming.")

    # --- oscillation modulation --------------------------------------------
    # While the vehicle swings, vertical velocity oscillates, so drag never
    # exactly balances weight. Because C_D*A goes as 1/v^2 and that function
    # is convex, the slow half of each swing raises the answer more than the
    # fast half lowers it: a systematic bias of (3/2)*eta^2, which averaging
    # over whole cycles does NOT remove.
    #
    # eta is computed in a ROLLING window, not once for the record: the
    # modulation decays as the swing decays, so a single global eta would
    # leave a declining bias that mimics genuine canopy degradation in the
    # trend test below.
    w_eta = max(int((period or 2.0) * fs), vel["window_samples"]) | 1
    v_trend = savgol_filter(v, min(len(v) - 1 | 1, w_eta), 2)
    sigma_local = pd.Series(v - v_trend).rolling(
        w_eta, center=True, min_periods=w_eta // 4).std()
    eta_t = np.clip((np.sqrt(2) * sigma_local / pd.Series(v_trend))
                    .fillna(0.0).to_numpy(), 0.0, 0.5)
    bias_t = 1.0 + 1.5 * eta_t ** 2

    # --- drag area ----------------------------------------------------------
    #   m g = 0.5 rho v^2 (C_D A)   ->   C_D A = 2 m g / (rho v^2)
    # C_D and reference area are not separately identifiable for a flexible,
    # porous, partly inflated canopy -- candidate reference areas differ by
    # large factors -- so the product is the reported quantity.
    valid = (v > 1.0) & np.isfinite(v) & np.isfinite(rho) & (rho > 0)
    cda = np.full_like(v, np.nan)
    cda[valid] = 2.0 * cfg.mass_kg * G / (rho[valid] * v[valid] ** 2)
    cda_corrected = cda / bias_t             # pointwise, using the local eta

    sel = steady & valid
    t_w, cda_w = t[sel], cda_corrected[sel]
    keep = mad_mask(cda_w, 4.0)
    t_w, cda_w = t_w[keep], cda_w[keep]

    cda_raw = float(np.median(cda[sel][keep]))
    cda_corr = float(np.median(cda_w))
    q1, q3 = np.percentile(cda_w, [25, 75])

    # Trend tested on the CORRECTED series, so a decaying oscillation bias is
    # not mistaken for real canopy degradation.
    trend = linregress(t_w, cda_w) if len(cda_w) > 10 else None

    # --- anomaly screen -----------------------------------------------------
    accel = {}
    if all(c in df for c in ACC):
        a_mag = np.linalg.norm(df[ACC].to_numpy(dtype=np.float64), axis=1)
        # A swinging payload feels centripetal and tangential terms even in
        # healthy descent, so the threshold is loose: this screens for gross
        # anomalies (canopy collapse, gust), not fine sensor error.
        fw = (np.abs(a_mag - G) > cfg.accel_gate_frac * G)[steady]
        accel = {"fail_fraction": float(fw.mean()),
                 "longest_contiguous_run": int(longest_run(fw)),
                 "note": "long contiguous run => physical event; "
                         "scattered singles => noise"}

    Re = rho * v * cfg.ref_diameter_m / MU_AIR

    return {
        "t": t, "rho": rho, "altitude_m": h, "velocity": vel,
        "steady_mask": steady, "cda_series": cda_corrected,
        "n_pressure_outliers": n_out,
        "rho_mean": float(np.mean(rho[steady])),
        "terminal_velocity_ms": v_t,
        "t_skip_s": float(t_skip),
        "window_periods": (float((t[steady][-1] - t[steady][0]) / period)
                           if period else None),
        "eta_modulation": float(np.mean(eta_t[steady])),
        "oscillation_bias_pct": float(100 * (np.mean(bias_t[steady]) - 1)),
        "cda_raw_m2": cda_raw,
        "cda_corrected_m2": cda_corr,
        "cda_iqr_m2": float(q3 - q1),
        "n_samples_used": int(len(cda_w)),
        "ballistic_coefficient_kgm2": float(cfg.mass_kg / cda_corr),
        "reynolds_range": [float(np.min(Re[steady])), float(np.max(Re[steady]))],
        "trend_slope_per_s": float(trend.slope) if trend else None,
        "trend_p_value": float(trend.pvalue) if trend else None,
        "quasi_steady_check": accel,
    }


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    from .load import load_bias, load_csv
    from . import oscillation, preprocess

    ap = argparse.ArgumentParser(description="Stage 4: aerodynamics")
    ap.add_argument("csv")
    ap.add_argument("--bias-file", default=None)
    ap.add_argument("--mass", type=float, default=0.35)
    ap.add_argument("--riser", type=float, default=0.50)
    ap.add_argument("--no-oscillation", action="store_true",
                    help="skip stage 3; no period, so no window snapping")
    args = ap.parse_args()

    cfg = Config(mass_kg=args.mass, riser_length_m=args.riser)
    df, _ = load_csv(args.csv)
    bias = None
    if args.bias_file:
        bias, cfg.bias_temp_c = load_bias(args.bias_file)
    df, _, pre = preprocess.run(df, cfg, bias)

    period = None
    if not args.no_oscillation:
        osc = oscillation.analyse(df, cfg, pre["fs_hz"])
        period = osc["damping"].get("period_s") if osc["damping"]["ok"] else None

    out = analyse(df, cfg, pre["fs_hz"], period)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (np.ndarray, dict))},
                     indent=2, default=float))
