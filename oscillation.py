"""
oscillation.py -- STAGE 3: PCA -> motion classification -> damping.

Runs BEFORE the aerodynamic stage, which consumes the oscillation period
returned here.

    python -m descent.oscillation flight.csv --bias-file bias.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks, get_window, savgol_filter
from scipy.stats import linregress

from .config import GYRO, Config


# =============================================================================
# 3.1  Dominant-mode extraction
# =============================================================================
def pca_mode(df: pd.DataFrame) -> dict:
    """Project 3-axis angular rate onto its dominant direction.

    The oscillation direction depends on how the vehicle left the rocket and
    bears no relation to how the IMU was mounted, so the motion is spread
    across all three channels and any single channel captures only a
    projection.

    The variance ratio is a physical result, not just a diagnostic:
        ev2/ev1 small      -> planar pendulum swing
        ev2/ev1 comparable -> two modes in quadrature, i.e. coning
    """
    X = df[GYRO].to_numpy(dtype=np.float64)
    Xc = X - X.mean(axis=0)
    # SVD rather than an explicit covariance eigendecomposition: better
    # conditioned when one axis is much noisier than the others.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev = (S ** 2) / (len(Xc) - 1)
    ratio = float(ev[1] / ev[0])

    if ratio < 0.25:
        cls, planar = "planar swing (single dominant mode)", True
    elif ratio > 0.60:
        cls, planar = "coning (two comparable modes)", False
    else:
        cls, planar = "mixed / transitional", False

    return {"pc1": Xc @ Vt[0], "pc2": Xc @ Vt[1],
            "explained": (ev / ev.sum()).tolist(),
            "ev_ratio_21": ratio,
            "classification": cls, "is_planar": planar,
            "axis_body_frame": Vt[0].tolist()}


# =============================================================================
# 3.2  Damping identification
# =============================================================================
def autocorr_period(t: np.ndarray, x: np.ndarray) -> float | None:
    """Coarse period from the first autocorrelation peak.

    Used only to set a minimum peak spacing. Kept in the time domain
    deliberately, so the FFT stays a genuinely independent cross-check.
    """
    x = detrend(x, type="linear")
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    if r[0] <= 0:
        return None
    r = r / r[0]
    neg = np.nonzero(r < 0)[0]
    if neg.size == 0:
        return None
    pk, _ = find_peaks(r[neg[0]:])
    if pk.size == 0:
        return None
    return float((neg[0] + pk[0]) * np.median(np.diff(t)))


def identify_damping(t: np.ndarray, x: np.ndarray, min_peaks: int = 5) -> dict:
    """Separate linear and quadratic damping from a single regression.

    Energy balance over one cycle gives the amplitude loss per cycle:

        dA_linear    = -2*pi*zeta * A        (proportional to A)
        dA_quadratic = -(8/3) * c  * A^2     (proportional to A^2)

    The two mechanisms therefore enter the logarithmic decrement at DIFFERENT
    POWERS OF AMPLITUDE, which is what makes them separable:

        delta = 2*pi*zeta/sqrt(1-zeta^2)  +  (8/3) * c * Abar

    a straight line in Abar. Intercept -> zeta, slope -> c.

    The quadratic term is the standard equivalent-viscous-damping result for
    velocity-squared damping (Stutts eqs. 21-22; Den Hartog; Rao), rewritten
    as a logarithmic decrement.

    UNITS: x is angular RATE, not angle. delta is a ratio so the factor omega
    cancels and zeta needs no correction, but the slope carries it:
    c = 3 * omega_n * slope / 8.
    """
    x = detrend(x, type="linear")
    dt = float(np.median(np.diff(t)))

    T_guess = autocorr_period(t, x)
    distance = max(3, int(0.6 * T_guess / dt)) if T_guess else 3
    idx, _ = find_peaks(x, prominence=0.05 * np.std(x), distance=distance)
    if len(idx) < min_peaks:
        return {"ok": False, "reason": f"only {len(idx)} peaks (need {min_peaks})"}

    # Truncate where the peak sequence stops carrying envelope information.
    #
    # This must keep a CONTIGUOUS leading run. Selecting a scattered subset
    # would leave delta_n = ln(A_n/A_n+1) comparing peaks that are no longer
    # adjacent cycles, which is meaningless.
    #
    # Two stopping conditions, whichever comes first:
    #   - amplitude falls into the noise floor
    #   - amplitude stops decreasing (noise, not physics: a decaying
    #     oscillation cannot have a peak larger than the one before it)
    win = min(len(x) - 1 | 1, max(5, distance | 1))
    resid = x - savgol_filter(x, win, 2)
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    floor = max(8.0 * noise, 0.03 * x[idx[0]])

    A_all = x[idx]
    stop = len(A_all)
    for i in range(1, len(A_all)):
        if A_all[i] < floor or A_all[i] >= A_all[i - 1]:
            stop = i
            break
    if stop >= min_peaks:
        idx = idx[:stop]

    tp, A = t[idx], x[idx]
    if len(A) < min_peaks:
        return {"ok": False,
                "reason": f"only {len(A)} usable peaks above the noise floor"}

    # Period from a straight-line fit of peak time vs peak index -- less
    # sensitive to one missed or extra peak than mean spacing.
    T_d = float(np.polyfit(np.arange(len(tp)), tp, 1)[0])
    omega_d = 2.0 * np.pi / T_d

    delta = np.log(A[:-1] / A[1:])
    # Geometric mean, not the starting amplitude: the second-order bias terms
    # cancel identically for this choice, leaving a third-order residual.
    Abar = np.sqrt(A[:-1] * A[1:])
    fit = linregress(Abar, delta)

    d = max(fit.intercept, 0.0)
    zeta = float(d / np.sqrt(4 * np.pi ** 2 + d ** 2))
    omega_n = float(omega_d / np.sqrt(1 - zeta ** 2)) if zeta < 1 else omega_d
    c = float(3.0 * omega_n * fit.slope / 8.0)
    quadratic = bool(fit.pvalue < 0.01 and fit.slope > 0)

    sigma = zeta * omega_n
    k = 4.0 * c * omega_n / (3.0 * np.pi)

    out = {
        "ok": True, "n_cycles": int(len(A)), "peak_idx": idx,
        "period_s": T_d, "f_damped_hz": float(1.0 / T_d),
        "f_natural_hz": float(omega_n / (2 * np.pi)), "omega_n_rads": omega_n,
        "intercept": float(fit.intercept), "slope": float(fit.slope),
        "r_value": float(fit.rvalue), "p_slope": float(fit.pvalue),
        "delta_max": float(delta.max()), "delta_min": float(delta.min()),
        "zeta_linear": zeta, "c_quadratic": c,
        "quadratic_significant": quadratic, "A0": float(A[0]),
    }

    # --- settling time ------------------------------------------------------
    # dA/dt = -sigma*A - k*A^2 is a Bernoulli equation; u = 1/A linearises it
    # exactly, giving
    #     A(t) = sigma*A0*exp(-sigma t) / [sigma + k*A0*(1 - exp(-sigma t))]
    # Neither pure limit is valid when both mechanisms are present, so solve
    # the combined envelope rather than using ln(50)/sigma (linear only) or
    # 49/(k*A0) (quadratic only).
    A0_angle = A[0] / omega_n          # peaks are angular RATE -> angle
    kA0 = k * A0_angle
    if sigma > 1e-9 and kA0 > 1e-9:
        u = 0.02 * (sigma + kA0) / (sigma + 0.02 * kA0)
        out["settling_time_2pct_s"] = float(-np.log(u) / sigma)
        out["settling_law"] = "combined envelope"
    elif sigma > 1e-9:
        out["settling_time_2pct_s"] = float(np.log(50) / sigma)
        out["settling_law"] = "exponential, ln(50)/sigma"
    elif kA0 > 1e-9:
        out["settling_time_2pct_s"] = float(49.0 / kA0)
        out["settling_law"] = "hyperbolic, 49/(k*A0)"
    else:
        out["settling_time_2pct_s"] = float("nan")
        out["settling_law"] = "undetermined"

    # Which decay quantities are meaningful depends on the regression outcome.
    if quadratic:
        out["mechanism"] = "quadratic dominant" if zeta < 1e-3 else "both mechanisms"
        # Amplitude-dependent, so it is quoted WITH the amplitude it refers to.
        out["zeta_equivalent_at_A0"] = float((4 / (3 * np.pi)) * c * A0_angle)
        out["zeta_equivalent_amplitude_rad"] = float(A0_angle)
        out["note"] = ("quadratic term significant: zeta is amplitude-dependent, "
                       "c is the system constant")
    else:
        out["mechanism"] = "linear viscous"
        out["decay_constant_sigma"] = float(sigma)
        out["log_decrement"] = float(fit.intercept)
        out["note"] = ("quadratic term not significant: classical zeta, delta "
                       "and sigma are valid as conventionally defined")
    return out


# =============================================================================
# 3.3  Independent frequency estimate
# =============================================================================
def fft_check(t: np.ndarray, x: np.ndarray, fs: float) -> dict:
    """Deliberately not derived from the peak finder, so agreement between the
    two estimates means something."""
    x = detrend(x, type="linear")
    X = np.abs(np.fft.rfft(x * get_window("hann", len(x))))
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    valid = f > 1.5 / (t[-1] - t[0])          # need at least one full cycle
    if not valid.any():
        return {"ok": False, "reason": "record too short"}
    return {"ok": True, "freqs": f, "mag": X,
            "f_peak_hz": float(f[valid][int(np.argmax(X[valid]))]),
            "resolution_hz": float(f[1] - f[0])}


# =============================================================================
# 3.4  Rotational energy
# =============================================================================
def rotational_energy(t, omega_xyz, cfg: Config) -> dict:
    """Without a measured inertia this reports specific energy (KE/I), which
    keeps the time history and decay meaningful without inventing a value."""
    ke = 0.5 * np.sum(omega_xyz ** 2, axis=1)
    has_I = cfg.inertia_kgm2 is not None
    if has_I:
        ke = ke * cfg.inertia_kgm2
    dE = np.gradient(ke, t)
    return {"has_inertia": has_I,
            "units": "J" if has_I else "J/(kg m^2) [specific]",
            "ke": ke, "dE_dt": dE,
            "ke_initial": float(ke[0]), "ke_final": float(ke[-1]),
            "mean_dissipation_rate": float(np.mean(dE))}


# =============================================================================
def analyse(df: pd.DataFrame, cfg: Config, fs: float) -> dict:
    t = df["t"].to_numpy()
    pca = pca_mode(df)
    damping = identify_damping(t, pca["pc1"])

    # The damping treatment is derived for a one-dimensional oscillator.
    if damping["ok"] and not pca["is_planar"]:
        damping["caveat"] = (
            "PCA indicates the vehicle is not in planar motion; the "
            "one-dimensional damping treatment does not apply unmodified and "
            "these coefficients are indicative only.")

    fft = fft_check(t, pca["pc1"], fs)
    energy = rotational_energy(t, df[GYRO].to_numpy(), cfg)

    cross = {}
    if damping["ok"] and fft["ok"]:
        ft, ff = damping["f_damped_hz"], fft["f_peak_hz"]
        cross = {"f_time_domain_hz": ft, "f_fft_hz": ff,
                 "disagreement_pct": float(100 * abs(ft - ff) / ff),
                 "agree": bool(abs(ft - ff) <= 2 * fft["resolution_hz"])}

    return {"pca": pca, "damping": damping, "fft": fft,
            "energy": energy, "cross_check": cross}


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json
    from .load import load_bias, load_csv
    from . import preprocess

    ap = argparse.ArgumentParser(description="Stage 3: oscillation and damping")
    ap.add_argument("csv")
    ap.add_argument("--bias-file", default=None)
    ap.add_argument("--riser", type=float, default=0.50)
    args = ap.parse_args()

    cfg = Config(riser_length_m=args.riser)
    df, _ = load_csv(args.csv)
    bias = None
    if args.bias_file:
        bias, cfg.bias_temp_c = load_bias(args.bias_file)
    df, _, pre = preprocess.run(df, cfg, bias)
    out = analyse(df, cfg, pre["fs_hz"])

    slim = {"pca": {k: v for k, v in out["pca"].items()
                    if k not in ("pc1", "pc2")},
            "damping": {k: v for k, v in out["damping"].items()
                        if k != "peak_idx"},
            "cross_check": out["cross_check"]}
    print(json.dumps(slim, indent=2, default=float))
