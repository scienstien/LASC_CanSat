"""
assessment.py -- STAGE 5: the three performance criteria.

    1. design intent  -- measured C_D*A against the design value
    2. consistency    -- did C_D*A hold steady through descent
    3. stability      -- planar swing or coning, and how it damped

Criteria 1 and 2 come from the aerodynamic stage, criterion 3 from the
oscillation stage. A recovery system can satisfy 1 and fail 3: correct drag
with severe coning. They are reported independently.
"""

from __future__ import annotations

from .config import Config

# Measurement uncertainty on C_D*A once the systematic terms are removed
# (hypsometric altitude, humidity correction, oscillation-bias correction).
# Random contributions: velocity ~1%, density ~0.6%, mass ~0.5% -> ~1.3%.
# A difference is called significant at roughly twice that.
CDA_UNCERTAINTY = 0.013


def design_intent(cfg: Config, aero: dict) -> dict:
    """Measured drag area against the design value.

    Descent rate varies as (C_D*A)^-1/2, so a given fractional shortfall in
    drag appears at only half that size in descent rate. Drag area is the more
    sensitive discriminator and is used as the primary metric.
    """
    if not cfg.design_cda_m2:
        return {"verdict": "no design C_D*A supplied", "available": False}

    measured = aero["cda_corrected_m2"]
    ratio = measured / cfg.design_cda_m2
    dev = 100 * (ratio - 1)
    threshold = 100 * 2 * CDA_UNCERTAINTY

    return {
        "available": True,
        "design_cda_m2": cfg.design_cda_m2,
        "measured_cda_m2": measured,
        "efficiency_ratio": float(ratio),
        "deviation_pct": float(dev),
        "significance_threshold_pct": float(threshold),
        "significant": bool(abs(dev) > threshold),
        "verdict": (f"within measurement uncertainty of design "
                    f"({dev:+.1f}%, threshold ±{threshold:.1f}%)"
                    if abs(dev) <= threshold else
                    f"differs from design beyond measurement uncertainty "
                    f"({dev:+.1f}%, threshold ±{threshold:.1f}%)"),
    }


def consistency(aero: dict) -> dict:
    """Did the drag area hold steady through the descent.

    Under a fully inflated canopy at roughly constant Reynolds number, C_D*A
    should be flat. Two independent tests:

      - trend      : regression of the CORRECTED series against time. The
                     oscillation-bias correction must already be applied, or
                     the decaying swing produces a spurious downward trend.
      - anomalies  : accelerometer departure from 1 g. Reported as BOTH a
                     failure fraction and the longest contiguous run, because
                     one 200-sample block is a physical event while 200
                     scattered samples are noise, and the fraction alone
                     cannot distinguish them.
    """
    flags = []
    p = aero["trend_p_value"]
    slope = aero["trend_slope_per_s"]
    cda = aero["cda_corrected_m2"]

    # Statistical significance alone is not enough: with thousands of samples
    # a physically negligible slope reaches p < 0.001. Require the drift to
    # also exceed the measurement uncertainty over the window.
    drift_pct = None
    if p is not None and slope is not None:
        span = aero["t"][aero["steady_mask"]]
        duration = float(span[-1] - span[0]) if len(span) > 1 else 0.0
        drift_pct = 100 * slope * duration / cda
        if p < 0.01 and abs(drift_pct) > 100 * CDA_UNCERTAINTY:
            flags.append(f"C_D*A drifts {drift_pct:+.1f}% across the window "
                         f"(p = {p:.1e}) -- larger than measurement uncertainty")

    q = aero["quasi_steady_check"]
    if q and q["longest_contiguous_run"] > 50:
        flags.append(f"{q['longest_contiguous_run']} consecutive samples failed "
                     f"the quasi-steady check -- possible canopy event")

    return {"trend_p_value": p, "drift_pct_over_window": drift_pct,
            "quasi_steady": q, "flags": flags,
            "verdict": "consistent through descent" if not flags
                       else "anomalies flagged"}


def stability(osc: dict) -> dict:
    """Motion type, damping mechanism, and how long the swing persists."""
    d, pca = osc["damping"], osc["pca"]
    out = {
        "motion": pca["classification"],
        "ev_ratio_21": pca["ev_ratio_21"],
        "is_planar": pca["is_planar"],
    }
    if d.get("ok"):
        out.update({
            "mechanism": d["mechanism"],
            "zeta_linear": d["zeta_linear"],
            "c_quadratic": d["c_quadratic"],
            "settling_time_2pct_s": d["settling_time_2pct_s"],
            "settling_law": d["settling_law"],
            "quantities_note": d["note"],
        })
        if "caveat" in d:
            out["caveat"] = d["caveat"]
    else:
        out["mechanism"] = f"not identified: {d.get('reason')}"

    out["verdict"] = ("stable planar swing" if pca["is_planar"]
                      else "coning present -- drag may be nominal while the "
                           "descent is not stable")
    return out


def assess(cfg: Config, osc: dict, aero: dict) -> dict:
    return {"design_intent": design_intent(cfg, aero),
            "consistency": consistency(aero),
            "stability": stability(osc)}
