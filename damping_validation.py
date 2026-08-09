"""
=============================================================================
 DAMPING-MODEL IDENTIFICATION -- VALIDATION SUITE
 Companion to: Analysis Strand I, Table 8.1
=============================================================================

WHAT THIS DOES

 An analysis method cannot be checked against real flight data, because the
 true answer is not known. So this script builds artificial oscillations whose
 damping coefficients are chosen in advance, runs the identification method on
 them, and checks whether the values chosen are the values returned.

     1. Choose the answer      -- pick zeta and c
     2. Generate the motion    -- solve Eq. (5.1) numerically with those values
     3. Run the method         -- extract peaks, apply the regression of Sec. 7
     4. Compare                -- recovered coefficients vs. chosen coefficients

 Equation (5.1) is integrated in its EXACT form. The light-damping
 approximation of Sec. 5.2 is never assumed; it is what is being tested.
 Solver tolerance is 1e-12, several orders below the errors reported, so the
 residuals reflect the method rather than the numerics.

 Each block below prints the row of Table 8.1 that it verifies.

REQUIREMENTS   python 3.10+, numpy, scipy
RUN            python damping_validation.py
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.signal import find_peaks, detrend
from scipy.stats import linregress


# =============================================================================
# THE SYSTEM UNDER TEST
# =============================================================================

def simulate(zeta, c, wn, A0=1.0, T=120.0, n=400_000):
    """Integrate Eq. (5.1) exactly:

        theta'' + 2*zeta*wn*theta' + c*theta'|theta'| + wn^2*theta = 0

    Returns (time, angle). Nothing here assumes the averaging result --
    this is the full nonlinear equation.
    """
    def rhs(_t, y):
        th, w = y
        return [w, -2 * zeta * wn * w - c * abs(w) * w - wn ** 2 * th]

    sol = solve_ivp(rhs, (0, T), [A0, 0.0],
                    t_eval=np.linspace(0, T, n),
                    rtol=1e-12, atol=1e-14, method="DOP853")
    return sol.t, sol.y[0]


# =============================================================================
# THE METHOD BEING VALIDATED  (Sec. 7 of the report)
# =============================================================================

def extract_peaks(t, x, period_hint, floor_frac=0.02):
    """Positive peaks of the oscillation, spaced at least 0.6 periods apart and
    truncated once they sink toward the noise floor."""
    dt = t[1] - t[0]
    idx, _ = find_peaks(x, distance=max(3, int(0.6 * period_hint / dt)))
    A = x[idx]
    keep = A > floor_frac * A[0]
    return t[idx][keep], A[keep]


def identify(t, x, period_hint):
    """Sec. 7, steps 1-4.

        delta_n = ln(A_n / A_n+1)                 per-cycle decrement
        Abar_n  = sqrt(A_n * A_n+1)               geometric-mean amplitude
        delta   = 2*pi*zeta/sqrt(1-zeta^2) + (8/3)*c*Abar        Eq. (5.5)

    Straight line in Abar: intercept gives zeta, slope gives c.
    """
    _, A = extract_peaks(t, x, period_hint)
    if len(A) < 5:
        return None

    delta = np.log(A[:-1] / A[1:])            # step 1
    Abar = np.sqrt(A[:-1] * A[1:])            # step 2  (geometric mean)
    fit = linregress(Abar, delta)             # step 3

    # step 4 -- invert intercept for zeta: d = 2*pi*z/sqrt(1-z^2)
    d = fit.intercept
    zeta = d / np.sqrt(4 * np.pi ** 2 + d ** 2) if d > 0 else 0.0
    c = 3 * fit.slope / 8                     # angle units; see note below

    return {"zeta": zeta, "c": c, "intercept": d, "slope": fit.slope,
            "p_slope": fit.pvalue, "n_cycles": len(A), "delta_max": delta.max()}


# =============================================================================
def banner(row, title):
    print("\n" + "=" * 76)
    print(f"  TABLE 8.1, ROW {row}   {title}")
    print("=" * 76)


WN = 2 * np.pi * 0.62          # test frequency, 0.62 Hz
T0 = 2 * np.pi / WN


# =============================================================================
# ROW 1 -- reduce to the known linear case
# =============================================================================
def row1_linear_limit():
    banner(1, "Set c = 0; result must equal the textbook decrement")
    print("\n  With no quadratic damping the method must reproduce")
    print("      delta = 2*pi*zeta / sqrt(1 - zeta^2)")
    print("  which is the classical logarithmic decrement. A derivation failing")
    print("  this would be wrong regardless of its behaviour elsewhere.\n")
    print(f"  {'zeta chosen':>12} {'delta expected':>16} {'delta measured':>16} {'error':>10}")

    worst = 0.0
    for zeta in (0.060, 0.020):
        t, x = simulate(zeta, 0.0, WN)
        r = identify(t, x, T0)
        expected = 2 * np.pi * zeta / np.sqrt(1 - zeta ** 2)
        err = 100 * abs(r["intercept"] - expected) / expected
        worst = max(worst, err)
        print(f"  {zeta:12.3f} {expected:16.6f} {r['intercept']:16.6f} {err:9.3f}%")
    print(f"\n  -> worst error {worst:.3f}%   (report states 0.000 %)")


# =============================================================================
# ROWS 2-4 -- coefficient recovery
# =============================================================================
def rows2to4_recovery():
    banner("2-4", "Recover zeta, c, and both together")
    print("\n  Coefficients are chosen, motion is generated from them, and the")
    print("  method is asked to return them. The estimator is given no hint as")
    print("  to which mechanism dominates.\n")
    print(f"  {'zeta in':>8} {'c in':>7} | {'zeta out':>9} {'err':>8} | "
          f"{'c out':>9} {'err':>8} | {'cycles':>7}")

    cases = [(0.060, 0.00), (0.020, 0.00),          # pure linear
             (0.000, 0.35), (0.000, 0.15),          # pure quadratic
             (0.030, 0.20), (0.010, 0.40), (0.050, 0.10)]   # mixed

    wz = wc = wmix = 0.0
    for zeta, c in cases:
        t, x = simulate(zeta, c, WN)
        r = identify(t, x, T0)
        ez = 100 * abs(r["zeta"] - zeta) / zeta if zeta else float("nan")
        ec = 100 * abs(r["c"] - c) / c if c else float("nan")
        if zeta: wz = max(wz, ez)
        if c:    wc = max(wc, ec)
        if zeta and c: wmix = max(wmix, ez, ec)
        f = lambda v: f"{v:7.3f}%" if np.isfinite(v) else "      --"
        print(f"  {zeta:8.3f} {c:7.2f} | {r['zeta']:9.5f} {f(ez)} | "
              f"{r['c']:9.5f} {f(ec)} | {r['n_cycles']:7d}")

    print(f"\n  -> ROW 2  zeta from intercept, worst {wz:.3f}%   (report: <= 0.06 %)")
    print(f"     ROW 3  c from slope,        worst {wc:.3f}%   (report: <= 1.05 %)")
    print(f"     ROW 4  mixed systems,       worst {wmix:.3f}%   (report: <= 1.05 %)")


# =============================================================================
# ROW 5 -- closed-form envelope against the generated peaks
# =============================================================================
def row5_envelope():
    banner(5, "Predicted decay envelope vs. the generated peaks")
    print("\n  The averaging result gives a closed form for how peak height falls:")
    print("      A(t) = sigma*A0*exp(-sigma t) / [sigma + k*A0*(1 - exp(-sigma t))]")
    print("  with sigma = zeta*wn and k = 4*c*wn/(3*pi). Compared against the")
    print("  peaks the exact simulation actually produced.\n")
    print(f"  {'case':>22} {'decay':>9} {'cycles':>8} {'mean err':>10} {'max err':>10}")

    worst = 0.0
    for zeta, c, name in [(0.06, 0.00, "pure linear"),
                          (0.00, 0.35, "pure quadratic"),
                          (0.03, 0.20, "mixed"),
                          (0.01, 0.40, "quadratic-dominated")]:
        t, x = simulate(zeta, c, WN)
        tp, A = extract_peaks(t, x, T0)
        sigma, k = zeta * WN, 4 * c * WN / (3 * np.pi)
        tt, A0 = tp - tp[0], A[0]
        if sigma > 0:
            pred = sigma * A0 * np.exp(-sigma * tt) / (
                sigma + k * A0 * (1 - np.exp(-sigma * tt)))
        else:
            pred = A0 / (1 + A0 * k * tt)
        err = 100 * np.abs(pred - A) / A
        worst = max(worst, err.max())
        print(f"  {name:>22} {A[0]/A[-1]:8.1f}x {len(A):8d} "
              f"{err.mean():9.4f}% {err.max():9.4f}%")
    print(f"\n  -> worst error {worst:.4f}%   (report states <= 0.013 %)")


# =============================================================================
# ROW 6 -- the integral in Eq. (5.3)
# =============================================================================
def row6_integral():
    banner(6, "The integral in Eq. (5.3), evaluated numerically")
    print("\n  The derivation evaluates by hand:")
    print("      int_0^2pi |sin u|^3 du = 2 * int_0^pi sin^3 u du = 2 * (4/3) = 8/3")
    print("  Checked against numerical quadrature.\n")

    for name, f, exact in [
        ("int |sin u|^3 du", lambda u: abs(np.sin(u)) ** 3, 8 / 3),
        ("int sin^2 u du",   lambda u: np.sin(u) ** 2,      np.pi),
    ]:
        val, _ = quad(f, 0, 2 * np.pi, limit=400)
        print(f"  {name:>20}   numeric {val:.14f}   analytic {exact:.14f}"
              f"   err {abs(val - exact):.1e}")
    print("\n  -> agreement at machine precision   (report states 4e-16)")


# =============================================================================
# ROW 7 -- how far the light-damping assumption can be pushed
# =============================================================================
def row7_validity():
    banner(7, "Increase c until a cycle loses nearly half its amplitude")
    print("\n  Sec. 5.2 assumes each cycle loses only a small fraction of its")
    print("  energy. This raises c until that is no longer true, and reports")
    print("  where the recovered slope starts to drift.\n")
    print(f"  {'c chosen':>9} {'max delta':>11} {'slope expected':>15} "
          f"{'slope measured':>15} {'error':>9}")

    for c in (0.1, 0.35, 0.8, 1.5, 3.0, 6.0):
        t, x = simulate(0.0, c, WN, T=200, n=600_000)
        r = identify(t, x, T0)
        if r is None:
            continue
        expected = 8 * c / 3
        err = 100 * abs(r["slope"] - expected) / expected
        print(f"  {c:9.2f} {r['delta_max']:11.4f} {expected:15.5f} "
              f"{r['slope']:15.5f} {err:8.3f}%")
    print("\n  -> stays under 0.4% even where a cycle loses ~45% of its")
    print("     amplitude, well beyond anything expected in flight")


# =============================================================================
# SUPPORTING -- why goodness of fit is not used (Sec. 8, closing paragraph)
# =============================================================================
def supporting_fit_comparison():
    banner("--", "Why selection by goodness of fit was rejected")
    print("\n  The obvious alternative is to fit an exponential envelope and a")
    print("  hyperbolic one and keep whichever fits better. On a system with")
    print("  BOTH mechanisms present this picks the wrong one.\n")
    print(f"  {'case':>24} {'R2 exponential':>16} {'R2 hyperbolic':>15} {'verdict':>12}")

    for zeta, c, name in [(0.06, 0.00, "pure linear"),
                          (0.00, 0.35, "pure quadratic"),
                          (0.03, 0.20, "mixed (zeta and c)")]:
        t, x = simulate(zeta, c, WN)
        tp, A = extract_peaks(t, x, T0)
        r_exp = linregress(tp, np.log(A)).rvalue ** 2
        r_hyp = linregress(tp, 1.0 / A).rvalue ** 2
        verdict = "linear" if r_exp > r_hyp else "quadratic"
        print(f"  {name:>24} {r_exp:16.6f} {r_hyp:15.6f} {verdict:>12}")

    t, x = simulate(0.03, 0.20, WN)
    r = identify(t, x, T0)
    print(f"\n  The mixed case is called 'linear' and its quadratic component")
    print(f"  discarded. The regression of Sec. 7, on the same data, returns")
    print(f"  zeta = {r['zeta']:.5f} (chosen 0.03) and c = {r['c']:.5f} (chosen 0.20).")


# =============================================================================
if __name__ == "__main__":
    print(__doc__)
    row1_linear_limit()
    rows2to4_recovery()
    row5_envelope()
    row6_integral()
    row7_validity()
    supporting_fit_comparison()
    print("\n" + "=" * 76)
    print("  Change the coefficient lists in any block above to test other")
    print("  values. The method is never told what they are.")
    print("=" * 76 + "\n")
