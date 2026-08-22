"""
plot.py -- STAGE 6: figures.

Each function returns a matplotlib Figure, so they can be saved, shown or
embedded without this module deciding for you. save_all() writes them out.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

from .config import GYRO

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def gyro_filter(raw, df, pre):
    f, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for a, c in zip(ax, GYRO):
        a.plot(raw["t"], raw[c], lw=.5, alpha=.45, label="de-biased raw")
        a.plot(df["t"], df[c], lw=1.2, label="zero-phase filtered")
        a.set_ylabel(f"{c} (rad/s)")
    ax[0].set_title(f"Angular velocity — {pre['filter_order']}th-order zero-phase "
                    f"Butterworth, fc = {pre['cutoff_hz']:.2f} Hz "
                    f"({pre['margin_over_estimate']:.1f}× pendulum estimate)")
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel("time since deployment (s)")
    f.tight_layout()
    return f


def pca(osc):
    p = osc["pca"]
    f, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 3.6))
    a1.bar(["PC1", "PC2", "PC3"], p["explained"],
           color=["#2b6cb0", "#63b3ed", "#bee3f8"])
    for i, v in enumerate(p["explained"]):
        a1.text(i, v + .02, f"{v:.1%}", ha="center", fontsize=8)
    a1.set_ylim(0, 1.12); a1.set_ylabel("explained variance")
    a1.set_title("Variance by component")

    a2.plot(p["pc1"], p["pc2"], lw=.6, alpha=.8)
    a2.set_xlabel("PC1"); a2.set_ylabel("PC2")
    a2.set_aspect("equal", adjustable="datalim")
    a2.set_title(f"Phase plot — ev2/ev1 = {p['ev_ratio_21']:.2f}")

    a3.axis("off")
    a3.text(0, .7, "Motion classification", fontsize=10, weight="bold",
            transform=a3.transAxes)
    a3.text(0, .45, p["classification"], fontsize=9, transform=a3.transAxes)
    a3.text(0, .15, "circular phase plot = coning\nline = planar swing",
            fontsize=8, alpha=.75, transform=a3.transAxes)
    f.tight_layout()
    return f


def damping(df, osc):
    """The discriminator: per-cycle decrement against cycle amplitude.

    A flat line means linear damping; a slope means quadratic. This is the
    figure that shows the mechanism directly, before any model is fitted.
    """
    t = df["t"].to_numpy()
    p, d = osc["pca"], osc["damping"]
    x = detrend(p["pc1"])

    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(t, x, lw=.9, label="PC1 (dominant mode)")
    if d["ok"]:
        a1.plot(t[d["peak_idx"]], x[d["peak_idx"]], "o", ms=3.5,
                color="crimson", label="peaks used")
    a1.set_xlabel("time since deployment (s)")
    a1.set_ylabel("angular rate (rad/s)")
    a1.legend(fontsize=8); a1.set_title("Extracted oscillation")

    if d["ok"]:
        A = x[d["peak_idx"]]
        delta = np.log(A[:-1] / A[1:])
        Abar = np.sqrt(A[:-1] * A[1:])
        a2.plot(Abar, delta, "o", ms=5, alpha=.75)
        xs = np.linspace(0, Abar.max(), 50)
        a2.plot(xs, d["intercept"] + d["slope"] * xs, "-", lw=1.5, color="crimson")
        a2.set_xlabel("geometric-mean amplitude  Ā")
        a2.set_ylabel("per-cycle log decrement  δ")
        a2.set_title(f"Damping identification — {d['mechanism']}\n"
                     f"intercept {d['intercept']:.4f} (ζ={d['zeta_linear']:.4f}), "
                     f"slope {d['slope']:.4f} (c={d['c_quadratic']:.4f}), "
                     f"p={d['p_slope']:.1e}")
    else:
        a2.text(.5, .5, f"damping not identified:\n{d['reason']}",
                ha="center", va="center", transform=a2.transAxes)
    f.tight_layout()
    return f


def fft(osc):
    ft, d = osc["fft"], osc["damping"]
    f, ax = plt.subplots(figsize=(8, 3.6))
    if ft["ok"]:
        keep = ft["freqs"] <= max(5 * ft["f_peak_hz"], 2.0)
        ax.plot(ft["freqs"][keep], ft["mag"][keep], lw=1)
        ax.axvline(ft["f_peak_hz"], color="crimson", ls="--",
                   label=f"FFT peak {ft['f_peak_hz']:.3f} Hz")
        if d["ok"]:
            ax.axvline(d["f_damped_hz"], color="seagreen", ls=":", lw=1.4,
                       label=f"time-domain {d['f_damped_hz']:.3f} Hz")
        ax.legend(fontsize=8)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|amplitude|")
    ax.set_title("Independent frequency estimate")
    f.tight_layout()
    return f


def altitude_velocity_density(df, aero):
    f, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = aero["t"]

    ax[0].plot(t, aero["altitude_m"], lw=1.2,
               label="barometric (hypsometric, measured T)")
    if "gps_alt_m" in df and "gps_fix" in df:
        m = df["gps_fix"].to_numpy(bool)
        ax[0].plot(t[m], df["gps_alt_m"].to_numpy()[m], lw=.7, alpha=.6,
                   label="GPS (cross-check only)")
    ax[0].set_ylabel("altitude (m)"); ax[0].legend(fontsize=8)

    v = aero["velocity"]["descent_speed"]
    ax[1].plot(t, v, lw=1)
    lbl = ("quasi-steady window"
           + (f" ({aero['window_periods']:.0f} periods)"
              if aero["window_periods"] else ""))
    ax[1].fill_between(t, 0, v, where=aero["steady_mask"], alpha=.12, label=lbl)
    ax[1].set_ylabel("descent speed (m/s)"); ax[1].legend(fontsize=8)

    ax[2].plot(t, aero["rho"], lw=1.2, color="darkslateblue")
    ax[2].set_ylabel("air density (kg/m³)")
    ax[2].set_xlabel("time since deployment (s)")
    f.tight_layout()
    return f


def drag_area(aero):
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    m = aero["steady_mask"]
    a1.plot(aero["t"][m], aero["cda_series"][m], lw=.8)
    a1.axhline(aero["cda_corrected_m2"], color="crimson", ls="--",
               label=f"corrected median {aero['cda_corrected_m2']:.4f} m²")
    a1.set_xlabel("time since deployment (s)"); a1.set_ylabel("C_D·A (m²)")
    a1.legend(fontsize=8)
    a1.set_title("Effective drag area over the steady window")

    fin = aero["cda_series"][np.isfinite(aero["cda_series"])]
    a2.hist(fin, bins=40, color="#2b6cb0", alpha=.85)
    a2.axvline(aero["cda_corrected_m2"], color="crimson", ls="--")
    a2.set_xlabel("C_D·A (m²)"); a2.set_ylabel("count")
    a2.set_title(f"β = {aero['ballistic_coefficient_kgm2']:.2f} kg/m²   "
                 f"(η = {aero['eta_modulation']:.3f}, bias "
                 f"{aero['oscillation_bias_pct']:.2f}%)")
    f.tight_layout()
    return f


def make_all(raw, df, pre, osc, aero) -> dict:
    return {
        "01_gyro_filter":              gyro_filter(raw, df, pre),
        "02_pca":                      pca(osc),
        "03_damping":                  damping(df, osc),
        "04_fft":                      fft(osc),
        "05_altitude_velocity_density": altitude_velocity_density(df, aero),
        "06_drag_area":                drag_area(aero),
    }


def save_all(figs: dict, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, fig in figs.items():
        p = os.path.join(out_dir, f"{name}.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
