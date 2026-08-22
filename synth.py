"""
synth.py -- synthetic flight generator for the self-test.

Builds a flight CSV in the real logged format whose damping coefficients and
drag area are CHOSEN IN ADVANCE, so the pipeline can be exercised before any
real data exists and the recovered values can be checked against known ones.

The atmosphere is built to be internally consistent, which matters more than
it sounds: three separate inconsistencies here (a non-hydrostatic pressure
profile, constant velocity through rising density, and a dry-air hydrostatic
integral paired with moist density) each produced apparent pipeline errors of
0.5-6% during development before being traced back to this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .config import G, R_DRY
from .aerodynamics import air_density, virtual_temperature


def make_synthetic(path: str, bias_path: str, zeta=0.030, c_quad=0.200,
                   v_t=6.30, f_n=0.62, coning=False, seed=0) -> dict:
    """Generate a flight CSV whose damping coefficients are known in advance.

    Lets the whole chain be exercised before any real data exists, and lets
    the recovered coefficients be checked against values chosen here.
    """
    rng = np.random.default_rng(seed)
    fs, T = 100.0, 70.0
    n = int(fs * T)
    t = np.arange(n) / fs
    wn = 2 * np.pi * f_n

    # exact damped oscillator, both mechanisms
    def rhs(_t, y):
        th, w = y
        return [w, -2 * zeta * wn * w - c_quad * abs(w) * w - wn ** 2 * th]
    sol = solve_ivp(rhs, (0, T), [0.35, 0.0], t_eval=t, rtol=1e-10,
                    atol=1e-12, method="DOP853")
    th, w = sol.y

    if coning:
        body = np.vstack([w, np.roll(w, int(0.25 / f_n * fs)), np.zeros_like(w)])
    else:
        body = np.vstack([w, np.zeros_like(w), np.zeros_like(w)])

    # rotate so the swing aligns with no single IMU axis
    cy, sy = np.cos(np.radians(35)), np.sin(np.radians(35))
    cp, sp = np.cos(np.radians(20)), np.sin(np.radians(20))
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ \
        np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    gyro = (R @ body).T + rng.normal(0, 0.02, (n, 3))

    true_bias_dps = np.array([0.42, -0.31, 0.18])
    gyro_dps = np.degrees(gyro) + true_bias_dps

    # descent: terminal velocity with the swing modulating vertical speed
    # --- self-consistent descent -------------------------------------------
    # The vehicle must descend at the terminal speed for the LOCAL density,
    #     v_t(z) = sqrt(2 m g / (rho(z) C_D A)),
    # and rho depends on altitude, which depends on v. Holding v constant
    # while density rises would mean C_D*A is genuinely falling through the
    # descent -- an inconsistency the analysis would correctly report as a
    # trend. Iterate to convergence instead.
    mass, cda_true = 0.35, 0.15
    p_land = 100_500.0
    swing = th / max(abs(th).max(), 1e-9)
    v = np.full(n, v_t)
    alt = pressure = rho = temp_c = rh = None

    # Iterate to a fixed point, then take ONE more pass so that the altitude,
    # pressure, density and velocity handed out are mutually consistent. If v
    # is updated after alt was built from the previous v, the arrays are one
    # iteration out of step and the "true" velocity does not match the
    # altitude profile the analysis will differentiate.
    for it in range(12):
        alt = 420.0 - np.cumsum(v) / fs
        temp_c = 24.0 + 0.0065 * (alt - alt[-1])
        rh = np.full(n, 55.0)

        # Pressure by HYDROSTATIC INTEGRATION, d(lnP) = -g/(R T(z)) dz.
        # p = p_land*exp(-g*dh/(R*T_local)) is not a consistent atmosphere
        # when T varies with height and cannot be inverted correctly.
        # Virtual temperature, so the profile is moist-consistent with the
        # density used for the dynamics. Building it with dry-air R while
        # computing rho moist would leave the atmosphere inconsistent by
        # exactly the humidity factor (~0.7% at 55% RH, 24 C).
        p_guess = p_land * np.exp(-G * (alt - alt.min()) / (R_DRY * (temp_c + 273.15)))
        T_abs = virtual_temperature(p_guess, temp_c, rh)
        order = np.argsort(alt)
        h_s, T_s = alt[order], T_abs[order]
        integ = -G / (R_DRY * T_s)
        lnp_s = np.concatenate([[0.0], np.cumsum(np.diff(h_s) * 0.5 *
                                                 (integ[1:] + integ[:-1]))])
        lnp = np.empty(n); lnp[order] = lnp_s
        lnp -= lnp[int(np.argmin(alt))]
        pressure = p_land * np.exp(lnp)

        rho = air_density(pressure, temp_c, rh)
        if it == 11:
            break                       # alt/pressure/rho now match this v
        v = np.sqrt(2 * mass * G / (rho * cda_true)) * (1 + 0.10 * swing)

    temp_c = temp_c + rng.normal(0, .05, n)
    rh = np.clip(rh + rng.normal(0, .8, n), 0, 100)
    pressure = pressure + rng.normal(0, 3, n)
    pressure[(t > .4) & (t < .7)] += 120.0          # deployment shock spike
    # Quote terminal speed over the SAME window the analysis uses (the tail
    # half), since density rises through the descent and v_t falls with it.
    tail = t >= 0.5 * t[-1]
    v_t = float(np.median(np.sqrt(2 * mass * G / (rho[tail] * cda_true))))

    acc = rng.normal(0, .25, (n, 3)); acc[:, 2] += G
    acc[0, 2] += 8 * G                              # deployment signature

    gps = alt + rng.normal(0, 4, n)
    gps[rng.random(n) < 0.05] = 0.0                 # dropped fixes

    pd.DataFrame({
        "t_ms": np.round((t + 12.0) * 1000).astype(np.int64),   # uptime, not 0
        "gx_dps": gyro_dps[:, 0], "gy_dps": gyro_dps[:, 1], "gz_dps": gyro_dps[:, 2],
        "ax_ms2": acc[:, 0], "ay_ms2": acc[:, 1], "az_ms2": acc[:, 2],
        "pressure_pa": pressure, "temp_c": temp_c, "rh_pct": rh,
        "gps_alt_m": gps,
    }).to_csv(path, index=False)

    # pre-flight stillness calibration
    m = 500
    pd.DataFrame({
        "gx_dps": true_bias_dps[0] + rng.normal(0, .05, m),
        "gy_dps": true_bias_dps[1] + rng.normal(0, .05, m),
        "gz_dps": true_bias_dps[2] + rng.normal(0, .05, m),
        "temp_c": np.full(m, 23.5),
    }).to_csv(bias_path, index=False)

    return {"zeta": zeta, "c": c_quad, "f_n": f_n, "v_t": v_t, "cda": cda_true,
            "bias_dps": true_bias_dps.tolist(), "coning": coning}
