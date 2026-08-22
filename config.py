"""
config.py -- physical constants and the run configuration.

Everything the analysis needs that is NOT in the flight CSV lives here, so
there are no magic numbers buried in the stage modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# --- physical constants ------------------------------------------------------
G        = 9.80665       # m/s^2
R_DRY    = 287.058       # J/(kg K)   specific gas constant, dry air
R_VAPOUR = 461.495       # J/(kg K)   specific gas constant, water vapour
MU_AIR   = 1.85e-5       # Pa s       dynamic viscosity, ~25 C

# --- axis / column names shared across stages --------------------------------
GYRO = ["gx", "gy", "gz"]            # radians/s, after conversion on load
ACC  = ["ax_ms2", "ay_ms2", "az_ms2"]


@dataclass
class Config:
    # --- vehicle, measured on the ground after recovery ---------------------
    mass_kg: float = 0.35            # total descending mass
    riser_length_m: float = 0.50     # parachute attachment point -> vehicle CG
    ref_diameter_m: float = 0.30     # canopy diameter, Reynolds number only
    inertia_kgm2: float | None = None  # None -> energy reported as KE/I

    # --- design intent, for the design-comparison criterion -----------------
    design_cda_m2: float | None = None

    # --- gyro bias ----------------------------------------------------------
    # Preferred source is a pre-flight stillness recording (--bias-file).
    # These constants are the fallback only.
    bias_rads: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bias_temp_c: float | None = None

    # --- filter -------------------------------------------------------------
    filter_order: int = 4
    cutoff_margin: float = 5.0       # cutoff = margin x pendulum estimate
    cutoff_hz_override: float | None = None

    # --- windows ------------------------------------------------------------
    min_periods_in_window: int = 5   # C_D*A window must span at least this many
    skip_end_s: float = 2.0          # trim ground impact
    velocity_window_s: float = 1.0   # Savitzky-Golay differentiation window

    # --- quality gates ------------------------------------------------------
    mad_threshold: float = 5.0       # outlier rejection
    accel_gate_frac: float = 0.15    # ||a|-g| > this*g flags non-quasi-steady

    # ------------------------------------------------------------------------
    def pendulum_frequency_hz(self) -> float:
        """First-order estimate, f = (1/2pi) sqrt(g/L).

        Idealised: point mass, undamped, planar, small angle. Used ONLY to
        place the filter cutoff -- never as a substitute for the measured
        frequency, which is compared against it afterwards.
        """
        return (1.0 / (2.0 * np.pi)) * np.sqrt(G / self.riser_length_m)

    def cutoff_hz(self, fs_hz: float) -> float:
        fc = (self.cutoff_hz_override if self.cutoff_hz_override is not None
              else self.cutoff_margin * self.pendulum_frequency_hz())
        return float(min(fc, 0.4 * fs_hz))          # stay clear of Nyquist

    def terminal_skip_s(self, v_t: float) -> float:
        """Deployment transient trim, derived rather than assumed.

        Linearising m dv/dt = mg - 0.5 rho C_D A v^2 about terminal velocity
        gives an exponential approach with tau = v_t/(2g). The factor 2 comes
        from drag varying as v^2: being 1% fast means 2% extra drag. Trimming
        8 tau leaves under 0.05% residual bias.
        """
        return 8.0 * v_t / (2.0 * G)
