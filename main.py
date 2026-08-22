"""
main.py -- orchestrator.

    python -m descent.main flight.csv --mass 0.35 --riser 0.50 \
           --bias-file bias.csv --design-cda 0.15

    python -m descent.main --selftest            # synthetic data, known answers
    python -m descent.main --selftest --coning

-----------------------------------------------------------------------------
 STAGE ORDER  (a requirement, not a convention)

   1  load           load.py           CSV -> typed DataFrame, integrity checks
   2  preprocess     preprocess.py     time base -> bias -> zero-phase filter
   3  oscillation    oscillation.py    PCA -> motion class -> damping
   4  aerodynamics   aerodynamics.py   density -> altitude -> velocity -> C_D*A
   5  assessment     assessment.py     design intent / consistency / stability
   6  plots          plot.py

 Stage 4 CONSUMES the oscillation period from stage 3: the analysis window is
 snapped to whole oscillation periods, and C_D*A is corrected for the
 oscillation bias. Running the two branches independently is incorrect.

-----------------------------------------------------------------------------
 DEBUGGING

 --stop-after {load,preprocess,oscillation,aerodynamics,assessment}
        halt after a stage and dump its output, so a failure can be isolated
        instead of propagating silently into every downstream number.

 --checkpoint DIR
        write each stage's output to DIR as it completes. If a later stage
        fails, the earlier outputs are already on disk to inspect.

 Any stage also runs standalone:
        python -m descent.load        flight.csv
        python -m descent.preprocess  flight.csv --bias-file bias.csv
        python -m descent.oscillation flight.csv --bias-file bias.csv
        python -m descent.aerodynamics flight.csv --no-oscillation
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from . import aerodynamics, assessment, load, oscillation, plot, preprocess
from .config import GYRO, Config

STAGES = ["load", "preprocess", "oscillation", "aerodynamics", "assessment", "plots"]


# =============================================================================
def _clean(o):
    """Strip numpy scalars and bulk arrays so results.json stays readable."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()
                if not (isinstance(v, np.ndarray) and v.size > 8)}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _checkpoint(name: str, obj, out_dir: str | None) -> None:
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"stage_{name}.json"), "w") as fh:
        json.dump(_clean(obj), fh, indent=2, default=float)


# =============================================================================
def run(csv_path: str, cfg: Config, out_dir: str = "results",
        bias_file: str | None = None, stop_after: str | None = None,
        checkpoint_dir: str | None = None, deploy_ms: int | None = None) -> dict:

    # --- 1 load -------------------------------------------------------------
    df, load_report = load.load_csv(csv_path)
    _checkpoint("1_load", load_report, checkpoint_dir)
    if stop_after == "load":
        return {"stopped_after": "load", "load": _clean(load_report)}

    bias = None
    if bias_file:
        bias, cfg.bias_temp_c = load.load_bias(bias_file)

    # --- 2 preprocess -------------------------------------------------------
    df, raw, pre = preprocess.run(df, cfg, bias, deploy_ms=deploy_ms)
    _checkpoint("2_preprocess", pre, checkpoint_dir)
    if checkpoint_dir:
        df.to_csv(os.path.join(checkpoint_dir, "stage_2_preprocessed.csv"),
                  index=False)
    if stop_after == "preprocess":
        return {"stopped_after": "preprocess", "load": _clean(load_report),
                "preprocessing": _clean(pre)}

    fs = pre["fs_hz"]

    # --- 3 oscillation (must precede stage 4) -------------------------------
    osc = oscillation.analyse(df, cfg, fs)
    _checkpoint("3_oscillation",
                {"pca": {k: v for k, v in osc["pca"].items()
                         if k not in ("pc1", "pc2")},
                 "damping": {k: v for k, v in osc["damping"].items()
                             if k != "peak_idx"},
                 "cross_check": osc["cross_check"]}, checkpoint_dir)
    if stop_after == "oscillation":
        return {"stopped_after": "oscillation",
                "oscillation": _clean({k: v for k, v in osc.items() if k != "fft"})}

    # --- 4 aerodynamics (consumes the period from stage 3) ------------------
    period = osc["damping"].get("period_s") if osc["damping"]["ok"] else None
    if period is None:
        print("  ! damping not identified -- aerodynamic window cannot be "
              "snapped to whole oscillation periods")
    aero = aerodynamics.analyse(df, cfg, fs, period)
    _checkpoint("4_aerodynamics",
                {k: v for k, v in aero.items()
                 if not isinstance(v, (np.ndarray, dict))}, checkpoint_dir)
    if stop_after == "aerodynamics":
        return {"stopped_after": "aerodynamics", "aerodynamics": _clean(aero)}

    # --- 5 assessment -------------------------------------------------------
    verdict = assessment.assess(cfg, osc, aero)
    _checkpoint("5_assessment", verdict, checkpoint_dir)
    if stop_after == "assessment":
        return {"stopped_after": "assessment", "assessment": _clean(verdict)}

    # --- 6 plots ------------------------------------------------------------
    figs = plot.save_all(plot.make_all(raw, df, pre, osc, aero), out_dir)

    results = _clean({
        "config": asdict(cfg),
        "load": load_report,
        "preprocessing": pre,
        "oscillation": {k: v for k, v in osc.items() if k != "fft"},
        "fft": {k: v for k, v in osc["fft"].items()
                if k in ("ok", "f_peak_hz", "resolution_hz")},
        "aerodynamics": {k: v for k, v in aero.items()
                         if k not in ("t", "rho", "altitude_m", "velocity",
                                      "steady_mask", "cda_series")},
        "assessment": verdict,
        "figures": figs,
    })
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results


# =============================================================================
def print_summary(r: dict) -> None:
    if "stopped_after" in r:
        print(f"\n[stopped after stage: {r['stopped_after']}]")
        print(json.dumps({k: v for k, v in r.items() if k != "stopped_after"},
                         indent=2, default=float)[:4000])
        return

    p, o, a, v = (r["preprocessing"], r["oscillation"],
                  r["aerodynamics"], r["assessment"])
    d, pca, L = o["damping"], o["pca"], r["load"]

    print("\n" + "=" * 68)
    print("  DESCENT ANALYSIS SUMMARY")
    print("=" * 68)
    print(f"  samples            {L['n_rows_raw']} over {L['duration_s']:.1f} s")
    print(f"  sample rate        {p['fs_hz']:.1f} Hz  (jitter {p['jitter_pct']:.2f}%)")
    print(f"  deployment at      t = {p['deployment_at_s']:.2f} s from record start "
          f"[{p['deployment_method']}]")
    print(f"  gyro bias          {np.round(p['bias_rads'], 5).tolist()} rad/s"
          f"  [{p['bias_source']}]")
    print(f"  pendulum estimate  {p['pendulum_estimate_hz']:.3f} Hz")
    print(f"  filter cutoff      {p['cutoff_hz']:.2f} Hz "
          f"({p['margin_over_estimate']:.1f}× estimate, "
          f"Nyquist {p['nyquist_hz']:.1f} Hz)")

    print("\n  -- oscillation --")
    print(f"  variance split     {[f'{x:.1%}' for x in pca['explained']]}")
    print(f"  classification     {pca['classification']}")
    if d["ok"]:
        print(f"  cycles used        {d['n_cycles']}")
        print(f"  period             {d['period_s']:.4f} s   "
              f"(f_d {d['f_damped_hz']:.4f} Hz, f_n {d['f_natural_hz']:.4f} Hz)")
        print(f"  δ range            {d['delta_min']:.4f} → {d['delta_max']:.4f}")
        print(f"  regression         intercept {d['intercept']:.5f}, "
              f"slope {d['slope']:.5f}, r={d['r_value']:.4f}, p={d['p_slope']:.2e}")
        print(f"  mechanism          {d['mechanism']}")
        print(f"  ζ (linear)         {d['zeta_linear']:.5f}")
        print(f"  c (quadratic)      {d['c_quadratic']:.5f}")
        print(f"  settling (2%)      {d['settling_time_2pct_s']:.2f} s  "
              f"[{d['settling_law']}]")
        if "caveat" in d:
            print(f"  ** {d['caveat']}")
    else:
        print(f"  damping FAILED: {d['reason']}")
    if o["cross_check"]:
        c = o["cross_check"]
        print(f"  FFT cross-check    {c['f_fft_hz']:.4f} Hz vs "
              f"{c['f_time_domain_hz']:.4f} Hz → "
              f"{'agree' if c['agree'] else 'DISAGREE'} "
              f"({c['disagreement_pct']:.2f}%)")
    e = o["energy"]
    print(f"  rotational energy  {e['ke_initial']:.4f} → {e['ke_final']:.4f}"
          f"  [{e['units']}]")

    print("\n  -- aerodynamics --")
    print(f"  air density        {a['rho_mean']:.4f} kg/m³")
    print(f"  terminal speed     {a['terminal_velocity_ms']:.3f} m/s")
    print(f"  transient trim     {a['t_skip_s']:.2f} s  (derived as 4·v_t/g)")
    if a["window_periods"]:
        print(f"  steady window      {a['window_periods']:.0f} whole oscillation periods")
    print(f"  η modulation       {a['eta_modulation']:.4f}  → bias "
          f"{a['oscillation_bias_pct']:.2f}% (corrected)")
    print(f"  C_D·A raw          {a['cda_raw_m2']:.5f} m²")
    print(f"  C_D·A corrected    {a['cda_corrected_m2']:.5f} m²  "
          f"(IQR {a['cda_iqr_m2']:.5f}, n={a['n_samples_used']})")
    print(f"  ballistic coeff    {a['ballistic_coefficient_kgm2']:.3f} kg/m²")
    print(f"  Reynolds range     {a['reynolds_range'][0]:.2e} – "
          f"{a['reynolds_range'][1]:.2e}")

    print("\n  -- assessment --")
    print(f"  1 design intent    {v['design_intent']['verdict']}")
    print(f"  2 consistency      {v['consistency']['verdict']}")
    for fl in v["consistency"]["flags"]:
        print(f"                     ! {fl}")
    print(f"  3 stability        {v['stability']['verdict']}")
    print("=" * 68)
    if "figures" in r and r["figures"]:
        print(f"  figures → {os.path.dirname(r['figures'][0])}\n")


# =============================================================================
def selftest(out_dir="results_selftest", coning=False) -> dict:
    """Run the whole chain on synthetic data with known answers."""
    from . import synth

    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, "synthetic_flight.csv")
    bias = os.path.join(out_dir, "bias_calibration.csv")
    truth = synth.make_synthetic(csv, bias, coning=coning)

    cfg = Config(mass_kg=0.35, riser_length_m=0.50, design_cda_m2=0.15)
    r = run(csv, cfg, out_dir, bias_file=bias, checkpoint_dir=out_dir)
    print_summary(r)

    d = r["oscillation"]["damping"]
    print("  -- self-test: recovered vs values used to generate the data --")
    print(f"  {'quantity':<24}{'truth':>12}{'recovered':>12}{'error':>10}")
    rows = [("damping ratio ζ", truth["zeta"], d.get("zeta_linear")),
            ("quadratic coeff c", truth["c"], d.get("c_quadratic")),
            ("natural freq (Hz)", truth["f_n"], d.get("f_natural_hz")),
            ("terminal speed (m/s)", truth["v_t"],
             r["aerodynamics"]["terminal_velocity_ms"]),
            ("drag area C_D·A (m²)", truth.get("cda"),
             r["aerodynamics"]["cda_corrected_m2"])]
    for key, tv, gv in rows:
        if gv is None or tv is None:
            continue
        print(f"  {key:<24}{tv:>12.4f}{gv:>12.4f}"
              f"{100 * abs(gv - tv) / abs(tv):>9.2f}%")
    print(f"  {'coning expected':<24}{str(truth['coning']):>12}   →  "
          f"{r['oscillation']['pca']['classification']}\n")
    return r


# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="CanSat descent analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="flight CSV from the SD card")
    ap.add_argument("--bias-file", default=None,
                    help="pre-flight gyro stillness calibration CSV")
    ap.add_argument("--deploy-ms", type=int, default=None,
                    help="override deployment instant (raw t_ms), instead of "
                         "inferring it from the accelerometer")
    ap.add_argument("--out", default="results")
    ap.add_argument("--mass", type=float, default=0.35, help="descending mass, kg")
    ap.add_argument("--riser", type=float, default=0.50,
                    help="attachment point to CG distance, m")
    ap.add_argument("--diameter", type=float, default=0.30,
                    help="canopy diameter, m (Reynolds number only)")
    ap.add_argument("--inertia", type=float, default=None, help="moment of inertia, kg m²")
    ap.add_argument("--design-cda", type=float, default=None, help="design C_D·A, m²")
    ap.add_argument("--cutoff", type=float, default=None, help="override filter cutoff, Hz")
    ap.add_argument("--margin", type=float, default=5.0, help="cutoff / pendulum estimate")
    ap.add_argument("--stop-after", choices=STAGES[:-1], default=None,
                    help="halt after a stage and dump its output")
    ap.add_argument("--checkpoint", default=None,
                    help="write each stage's output to this directory as it completes")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--coning", action="store_true", help="self-test with coning motion")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.out if args.out != "results" else "results_selftest",
                 coning=args.coning)
        return
    if not args.csv:
        ap.error("give a flight CSV, or use --selftest")

    cfg = Config(mass_kg=args.mass, riser_length_m=args.riser,
                 ref_diameter_m=args.diameter, inertia_kgm2=args.inertia,
                 design_cda_m2=args.design_cda,
                 cutoff_hz_override=args.cutoff, cutoff_margin=args.margin)
    print_summary(run(args.csv, cfg, args.out, bias_file=args.bias_file,
                      stop_after=args.stop_after, checkpoint_dir=args.checkpoint,
                      deploy_ms=args.deploy_ms))


if __name__ == "__main__":
    main()
