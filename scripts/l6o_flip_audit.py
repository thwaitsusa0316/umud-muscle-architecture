#!/usr/bin/env python3
"""L6o: is the PA/FL pipeline orientation-dependent?  (PLAN v19 rung 1, label-free on test)

L6t found the training set single-orientation (299/300 host frames have fascicles sloping
one way, signed angle median -13.4 deg) while 113/309 test frames -- two thirds of the
stills -- are mirrored (positive slope).  The pipeline's PA/FL skill was only ever measured
on the 35 expert-benchmark frames, which share the training orientation.

This audit runs the *unchanged* production pipeline (DL_Track aponeurosis model + our pw6
fascicle U-Net + Gm.analyse) on each benchmark frame twice: as delivered and mirrored
left-right (cv2.flip(g, 1)).  PA, FL and MT are mirror-invariant physical quantities, so the
expert consensus is the same truth for both; any change in MAE / Spearman vs GT is the
pipeline's orientation dependence, isolated as one variable.

Pre-registered read (written before the run): orientation dependence is CONFIRMED if, on
the flipped frames, PA MAE > 2.0 deg (as-delivered ~1.16, constant 2.75) OR PA Spearman
< 0.6 OR the usable-row count drops by more than 5 of 35.  If flipped PA MAE <= 1.6 and
Spearman >= 0.8 and FL MAE <= 6.5 mm the pipeline is orientation-robust and mirroring is
NOT the mechanism behind the missing test-domain PA/FL skill.

Writes reports/l6o_flip_audit.csv (one row per benchmark image: GT, as-delivered and
flipped predictions, fascicle orientation sign and count in each) and prints the summary.
"""
import sys, os, pathlib, argparse, warnings
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from scipy.stats import spearmanr
from eval_new_fascicle import load_model, predict_ours
from l6t_fascicle_evidence import mask_stats
from umud import validation as V, segment as S, geometry as Gm

TARGETS = ("pa_deg", "fl_mm", "mt_mm")


def measure(gray, am, fm, px_per_cm):
    res = Gm.analyse(am, fm, float(px_per_cm))
    st = mask_stats(fm, am)
    return dict(pa_deg=res.pa_deg, fl_mm=res.fl_mm, mt_mm=res.mt_mm,
                ang_med=st["ang_med"], n_comp=st["n_comp"], frac_band=st["frac_band"])


def summary(df: pd.DataFrame, tag: str) -> dict:
    """Per-target MAE / bias / Spearman vs GT on the rows the pipeline measured (non-NaN)."""
    out = {"variant": tag}
    for c in TARGETS:
        p, t = df[f"{c}_{tag}"], df[f"{c}_true"]
        ok = p.notna() & t.notna()
        out[f"{c}_n"] = int(ok.sum())
        if ok.sum() >= 3:
            err = (p[ok] - t[ok])
            out[f"{c}_mae"] = float(err.abs().mean())
            out[f"{c}_bias"] = float(err.mean())
            out[f"{c}_rho"] = float(spearmanr(p[ok], t[ok])[0])
        else:
            out[f"{c}_mae"] = out[f"{c}_bias"] = out[f"{c}_rho"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.9, help="fascicle probability threshold (production: make_submission.py --fasc-thr 0.9)")
    ap.add_argument("--ckpt", default=None, help="fascicle U-Net checkpoint (default: eval_new_fascicle's pw6 default)")
    ap.add_argument("--limit", type=int, default=None, help="first N benchmark frames (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "l6o_flip_audit.csv"))
    a = ap.parse_args()

    model, dev = load_model("unet", a.ckpt)
    t = V.load()
    if a.limit:
        t = t.head(a.limit).reset_index(drop=True)
    grays = []
    for p in t.path:
        g = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if g is None:
            raise SystemExit(f"cannot read {p}")
        grays.append(g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY))
    flips = [cv2.flip(g, 1) for g in grays]
    # aponeurosis masks from the same DL_Track model and thresholds as predict_test.py
    apo_o = [am for am, _ in S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)]
    apo_f = [am for am, _ in S.predict_batch(flips, thr_apo=0.35, thr_fasc=0.10)]

    rows = []
    for i, (_, r) in enumerate(t.iterrows()):
        fm_o = predict_ours(model, dev, grays[i], thr=a.thr)
        fm_f = predict_ours(model, dev, flips[i], thr=a.thr)
        mo = measure(grays[i], apo_o[i], fm_o, r.px_per_cm)
        mf = measure(flips[i], apo_f[i], fm_f, r.px_per_cm)
        row = dict(image_id=r.image_id, px_per_cm=float(r.px_per_cm))
        for c in TARGETS:
            row[f"{c}_true"] = float(r[c])
        for k, v in mo.items():
            row[f"{k}_orig"] = v
        for k, v in mf.items():
            row[f"{k}_flip"] = v
        rows.append(row)
        print(f"  {i + 1}/{len(t)} {r.image_id}: PA {mo['pa_deg']} -> {mf['pa_deg']} (GT {r.pa_deg:.2f}); "
              f"ang {mo['ang_med']} -> {mf['ang_med']}", flush=True)
    df = pd.DataFrame(rows)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    summ = pd.DataFrame([summary(df, "orig"), summary(df, "flip")]).set_index("variant")
    pd.set_option("display.width", 200)
    print("\nper-target vs expert consensus (n = rows the pipeline measured):")
    print(summ.round(3).T.to_string())
    neg_o = int((df.ang_med_orig.dropna() < 0).sum()); neg_f = int((df.ang_med_flip.dropna() < 0).sum())
    print(f"\nfascicle slope sign: as delivered {neg_o}/{df.ang_med_orig.notna().sum()} negative (training-like); "
          f"flipped {neg_f}/{df.ang_med_flip.notna().sum()} negative")
    both = df.pa_deg_orig.notna() & df.pa_deg_flip.notna()
    if both.sum():
        d = (df.pa_deg_flip - df.pa_deg_orig)[both]
        print(f"paired PA flip-orig on {int(both.sum())} rows: median {d.median():.2f} deg, median |d| {d.abs().median():.2f}")
    s = summ.loc["flip"]
    n_drop = int(summ.loc["orig", "pa_deg_n"] - s["pa_deg_n"])
    confirmed = bool((s["pa_deg_mae"] > 2.0) or (s["pa_deg_rho"] < 0.6) or (n_drop > 5))
    robust = bool((s["pa_deg_mae"] <= 1.6) and (s["pa_deg_rho"] >= 0.8) and (s["fl_mm_mae"] <= 6.5))
    verdict = ("ORIENTATION-DEPENDENT (pre-registered threshold met)" if confirmed
               else "ORIENTATION-ROBUST (pre-registered threshold met)" if robust
               else "INCONCLUSIVE (between the pre-registered bands)")
    print(f"\nverdict: {verdict}  [flip PA MAE {s['pa_deg_mae']:.3f}, rho {s['pa_deg_rho']:.3f}, usable drop {n_drop}]")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
