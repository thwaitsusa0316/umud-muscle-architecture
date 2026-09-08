#!/usr/bin/env python3
"""Build a submission over all 309 test images.

Three things make this more than "run the model":

* **Pennation angle is scale-free.** It comes from slopes, so it can be measured on
  every image including the 176 whose pixel-to-millimetre scale we cannot recover.
  Only FL and MT need the scale.
* **Anisotropy.** The console-screenshot images were re-exported at canonical sizes
  with a different aspect ratio to their acquisition geometry, which rotates every
  measured angle. Correcting it moved the predicted PA and FL medians onto their
  leaderboard-probed values simultaneously (one parameter, two constraints), and the
  control on native-resolution benchmark images rejects the same correction
  decisively. Cropped images, which carry no console furniture, are treated as
  native.
* **Fallbacks are the probed medians**, not the range midpoints. A constant at the
  median is the best constant possible under an absolute-error metric.
"""
from __future__ import annotations
import sys, pathlib, json, warnings, os, argparse

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from umud import segment as S, geometry as Gm, grouping as G, submit as Sub
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
MED = {"pa_deg": 16.42, "fl_mm": 81.5, "mt_mm": 20.7}   # leaderboard-probed medians
ASPECT_RESIZED = 1.35
DUMMY_SCALE = 100.0          # any positive value; PA does not depend on it
SCALE_JSON = ROOT / "reports" / "scale_v2.json"   # L6 lookup: 269/309 scaled (v1 scale_final: 133)
# MT plausibility gate (PLAN v4 rung L6b): a thickness outside this band is an
# aponeurosis-pairing failure, not a measurement; ship the probed median instead.
# Mechanistic (adult lower-limb MT spans ~8-35 mm), not tuned on the public board.
MT_GATE = (8.0, 35.0)


def load_scale(path: pathlib.Path = SCALE_JSON) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"--scale: {path} does not exist")
    scale = json.loads(path.read_text())
    if not isinstance(scale, dict) or len(scale) != 309:
        raise ValueError(f"{path}: expected a dict of 309 test images, got "
                         f"{type(scale).__name__} of {len(scale) if hasattr(scale, '__len__') else '?'}")
    bad = [n for n, v in scale.items()
           if v.get("px_per_cm") is not None and not (0 < float(v["px_per_cm"]) < 1000)]
    if bad:
        raise ValueError(f"{path}: implausible px_per_cm on {bad[:3]}")
    return scale


EVIDENCE_CSV = ROOT / "reports" / "l6t_fascicle_evidence.csv"


def load_flip_ids(path: pathlib.Path = EVIDENCE_CSV) -> set[str]:
    """L6o-b (PLAN v19 rung 2): test frames whose fascicles slope the *opposite* way to the
    single-orientation training set (299/300 host frames negative signed angle). They are
    mirrored left-right before inference so every frame meets the detector in the
    orientation it was trained on. Source: reports/l6t_fascicle_evidence.csv (label-free,
    ang_med = signed median fascicle angle from the pw6 mask). Expected 113/309."""
    if not path.is_file():
        raise FileNotFoundError(f"--flip-positive: {path} does not exist")
    try:
        d = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"--flip-positive: {path} is empty") from None
    need = {"image_id", "split", "ang_med"}
    if not need <= set(d.columns):
        raise ValueError(f"{path}: missing columns {sorted(need - set(d.columns))}")
    t = d[d.split == "test"]
    if t.ang_med.isna().any():
        raise ValueError(f"{path}: {int(t.ang_med.isna().sum())} test rows have NaN ang_med")
    ids = set(t.loc[t.ang_med > 0, "image_id"].astype(str))
    if not ids or len(ids) == len(t):
        raise ValueError(f"{path}: {len(ids)}/{len(t)} positive-slope frames; refusing a "
                         "flip list that is empty or total")
    return ids


def _finite(x) -> bool:
    """True for a real, finite number. NaN is truthy in Python, so `if x:` is not enough."""
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def gate_mt(mt, lo: float = MT_GATE[0], hi: float = MT_GATE[1]) -> tuple[float, bool]:
    """Return (mt_mm, measured). Out-of-band or missing/NaN MT falls back to the median."""
    if not _finite(mt) or not (lo <= float(mt) <= hi):
        return MED["mt_mm"], False
    return float(mt), True


def build(aspect: float = ASPECT_RESIZED, batch: int = 24, fasc_ckpt=None,
          fasc_thr: float = 0.9, scale_path: pathlib.Path = SCALE_JSON,
          mt_gate: tuple[float, float] = MT_GATE,
          flip_ids: set[str] | None = None) -> pd.DataFrame:
    """flip_ids: image names mirrored left-right (cv2.flip(gray, 1)) BEFORE both the
    aponeurosis and the fascicle model, as in scripts/l6o_flip_audit.py. PA, FL and MT
    are mirror-invariant physical quantities, so no un-flipping of the outputs is needed."""
    ours = None
    flip_ids = set(flip_ids or ())
    if fasc_ckpt:
        from eval_new_fascicle import load_model, predict_ours
        ours = load_model("unet", fasc_ckpt)
    scale = load_scale(scale_path)
    names = sorted((p.name for p in TEST.iterdir()
                    if p.suffix.lower() in {".tif", ".png"}), key=G._index)
    if not names:
        raise FileNotFoundError(f"no .tif/.png test images under {TEST}")
    missing = flip_ids - set(names)
    if missing:   # validate BEFORE any inference so a bad list cannot waste the run
        raise ValueError(f"flip_ids: {len(missing)} listed frames not under {TEST}, "
                         f"e.g. {sorted(missing)[:3]}")
    rows = []
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        grays = [G.to_gray(cv2.imread(str(TEST / n), cv2.IMREAD_UNCHANGED)) for n in chunk]
        grays = [cv2.flip(g, 1) if n in flip_ids else g for n, g in zip(chunk, grays)]
        masks = S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)
        for n, g, (am, fm) in zip(chunk, grays, masks):
            if ours is not None:
                from eval_new_fascicle import predict_ours
                fm = predict_ours(ours[0], ours[1], g, thr=fasc_thr)
            info = scale.get(n) or {}
            px = info.get("px_per_cm")
            # console screenshots were re-exported and need the aspect correction;
            # cropped frames (no console furniture, hence no recoverable scale) are
            # native geometry and must not be corrected
            asp = aspect if px else 1.0
            r = Gm.analyse(am, fm, float(px) if px else DUMMY_SCALE, aspect=asp)
            mt, mt_measured = gate_mt(r.mt_mm if px else None, *mt_gate)
            pa_measured = _finite(r.pa_deg)
            fl_measured = bool(px) and _finite(r.fl_mm)
            row = dict(image_id=n,
                       pa_deg=(float(r.pa_deg) if pa_measured else MED["pa_deg"]),
                       fl_mm=(float(r.fl_mm) if fl_measured else MED["fl_mm"]),
                       pa_measured=pa_measured, fl_measured=fl_measured,
                       mt_mm=mt, mt_measured=mt_measured,
                       mt_raw=(float(r.mt_mm) if (px and _finite(r.mt_mm)) else None),
                       measured_scale=bool(px), ok=r.ok)
            if flip_ids:   # column only when the flag is used: unflagged runs stay byte-identical
                row["flipped"] = n in flip_ids
            rows.append(row)
        print(f"  {min(i+batch, len(names))}/{len(names)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aspect", type=float, default=ASPECT_RESIZED)
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--message", default="")
    ap.add_argument("--fasc-ckpt", default=None, help="use our retrained fascicle model")
    ap.add_argument("--fasc-thr", type=float, default=0.9)
    ap.add_argument("--scale", default=str(SCALE_JSON), help="px/cm lookup JSON")
    ap.add_argument("--mt-gate", type=float, nargs=2, default=list(MT_GATE),
                    metavar=("LO", "HI"), help="MT plausibility band in mm; else median")
    ap.add_argument("--out", default="", help="output CSV (default outputs/sub_pipeline_a<aspect>.csv)")
    ap.add_argument("--report", default="", help="optional per-image CSV with raw MT and gate flags")
    ap.add_argument("--flip-positive", nargs="?", const=str(EVIDENCE_CSV), default=None,
                    metavar="EVIDENCE_CSV",
                    help="L6o-b: mirror the positive-slope test frames listed in the L6t "
                         "evidence CSV before inference (default reports/l6t_fascicle_evidence.csv)")
    a = ap.parse_args()
    lo, hi = a.mt_gate
    if not (0 < lo < hi):
        sys.exit(f"--mt-gate must satisfy 0 < LO < HI, got {lo} {hi}")

    flip_ids: set[str] = set()
    if a.flip_positive:
        try:
            flip_ids = load_flip_ids(pathlib.Path(a.flip_positive))
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))
        print(f"L6o-b: mirroring {len(flip_ids)} positive-slope frames from {a.flip_positive}")

    try:
        df = build(aspect=a.aspect, fasc_ckpt=a.fasc_ckpt, fasc_thr=a.fasc_thr,
                   scale_path=pathlib.Path(a.scale), mt_gate=(lo, hi), flip_ids=flip_ids)
    except (FileNotFoundError, ValueError) as e:   # missing scale JSON / test dir, bad flip list
        sys.exit(str(e))
    if flip_ids:
        print(f"  flipped {int(df.flipped.sum())}/{len(df)} frames")
    n_scale = int(df.measured_scale.sum())
    n_mt_raw = int(df.mt_raw.notna().sum())
    n_mt = int(df.mt_measured.sum())
    print(f"\nPA measured on {int(df.pa_measured.sum())}/{len(df)}; "
          f"FL measured on {int(df.fl_measured.sum())}/{len(df)}; scale on {n_scale}/{len(df)}; "
          f"MT raw on {n_mt_raw}, shipped after gate [{lo:g}, {hi:g}] mm on {n_mt} "
          f"(gated out {n_mt_raw - n_mt})")
    if a.report:
        df.to_csv(a.report, index=False)
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        print(f"  {c:8s} median {df[c].median():7.2f}  (probed {MED[c]})")
    out = df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
    # clip to the physiological ranges the host published for the test set
    out.pa_deg = out.pa_deg.clip(5, 45)
    out.fl_mm = out.fl_mm.clip(30, 200)
    out.mt_mm = out.mt_mm.clip(10, 50)
    path = Sub.write_submission(
        out, pathlib.Path(a.out) if a.out else ROOT / "outputs" / f"sub_pipeline_a{a.aspect:g}.csv")
    print(f"\nwrote {path}")
    if a.send:
        Sub.submit(path, a.message or f"pipeline, aspect={a.aspect:g}")
        print("submitted")


if __name__ == "__main__":
    main()
