#!/usr/bin/env python3
"""Label-domain pennation angle on the test frame type (PLAN v5 rung L6c, PA bias).

Our scaled pipeline measures a median PA of 13.0 deg on the 309 test images against
the leaderboard-probed median of 16.42 deg (-21 %). Two mechanisms could produce a
constant offset that large:

  (a) the delivered 800x1200 Siemens frames are anisotropically re-exported, so every
      angle traced on them is flatter than the physical angle the labels carry; or
  (b) our fascicle detector draws flatter fascicles than the host's annotators on this
      frame type (it is unbiased on the native benchmark: PA bias -0.20 deg, n = 34).

The host's own fascicle masks on the 800x1200 training frames separate the two. PA
from the HOST mask (with our predicted aponeuroses) is the label-domain angle on the
delivered geometry, independent of our detector: if it sits near 13 deg the frames
themselves are flat (a); if it sits near 16.4 deg the detector is at fault (b). The
same frames run through our U-Net give the detector's paired offset directly.

No leaderboard number is fitted here; it reads training masks and prints medians.

    label_domain_pa.py [--n 300] [--shape 800x1200] [--ckpt checkpoints/fasc_unet_pw6_best.pt]
"""
from __future__ import annotations
import sys, os, pathlib, argparse, random, warnings

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from PIL import Image
from umud import segment as S, geometry as Gm, grouping as G

FASC_IMG = ROOT / "data" / "raw" / "fasc_imgs_v1" / "fasc_images_new_model_v1"
FASC_MSK = ROOT / "data" / "raw" / "fasc_masks_v1" / "fasc_masks_new_model_v1"
EXT = {".tif", ".tiff", ".png", ".jpg"}
DUMMY_SCALE = 100.0        # PA is scale-free; any positive value
VAL_FRAC, VAL_MIN, SEED = 0.12, 40, 0   # must mirror train_fascicle.py


def _shape(p: pathlib.Path) -> tuple[int, int]:
    with Image.open(p) as im:
        return im.size[1], im.size[0]


def val_ids() -> set[str] | None:
    """Ids held out by train_fascicle.py (seed 0), so the detector column is honest."""
    try:
        from train_fascicle import pairs
        items = pairs()
    except Exception as e:                       # noqa: BLE001 - diagnostic only
        print(f"  [warn] cannot reproduce the training split ({e}); detector column unsplit")
        return None
    rng = random.Random(SEED); rng.shuffle(items)
    n_val = max(VAL_MIN, int(VAL_FRAC * len(items)))
    ids = set()
    for it in items[:n_val]:
        first = it[0] if isinstance(it, (tuple, list)) else (it.get("img") or it.get("image") if isinstance(it, dict) else it)
        ids.add(pathlib.Path(str(first)).stem)
    return ids


def select(shape: tuple[int, int], n: int, prefer_val: set[str] | None) -> list[str]:
    imgs = {p.stem: p for p in FASC_IMG.iterdir() if p.suffix.lower() in EXT}
    masks = {p.stem: p for p in FASC_MSK.iterdir() if p.suffix.lower() in EXT}
    stems = sorted(s for s in imgs if s in masks)          # any extension pairing
    if not stems:
        raise SystemExit(f"no image/mask pairs under {FASC_IMG} and {FASC_MSK}")
    stems = [s for s in stems if _shape(imgs[s]) == shape]
    if not stems:
        raise SystemExit(f"no training fascicle images of shape {shape}")
    if prefer_val:
        v = [s for s in stems if s in prefer_val]
        rest = [s for s in stems if s not in prefer_val]
        rng = random.Random(SEED); rng.shuffle(rest)
        stems = v + rest
    else:
        rng = random.Random(SEED); rng.shuffle(stems)
    return stems[:n]


def _find(d: pathlib.Path, stem: str) -> pathlib.Path:
    for p in d.glob(stem + ".*"):
        if p.suffix.lower() in EXT:
            return p
    raise SystemExit(f"no image file for {stem} under {d}")


def _read_mask(p: pathlib.Path) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"cannot read mask {p}")
    return (m > 127).astype(np.uint8)


def run(stems: list[str], ckpt: pathlib.Path | None, thr: float, val: set[str] | None,
        batch: int = 16) -> pd.DataFrame:
    model = dev = None
    if ckpt is not None:
        from eval_new_fascicle import load_model, predict_ours
        model, dev = load_model("unet", ckpt)
    rows = []
    for i in range(0, len(stems), batch):
        chunk = stems[i:i + batch]
        paths = [_find(FASC_IMG, s) for s in chunk]
        grays = [G.to_gray(cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) for p in paths]
        for s, g, (am, fm_dl) in zip(chunk, grays, S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)):
            gt = _read_mask(_find(FASC_MSK, s))
            mh, mw = gt.shape[:2]          # the mask's own grid: its aspect vs the image's is the export stretch
            if gt.shape != g.shape[:2]:
                gt = cv2.resize(gt, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
            r_gt = Gm.analyse(am, gt, DUMMY_SCALE, aspect=1.0)
            r_dl = Gm.analyse(am, fm_dl, DUMMY_SCALE, aspect=1.0)
            row = dict(image_id=s, h=g.shape[0], w=g.shape[1], mask_h=mh, mask_w=mw,
                       stretch=float((g.shape[1] / mw) / (g.shape[0] / mh)),
                       in_val=(s in val) if val else None,
                       pa_host=r_gt.pa_deg, n_host=r_gt.n_fascicles, note_host=r_gt.note,
                       pa_dltrack=r_dl.pa_deg,
                       mt_px=(r_gt.mt_mm * 10.0 if r_gt.mt_mm is not None else None))
            if model is not None:
                r_us = Gm.analyse(am, predict_ours(model, dev, g, thr=thr), DUMMY_SCALE, aspect=1.0)
                row.update(pa_ours=r_us.pa_deg, n_ours=r_us.n_fascicles)
            rows.append(row)
        print(f"  {min(i + batch, len(stems))}/{len(stems)}", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, probed: float = 16.42, test_median: float = 13.03) -> None:
    def med(c):
        v = df[c].dropna()
        return (len(v), float(v.median()), float(v.quantile(.25)), float(v.quantile(.75))) if len(v) else (0, float("nan"), float("nan"), float("nan"))
    if df.empty:
        print("\nno frames analysed")
        return
    print(f"\nframes {len(df)}  shape {df.h.iloc[0]}x{df.w.iloc[0]}  probed test median {probed}  our test median {test_median}")
    print(f"  host mask grids: {df.groupby(['mask_h', 'mask_w']).size().to_dict()}  "
          f"image/mask horizontal-over-vertical stretch median {df.stretch.median():.3f}")
    for c in [c for c in ("pa_host", "pa_ours", "pa_dltrack") if c in df]:
        n, m, q1, q3 = med(c)
        print(f"  {c:11s} n={n:4d}  median {m:6.2f}  IQR [{q1:.2f}, {q3:.2f}]")
    if "pa_ours" in df:
        both = df.dropna(subset=["pa_host", "pa_ours"])
        d = both.pa_ours - both.pa_host
        print(f"  paired ours-host: n={len(d)}  median {d.median():+.2f} deg  mean {d.mean():+.2f}  MAD {float((d - d.median()).abs().median()):.2f}")
        if both.in_val.notna().any():
            v = both[both.in_val == True]        # noqa: E712 - pandas mask
            if len(v):
                dv = v.pa_ours - v.pa_host
                print(f"  held-out only:    n={len(v)}  median {dv.median():+.2f} deg  host median {v.pa_host.median():.2f}  ours median {v.pa_ours.median():.2f}")
    n, m, *_ = med("pa_host")
    if n:
        verdict = ("frames-flat (a): host tracings on this frame type sit near our 13 deg, "
                   "so the 16.42 label median is not the delivered geometry -> aspect correction"
                   if abs(m - test_median) < abs(m - probed) else
                   "detector (b): host tracings sit near the probed 16.42 on the same frames, "
                   "so our U-Net draws fascicles too flat -> fix the detector, not the geometry")
        print(f"  verdict: {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--shape", default="800x1200", help="HxW of the training frames to use")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "fasc_unet_pw6_best.pt"))
    ap.add_argument("--no-detector", action="store_true", help="host masks only (no torch)")
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--out", default=str(ROOT / "reports" / "label_domain_pa.csv"))
    a = ap.parse_args()
    try:
        h, w = (int(x) for x in a.shape.lower().split("x"))
    except ValueError:
        return print(f"--shape must be HxW, got {a.shape!r}") or 2
    if a.n <= 0:
        return print("--n must be positive") or 2
    ckpt = None if a.no_detector else pathlib.Path(a.ckpt)
    if ckpt is not None and not ckpt.exists():
        return print(f"checkpoint not found: {ckpt}") or 2
    val = None if a.no_detector else val_ids()
    stems = select((h, w), a.n, val)
    print(f"{len(stems)} training frames of {h}x{w}"
          + (f" ({sum(s in val for s in stems)} held out of training)" if val else ""))
    df = run(stems, ckpt, a.thr, val)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summarise(df)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
