#!/usr/bin/env python3
"""L6t label-free evidence (PLAN v18 rung 1): does the pw6 fascicle detector under-fire on
test-domain appearance?

Runs the DL_Track aponeurosis model and our pw6 fascicle U-Net on the 309 test frames and on
N training frames (800x1200, the split used by label_domain_pa.py), and writes per-image,
label-free statistics of the fascicle mask: mask fraction over the whole frame and inside the
aponeurosis band, connected-component count, orientation median/MAD (signed deg from horizontal, direction folded to vx >= 0 so the value lies in (-90, 90]),
mean component length (px), plus the pipeline's own PA and fascicle count from Gm.analyse.
Families are label-free too: method (still/video from the scale manifest) x extension x
frame shape. Pre-registered read (PLAN v18): if the test-frame band mask fraction OR the
component count is below 0.5x the training median on a family, the detector under-fires on
that family's appearance (self-training is the right fix); if comparable, the fault is
downstream (pairing / FL extrapolation).
"""
from __future__ import annotations
import sys, os, pathlib, argparse, json, warnings

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from umud import segment as S, geometry as Gm, grouping as G

TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
DUMMY_SCALE = 100.0
MIN_AREA = 30          # px; smaller components are speckle
STATS = ["frac_all", "frac_band", "n_comp", "ang_med", "ang_mad", "len_mean_px", "pa_ours", "n_ours"]


def mask_stats(fm: np.ndarray, am: np.ndarray) -> dict:
    """Label-free statistics of one fascicle mask, using the apo mask only for the band."""
    h, w = fm.shape[:2]
    rows = np.flatnonzero(am.any(axis=1)) if am is not None and am.any() else None
    if rows is not None and rows[-1] - rows[0] >= 10:
        lo, hi, band_ok = int(rows[0]), int(rows[-1]), True
    else:
        lo, hi, band_ok = 0, h - 1, False
    band = fm[lo:hi + 1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(fm.astype(np.uint8), connectivity=8)
    angs, lens = [], []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        ys, xs = np.nonzero(lab == k)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        vx, vy = (float(v) for v in np.asarray(cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)).ravel()[:2])
        if vx < 0 or (vx == 0 and vy < 0):      # fitLine's direction sign is arbitrary: fold to vx >= 0
            vx, vy = -vx, -vy                    # so parallel lines share one signed angle in (-90, 90]
        angs.append(float(np.degrees(np.arctan2(vy, vx))))
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        lens.append(float(max(a, b)))
    angs = np.array(angs); lens = np.array(lens)
    ang_med = float(np.median(angs)) if len(angs) else None
    return dict(h=h, w=w, band_ok=band_ok, band_rows=hi - lo + 1,
                frac_all=float(fm.mean()), frac_band=float(band.mean()),
                n_comp=int(len(angs)), ang_med=ang_med,
                ang_mad=float(np.median(np.abs(angs - ang_med))) if len(angs) else None,
                len_mean_px=float(lens.mean()) if len(lens) else None)


def family(name: str, h: int, w: int, manifest: dict) -> str:
    m = manifest.get(name, {}).get("method", "train")
    return f"{m}:{pathlib.Path(name).suffix.lower().lstrip('.')}:{h}x{w}"


def test_items(limit: int | None) -> list[tuple[str, pathlib.Path]]:
    names = sorted((p.name for p in TEST.iterdir() if p.suffix.lower() in {".tif", ".png"}),
                   key=G._index)
    return [(n, TEST / n) for n in names[:limit]]


def train_items(n: int, limit: int | None) -> list[tuple[str, pathlib.Path]]:
    from label_domain_pa import select, val_ids
    items = select((800, 1200), n, val_ids())
    return [(s + ip.suffix.lower(), ip) for s, ip, _ in items[:limit]]


def run(items: list[tuple[str, pathlib.Path]], split: str, model, dev, thr: float,
        manifest: dict, batch: int = 16) -> list[dict]:
    from eval_new_fascicle import predict_ours
    rows = []
    for i in range(0, len(items), batch):
        chunk, grays = [], []
        for n, p in items[i:i + batch]:
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.size == 0:
                print(f"  [skip] unreadable image {p}")
                continue
            chunk.append(n); grays.append(G.to_gray(raw))
        if not grays:
            continue
        for n, g, (am, _fm_dl) in zip(chunk, grays, S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)):
            fm = predict_ours(model, dev, g, thr=thr)
            r = Gm.analyse(am, fm, DUMMY_SCALE, aspect=1.0)
            row = dict(image_id=n, split=split, family=family(n, g.shape[0], g.shape[1], manifest))
            row.update(mask_stats(fm, am))
            row.update(pa_ours=r.pa_deg, n_ours=r.n_fascicles, note=r.note)
            rows.append(row)
        print(f"  {split} {min(i + batch, len(items))}/{len(items)}", flush=True)
    return rows


def summarise(df: pd.DataFrame, ratio_thr: float = 0.5) -> pd.DataFrame:
    tr = df[df.split == "train"]
    ref = tr[STATS].median(numeric_only=True)
    out = []
    for fam, d in df[df.split == "test"].groupby("family"):
        med = d[STATS].median(numeric_only=True)
        rf = med["frac_band"] / ref["frac_band"] if ref["frac_band"] else np.nan
        rn = med["n_comp"] / ref["n_comp"] if ref["n_comp"] else np.nan
        out.append(dict(family=fam, n=len(d), **{k: med[k] for k in STATS},
                        ratio_frac_band=rf, ratio_n_comp=rn,
                        under_fires=bool((rf < ratio_thr) or (rn < ratio_thr))))
    out.append(dict(family="TRAIN (reference)", n=len(tr), **{k: ref[k] for k in STATS},
                    ratio_frac_band=1.0, ratio_n_comp=1.0, under_fires=False))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="frames per split (smoke test)")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "fasc_unet_pw6_best.pt"))
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--scale", default=str(ROOT / "reports" / "scale_v3d2.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "l6t_fascicle_evidence.csv"))
    a = ap.parse_args()
    manifest = json.loads(pathlib.Path(a.scale).read_text())
    if not all("method" in v for v in manifest.values()):
        raise SystemExit("scale manifest lacks a method field; use reports/scale_v3d2.json")
    from eval_new_fascicle import load_model
    model, dev = load_model("unet", pathlib.Path(a.ckpt))
    te, tr = test_items(a.limit), train_items(a.n_train, a.limit)
    print(f"test frames {len(te)}, train frames {len(tr)}, thr {a.thr}, device {dev}")
    rows = run(te, "test", model, dev, a.thr, manifest) + run(tr, "train", model, dev, a.thr, manifest)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    summ = summarise(df)
    sp = pathlib.Path(a.out).with_name(pathlib.Path(a.out).stem + "_summary.csv")
    summ.to_csv(sp, index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print("\nper-family medians (test) vs training reference:")
    print(summ.round(3).to_string(index=False))
    uf = summ[(summ.under_fires) & (summ.family != "TRAIN (reference)")]
    print(f"\nverdict: {'UNDER-FIRES on ' + ', '.join(uf.family) if len(uf) else 'no family under-fires (<0.5x) -> fault is downstream of the detector'}")
    print(f"wrote {a.out} and {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
