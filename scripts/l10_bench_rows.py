#!/usr/bin/env python3
"""L10 helper: per-image benchmark predictions for ONE fascicle checkpoint.

Runs the production geometry (DL_Track aponeurosis masks thr 0.35/0.10, our fascicle
U-Net at --thr, Gm.analyse with the benchmark's ground-truth px/cm) over the 35-image
expert benchmark and writes one row per image with pa_deg / fl_mm / mt_mm, so that the
L10 gate script can take the per-image MEDIAN across seeds and score it against the
consensus. eval_benchmark_ours.py prints only aggregates, which is why this exists.

Usage: l10_bench_rows.py --ckpt checkpoints/fasc_unet_pw6s1_best.pt --out reports/l10/bench_rows_pw6s1.csv [--thr 0.9]
"""
from __future__ import annotations
import sys, os, pathlib, warnings, argparse

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2  # noqa: E402
from eval_new_fascicle import load_model, predict_ours  # noqa: E402
from umud import validation as V, segment as S, geometry as Gm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--arch", default="unet", choices=["unet", "unetpp", "fpn", "segformer"],
                    help="smp architecture of --ckpt (default unet)")
    a = ap.parse_args()
    ckpt = pathlib.Path(a.ckpt)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    model, dev = load_model(a.arch, str(ckpt))
    t = V.load()
    grays = []
    for p in t.path:
        g = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if g is None:
            print(f"ERROR: unreadable benchmark image: {p}", file=sys.stderr)
            return 2
        grays.append(g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY))
    apo = [am for am, _ in S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)]
    rows = []
    for g, am, (_, r) in zip(grays, apo, t.iterrows()):
        fm = predict_ours(model, dev, g, thr=a.thr)
        res = Gm.analyse(am, fm, float(r.px_per_cm))
        rows.append(dict(image_id=r.image_id, pa_deg=res.pa_deg, fl_mm=res.fl_mm,
                         mt_mm=res.mt_mm, ok=bool(res.ok), note="" if res.ok else res.note))
    df = pd.DataFrame(rows)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    p = df.dropna(subset=["pa_deg", "fl_mm", "mt_mm"])
    print(f"ckpt={ckpt.name} thr={a.thr} usable {len(p)}/{len(df)} -> {out}")
    if len(p):
        print(f"UMUD = {V.score(p[['image_id','pa_deg','fl_mm','mt_mm']], t):.4f}")
        print(V.report(p[['image_id','pa_deg','fl_mm','mt_mm']], t).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
