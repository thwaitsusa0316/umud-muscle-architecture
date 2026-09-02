#!/usr/bin/env python3
"""Score a full image -> masks -> geometry pipeline on the expert benchmark.

This is the development loop: any change to segmentation or geometry gets an
honest number here, priced on the same scale as the human raters and DL_Track.
"""
from __future__ import annotations
import sys, pathlib, warnings, time, os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import numpy as np, pandas as pd, cv2
from umud import validation as V, segment as S, geometry as Gm

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(thr_apo: float = 0.35, thr_fasc: float = 0.10,
        verbose: bool = True) -> tuple[pd.DataFrame, float | None]:
    t = V.load()
    grays = []
    for p in t.path:
        a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        grays.append(a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
    t0 = time.time()
    masks = S.predict_batch(grays, thr_apo=thr_apo, thr_fasc=thr_fasc)
    if verbose:
        print(f"segmented {len(grays)} images in {time.time()-t0:.1f}s")
    rows = []
    for (a, f), (_, r) in zip(masks, t.iterrows()):
        res = Gm.analyse(a, f, float(r.px_per_cm))
        rows.append(dict(image_id=r.image_id, pa_deg=res.pa_deg, fl_mm=res.fl_mm,
                         mt_mm=res.mt_mm, n=res.n_fascicles, ok=res.ok, note=res.note))
    p = pd.DataFrame(rows)
    good = p.dropna(subset=["pa_deg", "fl_mm", "mt_mm"])
    s = V.score(good, t) if len(good) else None
    if verbose:
        bad = p[~p.ok]
        if len(bad):
            print("failures:", bad.note.value_counts().to_dict())
        print(f"usable {len(good)}/{len(p)}")
        if s is not None:
            print(f"\nUMUD = {s:.4f}")
            print(V.report(good, t).round(3).to_string(index=False))
        print("\nreference: DL_Track(published) 0.3306 | expert mean 0.3032 | "
              "expert best 0.2459 | best constant 0.7032")
    return p, s


if __name__ == "__main__":
    p, s = run()
    (ROOT / "reports").mkdir(exist_ok=True)
    p.to_csv(ROOT / "reports" / "bench_pipeline.csv", index=False)
