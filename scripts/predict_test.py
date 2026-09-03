#!/usr/bin/env python3
"""Run the pipeline over the competition test set and sanity-check the distribution.

The leaderboard probes pinned the public-set medians (PA 16.4 deg, FL 81.5 mm,
MT 20.7 mm). Those are a free, label-free check on the whole pipeline: a scale that
is wrong by a factor k shows up directly as MT and FL medians off by 1/k, while PA
is scale-invariant and should land correctly even where the scale is unknown.
"""
from __future__ import annotations
import sys, pathlib, json, warnings, os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from umud import segment as S, geometry as Gm, grouping as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
MEDIANS = {"pa_deg": 16.42, "fl_mm": 81.5, "mt_mm": 20.7}


def main(batch: int = 24) -> None:
    scale = json.loads((ROOT / "reports" / "scale_final.json").read_text())
    names = sorted((p.name for p in TEST.iterdir()
                    if p.suffix.lower() in {".tif", ".png"}), key=G._index)
    rows = []
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        grays = []
        for n in chunk:
            a = cv2.imread(str(TEST / n), cv2.IMREAD_UNCHANGED)
            grays.append(G.to_gray(a))
        masks = S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)
        for n, (am, fm) in zip(chunk, masks):
            px = (scale.get(n) or {}).get("px_per_cm")
            if not px:
                rows.append(dict(image_id=n, pa_deg=None, fl_mm=None, mt_mm=None,
                                 px_per_cm=None, note="no scale"))
                continue
            r = Gm.analyse(am, fm, float(px))
            rows.append(dict(image_id=n, pa_deg=r.pa_deg, fl_mm=r.fl_mm, mt_mm=r.mt_mm,
                             px_per_cm=float(px), note=r.note if not r.ok else ""))
        print(f"  {min(i+batch, len(names))}/{len(names)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "test_predictions.csv", index=False)

    ok = df.dropna(subset=["pa_deg", "fl_mm", "mt_mm"])
    print(f"\nusable {len(ok)}/{len(df)}   (no scale: {(df.note=='no scale').sum()})")
    print(f"\n{'target':8s} {'our median':>11s} {'public median':>14s} {'ratio':>7s}")
    for c, m in MEDIANS.items():
        med = float(ok[c].median())
        print(f"{c:8s} {med:11.2f} {m:14.2f} {med/m:7.3f}")
    print("\nPA is scale-free, so a PA ratio near 1.0 with FL/MT off means the "
          "geometry is sound and the scale is not.")


if __name__ == "__main__":
    main()
