#!/usr/bin/env python3
"""Gate for stage 3: does a VLM pick the muscle-bounding aponeurosis pair better?

Judged label-free on the real test images by **in-muscle fascicle coverage** — the
fraction of detected fascicle pixels falling between the chosen pair. If the pair is
right, the fascicles are inside it; if it is wrong, they are not. No labels are
needed, it runs on the actual test distribution, and it is the statistic that exposed
the failure in the first place.

The benchmark cannot serve as this gate: only 3 of 12 of its images have more than two
candidate bands, so there is almost no choice to get wrong there. On the test set 43
of 96 are ambiguous.
"""
from __future__ import annotations
import sys, os, json, pathlib, argparse, warnings

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2
from umud import segment as S, bands as B, grouping as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "reports" / "band_gate.csv"


def coverage(fasc: np.ndarray, cands: list[dict], pair) -> float | None:
    if not pair or pair[0] is None or pair[1] is None:
        return None
    lo, hi = sorted((cands[pair[0]]["y"], cands[pair[1]]["y"]))
    tot = int((fasc > 0).sum())
    if tot == 0:
        return None
    return float((fasc[int(lo):int(hi)] > 0).sum()) / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--models", default="qwen3-vl:30b")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]

    scale = json.loads((ROOT / "reports" / "scale_final.json").read_text())
    names = sorted((p.name for p in TEST.iterdir()
                    if p.suffix.lower() in {".tif", ".png"}), key=G._index)
    rows = []
    for i in range(0, len(names), 24):
        if len(rows) >= a.limit:
            break
        chunk = names[i:i + 24]
        grays = [G.to_gray(cv2.imread(str(TEST / n), cv2.IMREAD_UNCHANGED)) for n in chunk]
        masks = S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)
        for n, g, (am, fm) in zip(chunk, grays, masks):
            if len(rows) >= a.limit:
                break
            px = (scale.get(n) or {}).get("px_per_cm")
            if not px:
                continue
            cands = B.candidates(am)
            if len(cands) <= 2:            # nothing to choose
                continue
            geo = B.geometric_choice(cands, float(px))
            choice, _ = B.choose(g, am, float(px), models=models)
            rows.append({
                "image_id": n, "n_bands": len(cands), "source": choice.source,
                "cov_geo": coverage(fm, cands, geo),
                "cov_vlm": coverage(fm, cands, (choice.superficial, choice.deep)),
                "same": bool(geo and (choice.superficial, choice.deep) == tuple(sorted(geo))),
            })
            print(f"  {n} bands={len(cands)} src={choice.source} "
                  f"geo={rows[-1]['cov_geo']} vlm={rows[-1]['cov_vlm']}", flush=True)

    d = pd.DataFrame(rows).dropna(subset=["cov_geo", "cov_vlm"])
    d.to_csv(OUT, index=False)
    if d.empty:
        print("no comparable images"); return
    print(f"\nambiguous test images compared: {len(d)}   models: {models}")
    print(f"  VLM agreed with the geometric rule on {int(d.same.sum())}/{len(d)}")
    print(f"\n  in-muscle coverage   geometric {d.cov_geo.median():.3f}   "
          f"VLM {d.cov_vlm.median():.3f}   (median)")
    print(f"                       geometric {d.cov_geo.mean():.3f}   "
          f"VLM {d.cov_vlm.mean():.3f}   (mean)")
    better = int((d.cov_vlm > d.cov_geo + 1e-9).sum())
    worse = int((d.cov_vlm < d.cov_geo - 1e-9).sum())
    print(f"\n  VLM better on {better}, worse on {worse}, equal on {len(d)-better-worse}")
    print(f"\nGATE: {'PASS' if d.cov_vlm.median() > d.cov_geo.median() else 'FAIL'} "
          "(VLM must raise median in-muscle coverage)")


if __name__ == "__main__":
    main()
