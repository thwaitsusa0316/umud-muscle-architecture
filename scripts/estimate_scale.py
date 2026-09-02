#!/usr/bin/env python3
"""Estimate pixels-per-centimetre for every test image, and report agreement.

Combines three independent sources, strongest first:

  depth   sector height / OCR'd depth annotation. Robust and broadly available,
          but only as accurate as the sector boundary (~1-3 %).
  ruler   tick-comb pitch. Sub-pixel precise where it fires, but the tick *unit*
          (0.5 cm vs 1 cm) is ambiguous on its own.
  fused   the two together: use `depth` to decide the tick unit, then take the
          scale from the precise pitch. The unit only has to be resolved to
          within a factor of two, so a 25 % error in `depth` is still tolerable,
          and the result inherits the precision of the comb.

Group consensus then fills images where nothing fired, since frames from one
acquisition share a scale by construction.
"""
from __future__ import annotations
import sys, json, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, cv2
from umud import depth as D, ruler as R, grouping as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
UNITS = np.array([0.25, 0.5, 1.0, 2.0])          # plausible tick spacings, cm


def fuse(depth_pxcm: float | None, pitch_px: float | None) -> tuple[float | None, str]:
    if depth_pxcm and pitch_px:
        unit = UNITS[int(np.argmin(np.abs(UNITS - pitch_px / depth_pxcm)))]
        return pitch_px / unit, f"fused(unit={unit}cm)"
    if depth_pxcm:
        return depth_pxcm, "depth"
    return (None, "none")


def main() -> None:
    imgs, raw = {}, {}
    for p in sorted(TEST.iterdir(), key=lambda q: G._index(q.name)):
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            continue
        raw[p.name] = a
        imgs[p.name] = G.to_gray(a)

    ch = G.cluster_chrome({n: G.chrome_signature(a) for n, a in raw.items()}, thresh=0.82)
    runs = G.find_runs({n: G.frame_signature(a) for n, a in raw.items()}, ch)

    rows = {}
    for n, a in raw.items():
        d = D.estimate(a)
        r = R.find_ruler(imgs[n], "y") or R.find_ruler(imgs[n], "x")
        pitch = r.pitch_px if (r and r.confidence > 0.5) else None
        px, src = fuse(d.px_per_cm, pitch)
        rows[n] = {"depth_pxcm": d.px_per_cm, "depth_cm": d.depth_cm,
                   "sector": d.sector, "pitch_px": pitch, "px_per_cm": px,
                   "source": src, "chrome": ch[n], "run": runs[n]}

    direct = {n: v["px_per_cm"] for n, v in rows.items() if v["px_per_cm"]}
    print(f"direct estimate: {len(direct)}/{len(rows)}")

    filled = G.consensus(direct, ch)
    for n, v in rows.items():
        if not v["px_per_cm"] and n in filled:
            v["px_per_cm"], v["source"] = filled[n], "chrome-consensus"
    have = sum(1 for v in rows.values() if v["px_per_cm"])
    print(f"after chrome-cluster consensus: {have}/{len(rows)}")
    print("sources:", collections.Counter(v["source"].split("(")[0] for v in rows.values()).most_common())

    # cross-check: where depth and comb both fired, do they agree?
    both = [(v["depth_pxcm"], v["pitch_px"], v["px_per_cm"]) for v in rows.values()
            if v["depth_pxcm"] and v["pitch_px"]]
    if both:
        err = np.array([abs(f - d) / d for d, _p, f in both])
        print(f"\ndepth vs fused, n={len(both)}: median |diff| {100*np.median(err):.1f}%, "
              f"90th pct {100*np.percentile(err,90):.1f}%")

    # within-run agreement: frames of one take must share a scale
    byrun = collections.defaultdict(list)
    for n, v in rows.items():
        if v["px_per_cm"]:
            byrun[v["run"]].append(v["px_per_cm"])
    spreads = [(max(v) - min(v)) / np.median(v) for v in byrun.values() if len(v) >= 2]
    if spreads:
        s = np.array(spreads)
        print(f"within-run spread over {len(s)} multi-frame runs: "
              f"median {100*np.median(s):.2f}%, 90th pct {100*np.percentile(s,90):.2f}%, "
              f"within 2%: {int((s<0.02).sum())}/{len(s)}")

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "scale_final.json").write_text(
        json.dumps(rows, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\nwrote reports/scale_final.json")


if __name__ == "__main__":
    main()
