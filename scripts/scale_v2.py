#!/usr/bin/env python3
"""L6: px/cm for every test image (PLAN v3 rung 1), replacing reports/scale_final.json.

scale_final.json left 176/309 images without a scale and mis-resolved the tick unit on
the 3.0 cm Siemens stills. Three mechanistic fixes, no leaderboard tuning:

  1. re-fuse    the stored depth estimate and ruler pitch are re-fused with 0.2 cm and
                0.1 cm added to the admissible tick units (Siemens draws 2 mm ticks at
                3.0 cm depth; the old set {0.25, 0.5, 1, 2} forced 0.25 -> 20 % low).
  2. footprint  Siemens Juniper stills (chrome sector at y ~ 87) show a constant 5.75 cm
                lateral field at every depth seen (3.0, 3.5, 4.0, 4.5, 6.0, 6.5 cm), so
                px/cm = sector_w / 5.75 fills stills with no readable depth or ruler and
                cross-checks the rest.
  3. left ruler video-mode frames (sector detection returned the full frame) carry a
                left ruler labelled 0 .. D mm with major ticks; px/cm = tick span / D.
                D comes from OCR of the bottom label ("50") and, on 1200-wide frames,
                the "De 50 mm" panel; failing both, the run / chrome-cluster consensus;
                failing that, 1 cm per major tick is assumed and flagged.

Writes reports/scale_v2.json (same schema as scale_final.json plus `method`, `flags`)
and prints coverage, source counts and the consistency checks.
"""
from __future__ import annotations
import sys, re, json, pathlib, argparse, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, cv2
from umud import grouping as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
UNITS = np.array([0.1, 0.2, 0.25, 0.5, 1.0, 2.0])   # admissible tick spacings, cm
FOOTPRINT_CM = 5.75                                  # Siemens Juniper linear field width
UNIT_TOL = 0.12                                      # pitch/depth must sit this close to a unit


def refuse(depth_pxcm, pitch_px):
    """Combine depth-derived px/cm with the ruler pitch; the unit must be admissible."""
    if depth_pxcm and pitch_px:
        ratio = pitch_px / depth_pxcm
        unit = float(UNITS[int(np.argmin(np.abs(UNITS - ratio)))])
        if abs(ratio - unit) / unit <= UNIT_TOL:
            return pitch_px / unit, f"fused(unit={unit}cm)"
        return depth_pxcm, "depth(unit-ambiguous)"
    if depth_pxcm:
        return depth_pxcm, "depth"
    return None, "none"


def siemens_still(sector, w):
    return bool(sector) and w == 1200 and 84 <= sector[1] <= 92 and sector[0] >= 100


def video_layout(sector, w):
    return (not sector) or sector[0] == 0


def _ocr_digits(region: np.ndarray, psm: int = 7) -> str:
    if region.size == 0:
        return ""
    r = cv2.resize(region, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    r = 255 - r
    r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    try:
        import pytesseract
        return pytesseract.image_to_string(
            r, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789Demm ")
    except Exception:
        return ""


def left_ruler(gray: np.ndarray, debug: bool = False) -> dict:
    """Major-tick comb of the left depth ruler on a video-mode frame."""
    h, w = gray.shape
    strip = gray[:, : int(0.15 * w)].astype(np.int32)
    bg = int(np.median(strip))
    ink = (strip > bg + 25).astype(np.uint8)
    # drop solid vertical lines (ruler spine, tissue edge) before looking for ticks
    col_fill = ink.mean(axis=0)
    ink[:, col_fill > 0.5] = 0
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(ink, connectivity=8)
    ticks = []
    for i in range(1, n):
        x, y, cw, ch, _a = stats[i]
        if ch <= 3 and 4 <= cw <= 24:
            ticks.append((y + ch / 2.0, x + cw))      # (row, right edge)
    if len(ticks) < 3:
        return {"ok": False, "why": "no tick components"}
    # ticks of one ruler share a right edge; take the most populated edge
    edges = collections.Counter(int(round(e / 2.0)) for _r, e in ticks)
    edge = edges.most_common(1)[0][0] * 2
    rows = sorted(r for r, e in ticks if abs(e - edge) <= 3)
    if len(rows) < 3:
        return {"ok": False, "why": "no aligned ticks"}
    d = np.diff(rows)
    pitch = float(np.median(d))
    if pitch < 20 or (np.abs(d - pitch) > 0.08 * pitch).any():
        # minor ticks or clutter: keep the dominant spacing only
        keep = [rows[0]] + [rows[i + 1] for i in range(len(d)) if abs(d[i] - pitch) <= 0.08 * pitch]
        rows = sorted(set(keep))
        d = np.diff(rows)
        if len(rows) < 3 or (np.abs(d - pitch) > 0.08 * pitch).any():
            return {"ok": False, "why": f"irregular comb {np.round(d, 1).tolist()}"}
    span = rows[-1] - rows[0]
    n_int = int(round(span / pitch))
    # depth label sits just below the last tick, right of the tick edge
    y1 = int(rows[-1]); x0 = edge + 2
    lab = gray[max(y1 - 6, 0): min(y1 + 26, h), x0: min(x0 + 60, w)]
    txt = _ocr_digits(lab)
    m = re.findall(r"\d+", txt)
    depth_mm = None
    for t in m:
        v = int(t)
        if 20 <= v <= 120:
            depth_mm = v
    out = dict(ok=True, pitch_px=pitch, n_intervals=n_int, first=float(rows[0]),
               last=float(rows[-1]), edge=int(edge), label_mm=depth_mm, label_txt=txt.strip())
    if w == 1200:                                   # parameter panel "De 50 mm"
        pan = gray[30: 160, 1070: 1195]
        t2 = _ocr_digits(pan, psm=6)
        mm = re.search(r"De\s*(\d{2,3})\s*m", t2)
        out["panel_mm"] = int(mm.group(1)) if mm else None
        out["panel_txt"] = t2.strip().replace("\n", "|")[:60]
    if debug:
        print("   ", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="image ids to process (debug; no scale_v2.json written)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    src_json = ROOT / "reports" / "scale_final.json"
    try:
        base = json.loads(src_json.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot load {src_json}: {e}")
    names = [n for n in base if not a.only or n in a.only]
    if a.only and len(names) != len(a.only):
        raise SystemExit(f"unknown image ids: {sorted(set(a.only) - set(names))}")
    rows = {}
    for n in names:
        b = base[n]
        img = cv2.imread(str(TEST / n), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"cannot read {TEST / n}")
        g = G.to_gray(img)
        h, w = g.shape
        sector = b.get("sector")
        r = dict(b)
        r["flags"] = []
        r["px_fp"] = sector[2] / FOOTPRINT_CM if siemens_still(sector, w) else None
        px, src = refuse(b.get("depth_pxcm"), b.get("pitch_px"))
        if src.startswith("fused") and b["source"] != src:
            r["flags"].append(f"unit-refused:{b['source']}->{src}")
        method = "still"
        if video_layout(sector, w):
            method = "video"
            lr = left_ruler(g, a.debug)
            r["ruler"] = lr
            if lr.get("ok"):
                mm = lr.get("panel_mm") or lr.get("label_mm")
                if lr.get("panel_mm") and lr.get("label_mm") and lr["panel_mm"] != lr["label_mm"]:
                    r["flags"].append(f"label/panel disagree {lr['label_mm']}/{lr['panel_mm']}")
                    mm = lr["panel_mm"]
                # the OCR'd depth must reconcile with the comb: depth / intervals has
                # to be an admissible tick unit, else the label was misread
                unit = (mm / 10.0) / lr["n_intervals"] if mm and lr["n_intervals"] > 0 else None
                near = float(UNITS[int(np.argmin(np.abs(UNITS - unit)))]) if unit else None
                if unit and abs(unit - near) / near <= UNIT_TOL:
                    px, src = (lr["last"] - lr["first"]) / (mm / 10.0), "left-ruler"
                    r["depth_cm"] = mm / 10.0
                else:
                    # keep whatever refuse() measured; only the label is unusable
                    if mm:
                        r["flags"].append(f"label {mm} mm inconsistent with {lr['n_intervals']} intervals")
                    if px is None:
                        src = "ruler-no-label"
            elif px is None:
                src = "none"
        elif px is None and r["px_fp"]:
            px, src = r["px_fp"], "footprint"
        if r["px_fp"] and px and abs(px - r["px_fp"]) / r["px_fp"] > 0.04 and src != "footprint":
            r["flags"].append(f"footprint-mismatch {px:.1f} vs {r['px_fp']:.1f}")
        r.update(px_per_cm=px, source=src, method=method)
        rows[n] = r
        if a.debug:
            print(n, (h, w), method, src, None if px is None else round(px, 2), r["flags"])

    # consensus fills: run first (frames of one take share a scale), then the ruler
    # pitch within a chrome cluster (same layout, same comb -> same depth setting)
    chrome = {n: rows[n]["chrome"] for n in rows}
    runs = {n: rows[n]["run"] for n in rows}
    # seeds are measured values only, fixed before either pass, so a consensus fill
    # is never re-propagated as if it were a measurement
    direct = {n: v["px_per_cm"] for n, v in rows.items()
              if v["px_per_cm"] and "consensus" not in v["source"] and v["source"] != "assumed-unit"}
    for level, grp in (("run", runs), ("chrome", chrome)):
        filled = G.consensus(direct, grp)
        for n, v in rows.items():
            if not v["px_per_cm"] and n in filled:
                v["px_per_cm"], v["source"] = filled[n], f"{level}-consensus"
    for n, v in rows.items():
        lr = v.get("ruler") or {}
        # only a comb that starts at the top of the frame (the 0 mark) and spans
        # >= 4 intervals is trusted as a whole ruler; a partial comb is not
        if (not v["px_per_cm"] and lr.get("ok") and 4 <= lr["n_intervals"] <= 8
                and lr["first"] <= 0.08 * (v.get("sector") or [0, 0, 0, 0])[3] + 60):
            v["px_per_cm"], v["source"] = lr["pitch_px"], "assumed-unit"
            v["flags"].append("assumed 1 cm per major tick")

    have = sum(1 for v in rows.values() if v["px_per_cm"])
    print(f"scaled {have}/{len(rows)}   (scale_final had "
          f"{sum(1 for n in rows if base[n]['px_per_cm'])})")
    print("sources:", collections.Counter(v["source"].split("(")[0] for v in rows.values()).most_common())
    fl = collections.Counter(f.split(" ")[0].split(":")[0] for v in rows.values() for f in v["flags"])
    print("flags:", fl.most_common())
    fp = [(v["px_per_cm"], v["px_fp"]) for v in rows.values() if v["px_fp"] and v["px_per_cm"] and v["source"] != "footprint"]
    if fp:
        e = np.array([abs(p - f) / f for p, f in fp])
        print(f"footprint check on {len(fp)} stills: median |diff| {100*np.median(e):.1f}%, "
              f"90th {100*np.percentile(e, 90):.1f}%, >4%: {int((e > 0.04).sum())}")
    byrun = collections.defaultdict(list)
    for v in rows.values():
        if v["px_per_cm"]:
            byrun[v["run"]].append(v["px_per_cm"])
    sp = np.array([(max(x) - min(x)) / np.median(x) for x in byrun.values() if len(x) >= 2])
    if sp.size:
        print(f"within-run spread over {sp.size} runs: median {100*np.median(sp):.2f}%, "
              f"90th {100*np.percentile(sp, 90):.2f}%, within 2%: {int((sp < 0.02).sum())}/{sp.size}")
    vals = collections.Counter(round(v["px_per_cm"], 1) for v in rows.values() if v["px_per_cm"])
    print("px/cm values:", sorted(vals.items(), key=lambda x: -x[1])[:14])
    changed = [(n, round(base[n]["px_per_cm"], 1), round(v["px_per_cm"], 1)) for n, v in rows.items()
               if base[n]["px_per_cm"] and v["px_per_cm"] and abs(v["px_per_cm"] - base[n]["px_per_cm"]) / base[n]["px_per_cm"] > 0.03]
    print(f"changed >3% vs scale_final: {len(changed)}", changed[:8])
    if not a.only:
        out = ROOT / "reports" / "scale_v2.json"
        out.write_text(json.dumps(rows, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print("wrote", out)


if __name__ == "__main__":
    main()
