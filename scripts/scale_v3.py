#!/usr/bin/env python3
"""L6f (PLAN v8 rung 1): two mechanistic scale fixes on top of reports/scale_v2.json.

  A  chrome-71/72 PNG stills (IMG_00252-00309, 51 rows). The sector WIDTH is quantized
     by the depth setting (499 px @ 3.5 cm, 437 @ 4.0, 388 @ 4.5, 350 @ 5.0, 291 @ 6.0)
     and the fused ruler reads a constant 172.4 px/cm across sector heights 530-604 on
     the 499-wide frames: the sector is bottom-CROPPED per image, not stretched. The
     31 rows scaled from height / depth_cm are therefore low (ratio 1.03-1.61).
     Rule: px/cm = sector_w / 2.894 cm (footprint), 3.18 cm on the 582-px zoom rows.
     Verifies assumption A15.

  B  native 644x1088 video frames (50 rows). scale_v2 gave them the 800x1200 chrome
     consensus 148.1 or nothing. Their own left ruler runs 0..50 mm with 1 cm major
     ticks over the full frame height (label 0 at the top edge, 50 at ~630 px; read
     on IMG_00056), and the comb already measured a regular 126.0 px pitch on 46 of
     them but scale_v2 refused it (< 4 intervals). Rule: px/cm = comb pitch when the
     comb is regular, 5 x pitch fits the frame height and is >= 90 % of it (a 50 mm
     ruler spanning the frame), and the pitch sits within 4 % of the family median;
     the rest take the family median (native-consensus). Verifies assumption A16
     (which predicted 119.2; the ruler says 126.0).

    scale_v3.py --variant a   -> reports/scale_v3a.json   (fix A only)
    scale_v3.py --variant b   -> reports/scale_v3b.json   (A + B)

Each variant is one leaderboard variable (v3a = A15 falsifier, v3b = A16 falsifier).
Every other row is copied from scale_v2.json unchanged; `px_v2` keeps the old value.
"""
from __future__ import annotations
import sys, json, pathlib, argparse, collections
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
SRC = ROOT / "reports" / "scale_v2.json"
CHROME_71 = {71, 72}
FOOTPRINT_71 = 2.894          # cm, lateral field of the chrome-71 device at 3.5-6.0 cm depth
FOOTPRINT_71_ZOOM = 3.18      # cm, the 582-px rows (3.3 cm depth zoom)
ZOOM_MIN_W = 540              # sector width above which the zoom footprint applies
NATIVE_H, NATIVE_W = 644, 1088   # native video grid: 644 ROWS tall x 1088 COLUMNS wide (PIL size = (1088, 644))
N_NATIVE = 50                    # how many test frames are on that grid (inventory census)
PITCH_TOL = 0.04              # a native comb pitch must sit this close to the family median
N_TEST = 309


def image_hw(name: str) -> tuple[int, int]:
    """(rows, columns) of a test image without decoding the pixels. PIL's size is (w, h)."""
    from PIL import Image
    with Image.open(TEST / name) as im:
        w, h = im.size
    return h, w


def fix_chrome71(v: dict) -> dict | None:
    """Fix A. Returns the updated row or None when the rule does not apply."""
    if v.get("chrome") not in CHROME_71 or v.get("method") != "still":
        return None
    sector = v.get("sector")
    if not sector or len(sector) < 3 or not (0 < sector[2] < 1200):
        return None
    sw = float(sector[2])
    fp = FOOTPRINT_71_ZOOM if sw >= ZOOM_MIN_W else FOOTPRINT_71
    px = sw / fp
    r = dict(v)
    r["px_v2"] = v.get("px_per_cm")
    r["px_per_cm"] = px
    r["source"] = f"chrome71-footprint({fp}cm)"
    r["flags"] = list(v.get("flags") or []) + [f"L6f-A: {v.get('source')} -> sector_w/{fp}"]
    return r


def native_pitch(v: dict, h: int) -> float | None:
    """A trustworthy comb pitch on a native frame: regular, >= 2 intervals, 50 mm fits.

    h is the frame height in ROWS (644 on the native grid). The left ruler runs 0..50 mm
    top to bottom with 1 cm major ticks, so 5 x pitch (~630 px at 126 px/cm) must fit
    inside the 644 rows and span at least 90 % of them (>= 580 px)."""
    lr = v.get("ruler") or {}
    if not lr.get("ok"):
        return None
    pitch = lr.get("pitch_px")
    n_int = lr.get("n_intervals") or 0
    if not pitch or pitch <= 0 or n_int < 2:
        return None
    span5 = 5.0 * float(pitch)
    if not (0.90 * h <= span5 <= h):   # 0..50 mm ruler spans the frame, 1 cm per tick
        return None
    return float(pitch)


def fix_native(rows: dict, hw: dict) -> tuple[dict, dict]:
    """Fix B on the native-grid video frames. Returns (updated rows, summary)."""
    names = [n for n, v in rows.items()
             if hw[n] == (NATIVE_H, NATIVE_W) and v.get("method") == "video"]   # hw = (rows, cols)
    if len(names) != N_NATIVE:
        raise SystemExit(f"scale_v3: expected {N_NATIVE} native {NATIVE_H}x{NATIVE_W} frames, matched {len(names)}")
    pitches = {n: native_pitch(rows[n], hw[n][0]) for n in names}
    direct = {n: p for n, p in pitches.items() if p}
    if not direct:
        raise SystemExit("scale_v3: no trustworthy comb on any native frame; fix B cannot run")
    fam = float(np.median(list(direct.values())))
    n_direct = n_cons = 0
    for n in names:
        v = dict(rows[n])
        p = pitches[n]
        v["px_v2"] = rows[n].get("px_per_cm")
        if p and abs(p - fam) / fam <= PITCH_TOL:
            v["px_per_cm"], v["source"] = p, "native-ruler"
            n_direct += 1
        else:
            v["px_per_cm"], v["source"] = fam, "native-consensus"
            n_cons += 1
        v["flags"] = list(rows[n].get("flags") or []) + [f"L6f-B: {rows[n].get('source')} -> {v['source']}"]
        rows[n] = v
    return rows, dict(n=len(names), family_median=fam, direct=n_direct, consensus=n_cons,
                      pitch_min=min(direct.values()), pitch_max=max(direct.values()))


def validate(rows: dict) -> None:
    if len(rows) != N_TEST:
        raise SystemExit(f"scale_v3: expected {N_TEST} rows, got {len(rows)}")
    bad = [n for n, v in rows.items()
           if v.get("px_per_cm") is not None and not (0 < float(v["px_per_cm"]) < 1000)]
    if bad:
        raise SystemExit(f"scale_v3: implausible px_per_cm on {bad[:3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["a", "b"], required=True,
                    help="a = chrome-71 footprint only; b = a + native 644x1088 ruler")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default="", help="default reports/scale_v3<variant>.json")
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    a = ap.parse_args()

    try:
        base = json.loads(pathlib.Path(a.src).read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"scale_v3: cannot load {a.src}: {e}")
    if not isinstance(base, dict):
        raise SystemExit(f"scale_v3: {a.src} is not a dict of rows")
    validate(base)
    rows = {n: dict(v) for n, v in base.items()}

    # fix A
    changed_a = []
    for n in list(rows):
        r = fix_chrome71(rows[n])
        if r is not None:
            rows[n] = r
            old = r["px_v2"]
            changed_a.append((n, old, r["px_per_cm"], (r["px_per_cm"] / old) if old else None))
    ratios = np.array([c[3] for c in changed_a if c[3]])
    moved = int((np.abs(ratios - 1) > 0.02).sum()) if ratios.size else 0
    print(f"fix A chrome-71/72: {len(changed_a)} rows re-scaled by footprint, {moved} moved > 2 %; "
          f"ratio new/old median {np.median(ratios):.3f} range "
          f"[{ratios.min():.3f}, {ratios.max():.3f}]" if ratios.size else "fix A: no rows matched")
    if not changed_a:
        raise SystemExit("scale_v3: fix A matched no chrome-71/72 rows; wrong source file?")

    # fix B
    summary_b = None
    if a.variant == "b":
        try:
            hw = {n: image_hw(n) for n in rows}
        except (OSError, ValueError) as e:
            raise SystemExit(f"scale_v3: cannot read a test image header: {e}")
        rows, summary_b = fix_native(rows, hw)
        print(f"fix B native {NATIVE_H}x{NATIVE_W} (rows x cols): {summary_b['n']} rows, direct comb {summary_b['direct']}, "
              f"consensus {summary_b['consensus']}, family median {summary_b['family_median']:.1f} px/cm "
              f"(pitch {summary_b['pitch_min']:.1f}-{summary_b['pitch_max']:.1f})")

    validate(rows)
    have_old = sum(1 for v in base.values() if v.get("px_per_cm"))
    have_new = sum(1 for v in rows.values() if v.get("px_per_cm"))
    print(f"scaled {have_new}/{len(rows)} (scale_v2 had {have_old})")
    print("sources:", collections.Counter(str(v.get("source", "")).split("(")[0] for v in rows.values()).most_common())
    untouched = [n for n in rows if rows[n].get("px_per_cm") != base[n].get("px_per_cm")
                 and "px_v2" not in rows[n]]
    if untouched:
        raise SystemExit(f"scale_v3: rows changed outside the two rules: {untouched[:3]}")

    if a.dry_run:
        return 0
    out = pathlib.Path(a.out) if a.out else ROOT / "reports" / f"scale_v3{a.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    back = json.loads(out.read_text())
    validate(back)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
