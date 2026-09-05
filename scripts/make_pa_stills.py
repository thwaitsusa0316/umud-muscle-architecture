#!/usr/bin/env python3
"""PA isolation routed by frame type (PLAN v6 rung L6c-pa).

Label-free finding 2026-09-05 (reports/pipeline_a1_scalev2_rows.csv x reports/scale_v2.json):
measured PA on the 147 800x1200 STILLS has median 15.91 deg, on the probed 16.42; the
135 VIDEO frames (85 at 800x1200, 50 on the native 644x1088 grid) have median 11.3 deg.
The -21 % "PA bias" is therefore the video frames, not an aspect stretch (a native grid
shows the same 11.3). This script ships measured PA on stills only and the probed
median on video frames, with FL and MT at the probed medians, so one submission tells
whether still-frame PA beats the constant (falsifier in PLAN.md).

    make_pa_stills.py --source outputs/sub_pipeline_a1_scalev2.csv --out outputs/sub_L6c_isolate_pa_stills.csv

Writes the CSV only; submission is pilot's job (submit-gate checks the test record).
"""
from __future__ import annotations
import sys, json, pathlib, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIANS = ROOT / "reports" / "public_medians.json"
SCALE = ROOT / "reports" / "scale_v2.json"
NEED = ["image_id", "pa_deg", "fl_mm", "mt_mm"]


def load_medians(path: pathlib.Path) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
        vals = {"pa_deg": float(m["median_pa_deg"]), "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_pa_stills: medians file {path} unreadable or malformed: {e}")
    if not (vals["pa_deg"] >= 0 and vals["fl_mm"] > 0 and vals["mt_mm"] > 0):
        sys.exit(f"make_pa_stills: implausible medians {vals}")
    return vals


def load_methods(path: pathlib.Path) -> dict[str, str]:
    try:
        sc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"make_pa_stills: cannot read scale lookup {path}: {e}")
    if not isinstance(sc, dict) or not sc:
        sys.exit(f"make_pa_stills: scale lookup {path} is not a non-empty dict")
    meth = {}
    for k, v in sc.items():
        m = (v or {}).get("method") if isinstance(v, dict) else None
        if m not in ("still", "video"):
            sys.exit(f"make_pa_stills: {k} has method {m!r}, expected 'still' or 'video'")
        meth[k] = m
    return meth


def build(source: pd.DataFrame, meth: dict[str, str], vals: dict[str, float]) -> tuple[pd.DataFrame, int]:
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    ids = submit.test_ids()
    src = source.set_index("image_id")
    missing = [i for i in ids if i not in src.index]
    if missing:
        raise ValueError(f"source lacks {len(missing)} test ids, e.g. {missing[:3]}")
    unknown = [i for i in ids if i not in meth]
    if unknown:
        raise ValueError(f"scale lookup lacks {len(unknown)} test ids, e.g. {unknown[:3]}")
    pa = src.loc[ids, "pa_deg"].astype(float).to_numpy()
    if pd.isna(pa).any():
        raise ValueError("NaN in source pa_deg")
    still = pd.Series([meth[i] == "still" for i in ids], dtype=bool).to_numpy()
    out = pd.DataFrame({"image_id": ids})
    out["pa_deg"] = [float(p) if s else vals["pa_deg"] for p, s in zip(pa, still)]
    out["fl_mm"] = vals["fl_mm"]
    out["mt_mm"] = vals["mt_mm"]
    return out, int(still.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "outputs" / "sub_pipeline_a1_scalev2.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sub_L6c_isolate_pa_stills.csv"))
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--scale", default=str(SCALE))
    a = ap.parse_args()
    vals = load_medians(pathlib.Path(a.medians))
    meth = load_methods(pathlib.Path(a.scale))
    try:
        source = pd.read_csv(a.source)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_pa_stills: cannot read source CSV {a.source}: {e}")
    if source.empty:
        sys.exit(f"make_pa_stills: source CSV {a.source} is empty")
    try:
        df, n_still = build(source, meth, vals)
    except ValueError as e:
        sys.exit(f"make_pa_stills: {e}")
    out = pathlib.Path(a.out)
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out)
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_pa_stills: cannot write/read back {out}: {e}")
    if len(back) != 309 or list(back.columns) != NEED:
        sys.exit(f"make_pa_stills: read-back shape wrong: rows={len(back)} cols={list(back.columns)}")
    for col in ("fl_mm", "mt_mm"):
        if not (back[col].round(4) == round(vals[col], 4)).all():
            sys.exit(f"make_pa_stills: {col} not constant at {vals[col]} on read-back")
    if back.pa_deg.isna().any():
        sys.exit("make_pa_stills: NaN in pa_deg on read-back")
    video_rows = [i for i, n in enumerate(back.image_id) if meth.get(n) == "video"]
    if not (back.pa_deg.iloc[video_rows].round(4) == round(vals["pa_deg"], 4)).all():
        sys.exit("make_pa_stills: a video row does not carry the PA median on read-back")
    n_meas = int((back.pa_deg.round(4) != round(vals["pa_deg"], 4)).sum())
    print(f"wrote {out}  rows={len(back)}  stills={n_still}  video={len(video_rows)}  "
          f"measured(non-median) PA rows={n_meas}  still PA median={back.pa_deg.iloc[[i for i in range(len(back)) if meth[back.image_id[i]]=='still']].median():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
