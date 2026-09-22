#!/usr/bin/env python3
"""FL geometric-closure isolation (branch C of PLAN.md human gate 6, rung L9c).

For every row of a scored pipeline submission whose pa_deg is MEASURED (differs from the
leaderboard-probed median), replace fl_mm with the geometric closure

    fl_mm = mt_mm / sin(pa_deg)

using that row's own mt_mm and pa_deg. Rows whose pa_deg sits at the probed median keep
fl_mm at the probed median. pa_deg and mt_mm are then held at the probed medians on EVERY
row (isolation), so the public score reads the closure's FL term alone and is directly
comparable to the L9i FL isolation (0.83845). Writes the CSV only; submission is pilot's job.

    make_fl_closure.py --source outputs/sub_L8a_compose_fl075.csv --out outputs/sub_L9c_fl_closure_isolate.csv
"""
from __future__ import annotations
import sys, json, math, pathlib, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIANS = ROOT / "reports" / "public_medians.json"
NEED = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
FL_MAX_MM = 150.0  # plausibility ceiling used by the L7c clamp; closure rows above it are reported, never shipped silently


def load_medians(path: pathlib.Path = MEDIANS) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"make_fl_closure: cannot read medians file {path}: {e}")
    try:
        vals = {"pa_deg": float(m["median_pa_deg"]),
                "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_fl_closure: medians file {path} malformed: {e}")
    if not (0.0 < vals["pa_deg"] < 90.0):
        sys.exit(f"make_fl_closure: pa_deg median must lie in (0, 90), got {vals['pa_deg']}")
    for k in ("fl_mm", "mt_mm"):
        if not (vals[k] > 0):
            sys.exit(f"make_fl_closure: {k} median must be positive, got {vals[k]}")
    return vals


def build(source: pd.DataFrame, vals: dict[str, float]) -> tuple[pd.DataFrame, dict[str, float]]:
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    ids = submit.test_ids()
    src = source.set_index("image_id")
    if src.index.has_duplicates:
        raise ValueError("source has duplicate image_id rows")
    missing = [i for i in ids if i not in src.index]
    if missing:
        raise ValueError(f"source lacks {len(missing)} test ids, e.g. {missing[:3]}")
    pa = src.loc[ids, "pa_deg"].to_numpy(dtype=float)
    mt = src.loc[ids, "mt_mm"].to_numpy(dtype=float)
    if pd.isna(pa).any() or pd.isna(mt).any():
        raise ValueError("NaN in source pa_deg or mt_mm")
    measured = pa.round(4) != round(vals["pa_deg"], 4)
    if not (0.0 < pa[measured].min() and pa[measured].max() < 90.0):
        raise ValueError(f"measured pa_deg outside (0, 90): {pa[measured].min()}..{pa[measured].max()}")
    if not (mt[measured] > 0).all():
        raise ValueError("non-positive mt_mm on a measured-PA row")
    fl = [vals["fl_mm"]] * len(ids)
    for i in range(len(ids)):
        if measured[i]:
            fl[i] = mt[i] / math.sin(math.radians(pa[i]))
    out = pd.DataFrame({"image_id": ids,
                        "pa_deg": vals["pa_deg"],
                        "fl_mm": fl,
                        "mt_mm": vals["mt_mm"]})
    fl_meas = out.loc[measured, "fl_mm"]
    stats = {"n_measured": int(measured.sum()),
             "n_mt_at_median": int((mt[measured].round(4) == round(vals["mt_mm"], 4)).sum()),
             "fl_median": float(fl_meas.median()), "fl_min": float(fl_meas.min()),
             "fl_max": float(fl_meas.max()), "n_over_max": int((fl_meas > FL_MAX_MM).sum())}
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--report", default="", help="optional JSON path for the closure stats")
    a = ap.parse_args()
    vals = load_medians(pathlib.Path(a.medians))
    try:
        source = pd.read_csv(a.source)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_fl_closure: cannot read source CSV {a.source}: {e}")
    if source.empty:
        sys.exit(f"make_fl_closure: source CSV {a.source} is empty")
    try:
        df, stats = build(source, vals)
    except ValueError as e:
        sys.exit(f"make_fl_closure: {e}")
    out = pathlib.Path(a.out) if a.out else ROOT / "outputs" / "sub_L9c_fl_closure_isolate.csv"
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out)
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_fl_closure: cannot write/read back {out}: {e}")
    if len(back) != 309 or list(back.columns) != NEED:
        sys.exit(f"make_fl_closure: read-back shape wrong: rows={len(back)} cols={list(back.columns)}")
    for col in ("pa_deg", "mt_mm"):
        if not (back[col].round(4) == round(vals[col], 4)).all():
            sys.exit(f"make_fl_closure: {col} not constant at {vals[col]} on read-back")
    if back["fl_mm"].isna().any() or not (back["fl_mm"] > 0).all():
        sys.exit("make_fl_closure: NaN or non-positive fl_mm on read-back")
    n_meas_back = int((back["fl_mm"].round(4) != round(vals["fl_mm"], 4)).sum())
    if n_meas_back > stats["n_measured"]:
        sys.exit(f"make_fl_closure: read-back has {n_meas_back} non-median FL rows > {stats['n_measured']} measured")
    stats["source"] = str(a.source)
    stats["out"] = str(out)
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out}  rows={len(back)}  closure rows={stats['n_measured']} "
          f"(mt at median on {stats['n_mt_at_median']})  fl median={stats['fl_median']:.2f} "
          f"range {stats['fl_min']:.2f}-{stats['fl_max']:.2f}  n>{FL_MAX_MM:.0f}={stats['n_over_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
