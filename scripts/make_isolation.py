#!/usr/bin/env python3
"""Per-target isolation submission (rung L4 of PLAN.md v2).

Take ONE measured column from a scored pipeline submission and hold the other two
targets at the leaderboard-probed medians (reports/public_medians.json). Against the
measured constant floor (1.03662, rung L1) the score attributes that target's share of
the pipeline's excess error. Writes the CSV only; submission is pilot's job.

    make_isolation.py --keep mt --source outputs/sub_pipeline_a1.csv
"""
from __future__ import annotations
import sys, json, pathlib, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIANS = ROOT / "reports" / "public_medians.json"
COLS = {"pa": "pa_deg", "fl": "fl_mm", "mt": "mt_mm"}
NEED = ["image_id", "pa_deg", "fl_mm", "mt_mm"]


def load_medians(path: pathlib.Path = MEDIANS) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"make_isolation: cannot read medians file {path}: {e}")
    try:
        vals = {"pa_deg": float(m["median_pa_deg"]),
                "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_isolation: medians file {path} malformed: {e}")
    # an angle of 0 is geometrically valid; lengths must be strictly positive
    if not (vals["pa_deg"] >= 0):
        sys.exit(f"make_isolation: pa_deg median must be >= 0, got {vals['pa_deg']}")
    for k in ("fl_mm", "mt_mm"):
        if not (vals[k] > 0):
            sys.exit(f"make_isolation: {k} median must be positive, got {vals[k]}")
    return vals


def build(source: pd.DataFrame, keep: str, vals: dict[str, float]) -> pd.DataFrame:
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    keep_col = COLS[keep]
    ids = submit.test_ids()
    src = source.set_index("image_id")
    missing = [i for i in ids if i not in src.index]
    if missing:
        raise ValueError(f"source lacks {len(missing)} test ids, e.g. {missing[:3]}")
    out = pd.DataFrame({"image_id": ids})
    for col in NEED[1:]:
        if col == keep_col:
            out[col] = src.loc[ids, col].to_numpy(dtype=float)
        else:
            out[col] = vals[col]
    if out[keep_col].isna().any():
        raise ValueError(f"NaN in kept column {keep_col}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", choices=sorted(COLS), required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--medians", default=str(MEDIANS))
    a = ap.parse_args()
    vals = load_medians(pathlib.Path(a.medians))
    try:
        source = pd.read_csv(a.source)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_isolation: cannot read source CSV {a.source}: {e}")
    if source.empty:
        sys.exit(f"make_isolation: source CSV {a.source} is empty")
    df = build(source, a.keep, vals)
    out = pathlib.Path(a.out) if a.out else ROOT / "outputs" / f"sub_L4_isolate_{a.keep}.csv"
    submit.write_submission(df, out)
    back = pd.read_csv(out)
    assert len(back) == 309 and list(back.columns) == NEED
    kept = COLS[a.keep]
    for col in NEED[1:]:
        if col != kept:
            assert (back[col].round(4) == round(vals[col], 4)).all(), f"{col} not constant on read-back"
    n_meas = int((back[kept].round(4) != round(vals[kept], 4)).sum())
    print(f"wrote {out}  rows={len(back)}  kept={kept}  measured(non-median) rows={n_meas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
