#!/usr/bin/env python3
"""Per-target isolation routed by frame type: one target measured on STILL frames only.

Written for PLAN v6 rung L6c-pa (PA on stills); generalised for v12 rung L6m (MT on stills).

Label-free finding 2026-09-05 (reports/pipeline_a1_scalev2_rows.csv x reports/scale_v2.json):
measured PA on the 148 800x1200 STILLS has median 15.91 deg, on the probed 16.42; the
161 VIDEO frames (111 at 800x1200, 50 on the native 644x1088 grid) have median 11.3 deg.
The -21 % "PA bias" is therefore the video frames, not an aspect stretch. L6c-pa shipped
measured PA on stills only (1.04140, fired). L6m (v12) asks the same question of MT:
measured MT on stills only, probed median on video frames, PA/FL at the probed medians,
read against the all-rows MT isolation (v3b 0.86256).

    make_pa_stills.py --keep pa --source outputs/sub_pipeline_a1_scalev2.csv --out outputs/sub_L6c_isolate_pa_stills.csv
    make_pa_stills.py --keep mt --source outputs/sub_pipeline_a1_scalev3b.csv --scale reports/scale_v3b.json \
                      --out outputs/sub_L6m_isolate_mt_stills.csv

--keep {pa,fl,mt}: the ONE target carried measured on still frames; that target on video
frames and the other two targets everywhere sit at the probed medians. Default pa (the
L6c-pa behaviour, unchanged). Writes the CSV only; submission is pilot's job (submit-gate
checks the test record).
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
N_TEST = 309  # fixed size of the competition test set; any other count is a data fault
COLS = {"pa": "pa_deg", "fl": "fl_mm", "mt": "mt_mm"}


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


def build(source: pd.DataFrame, meth: dict[str, str], vals: dict[str, float],
          keep: str = "pa") -> tuple[pd.DataFrame, int]:
    if keep not in COLS:
        raise ValueError(f"keep must be one of {sorted(COLS)}, got {keep!r}")
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    kept = COLS[keep]
    ids = submit.test_ids()
    if len(ids) != N_TEST or len(set(ids)) != N_TEST:
        raise ValueError(f"submit.test_ids() returned {len(ids)} ids ({len(set(ids))} unique), expected {N_TEST}")
    # label-based alignment: src.loc[ids] reorders the source to the official id order, so
    # the source CSV may be in any row order; missing ids are caught explicitly below.
    src = source.set_index("image_id")
    if src.index.has_duplicates:
        raise ValueError("source has duplicate image_id rows")
    missing = [i for i in ids if i not in src.index]
    if missing:
        raise ValueError(f"source lacks {len(missing)} test ids, e.g. {missing[:3]}")
    unknown = [i for i in ids if i not in meth]
    if unknown:
        raise ValueError(f"scale lookup lacks {len(unknown)} test ids, e.g. {unknown[:3]}")
    try:
        meas = src.loc[ids, kept].astype(float).to_numpy()
    except (TypeError, ValueError) as e:
        raise ValueError(f"source column {kept} is not numeric: {e}")
    if pd.isna(meas).any():
        raise ValueError(f"NaN in source {kept}")
    still = pd.Series([meth[i] == "still" for i in ids], dtype=bool).to_numpy()
    out = pd.DataFrame({"image_id": ids})
    for col in NEED[1:]:
        if col == kept:
            out[col] = [float(v) if s else vals[col] for v, s in zip(meas, still)]
        else:
            out[col] = vals[col]
    return out, int(still.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", choices=sorted(COLS), default="pa",
                    help="target carried measured on STILL frames; medians elsewhere (default pa)")
    ap.add_argument("--source", default=str(ROOT / "outputs" / "sub_pipeline_a1_scalev2.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sub_L6c_isolate_pa_stills.csv"))
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--scale", default=str(SCALE))
    a = ap.parse_args()
    kept = COLS[a.keep]
    vals = load_medians(pathlib.Path(a.medians))
    meth = load_methods(pathlib.Path(a.scale))
    try:
        source = pd.read_csv(a.source)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_pa_stills: cannot read source CSV {a.source}: {e}")
    if source.empty:
        sys.exit(f"make_pa_stills: source CSV {a.source} is empty")
    try:
        df, n_still = build(source, meth, vals, a.keep)
    except ValueError as e:
        sys.exit(f"make_pa_stills: {e}")
    out = pathlib.Path(a.out)
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out)
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_pa_stills: cannot write/read back {out}: {e}")
    if len(back) != N_TEST or list(back.columns) != NEED:
        sys.exit(f"make_pa_stills: read-back shape wrong: rows={len(back)} cols={list(back.columns)}")
    for col in NEED[1:]:
        if col != kept and not (back[col].round(4) == round(vals[col], 4)).all():
            sys.exit(f"make_pa_stills: {col} not constant at {vals[col]} on read-back")
    if back[kept].isna().any():
        sys.exit(f"make_pa_stills: NaN in {kept} on read-back")
    if list(back.image_id) != list(df.image_id):
        sys.exit("make_pa_stills: image_id order changed between build and read-back")
    is_still = [meth.get(n) == "still" for n in back.image_id]
    video_rows = [i for i, s in enumerate(is_still) if not s]
    still_rows = [i for i, s in enumerate(is_still) if s]
    if not (back[kept].iloc[video_rows].round(4) == round(vals[kept], 4)).all():
        sys.exit(f"make_pa_stills: a video row does not carry the {kept} median on read-back")
    n_meas = int((back[kept].round(4) != round(vals[kept], 4)).sum())
    still_med = float(back[kept].iloc[still_rows].median()) if still_rows else float("nan")
    print(f"wrote {out}  rows={len(back)}  kept={kept}  stills={n_still}  video={len(video_rows)}  "
          f"measured(non-median) {kept} rows={n_meas}  still {kept} median={still_med:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
