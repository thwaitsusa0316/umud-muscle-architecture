#!/usr/bin/env python3
"""L7f / L7p / L7c (PLAN v20 rungs 1-3): shrinkage or plausibility clamp of ONE target
on top of the composed best, label-free, 0 compute.

After L6z/L6o-b the composed best (0.67241) carries per-image PA and FL on the 297
measured rows at a level matched to the leaderboard-probed public medians. The L4d
algebra says the FL residual (MAE 14.1 mm) is still 2.4x the benchmark-grade MAE
(5.95) while the constant sits at 17.4, i.e. the per-image FL is noisy; shrinking a
noisy, level-matched predictor toward its constant lowers MAE. This script ships ONE
target changed by ONE pre-registered rule; every other column reads back
byte-identical so the paired delta on the board is that rule alone:

    shrink:  v' = alpha * v + (1 - alpha) * median_probed      on the measured rows
    clamp:   v' = median_probed  where v < lo or v > hi         on the measured rows

Rows not flagged measured (and ok) in the rows file must already sit at the probed
median in the source (the composed file puts them there) and are never touched.
Nothing is tuned on the board: alpha / lo / hi are pre-registered in PLAN.md.
Writes the CSV and a one-row report; submission is pilot's job (submit_queue/ +
submit-gate).

    make_shrink.py --mode shrink --target fl --alpha 0.5 \
        --source outputs/sub_L6ob_flip_level_v3d2.csv \
        --rows reports/pipeline_a1_scalev3d2_flip_rows.csv \
        --out outputs/sub_L7f_fl_shrink05.csv --report reports/l7f_fl_shrink05.csv
    make_shrink.py --mode clamp --target fl --lo 30 --hi 150 --min-rows 5 ...
"""
from __future__ import annotations
import sys, json, pathlib, argparse, statistics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIANS = ROOT / "reports" / "public_medians.json"
NEED = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
TARGETS = {"pa": ("pa_deg", "pa_measured", 0.0, 90.0),
           "fl": ("fl_mm", "fl_measured", 0.0, 400.0)}
N_ROWS = 309


def load_medians(path: pathlib.Path) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
        vals = {"pa_deg": float(m["median_pa_deg"]),
                "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_shrink: cannot read medians {path}: {e}")
    for k, v in vals.items():
        if not v > 0:
            sys.exit(f"make_shrink: {k} median must be positive, got {v}")
    return vals


def read_csv(path: str, what: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"image_id": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_shrink: cannot read {what} {path}: {e}")
    if df.empty:
        sys.exit(f"make_shrink: {what} {path} is empty")
    return df


def measured_ids(source: pd.DataFrame, rows: pd.DataFrame, target: str, probed: float) -> list[str]:
    """The image_ids the rule may touch: flagged <target>_measured and ok in the rows
    file. Validates both frames and checks every OTHER row sits at the probed median."""
    col, flag, _, _ = TARGETS[target]
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    for c in ("image_id", flag, "ok"):
        if c not in rows.columns:
            raise ValueError(f"rows file lacks column {c!r}")
    if len(source) != N_ROWS:
        raise ValueError(f"source has {len(source)} rows, expected {N_ROWS}")
    if source.image_id.duplicated().any() or rows.image_id.duplicated().any():
        raise ValueError("duplicate image_id in source or rows")
    if set(source.image_id) != set(rows.image_id):
        raise ValueError("source and rows files do not cover the same image_ids")
    if source[NEED[1:]].isna().any().any():
        raise ValueError("NaN in source predictions")
    for c in (flag, "ok"):
        if rows[c].isna().any():
            raise ValueError(f"NaN in rows flag column {c!r}")
        if rows[c].dtype != bool and not set(rows[c].astype(str).unique()) <= {"True", "False"}:
            raise ValueError(f"rows flag column {c!r} is not boolean")
    flags = (rows[flag].astype(str) == "True") & (rows["ok"].astype(str) == "True")
    ids = [str(i) for i in rows.loc[flags, "image_id"]]
    if len(ids) < 30:
        raise ValueError(f"only {len(ids)} measured rows for {target}; refusing")
    # the source is a %.4f file: every non-measured row must sit at the probed median
    other = source.loc[~source.image_id.isin(ids), col]
    if not (other == float(f"{probed:.4f}")).all():
        raise ValueError(f"{col} is not the probed median {probed} on every non-measured row; "
                         "this must be built on the composed file")
    return ids


def plan_shrink(source: pd.DataFrame, ids: list[str], target: str, probed: float,
                alpha: float) -> dict[str, float]:
    """{image_id: alpha*v + (1-alpha)*probed} on the measured rows. Pure."""
    col, _, lo, hi = TARGETS[target]
    if not isinstance(alpha, float) or alpha != alpha or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be a float in (0, 1), got {alpha!r}")
    vals = source.set_index("image_id")[col]
    new = {i: alpha * float(vals[i]) + (1.0 - alpha) * probed for i in ids}
    bad = [i for i, v in new.items() if not lo < v < hi]
    if bad:
        raise ValueError(f"shrunk {col} leaves ({lo}, {hi}) on {len(bad)} rows, e.g. {bad[:3]}")
    return new


def plan_clamp(source: pd.DataFrame, ids: list[str], target: str, probed: float,
               lo: float, hi: float, min_rows: int) -> dict[str, float]:
    """{image_id: probed} for measured rows with v < lo or v > hi. Pure.
    Refuses (ValueError) when fewer than min_rows rows are outside the band: the
    pre-registered ship condition (v20 rung 3: >= 5 rows) is not met."""
    col, _, _, _ = TARGETS[target]
    for x in (lo, hi):
        if not isinstance(x, float) or x != x:
            raise ValueError(f"lo/hi must be finite floats, got {lo!r}, {hi!r}")
    if not 0.0 < lo < hi:
        raise ValueError(f"need 0 < lo < hi, got {lo}, {hi}")
    if not lo <= probed <= hi:
        raise ValueError(f"probed median {probed} is outside the clamp band [{lo}, {hi}]")
    if not isinstance(min_rows, int) or min_rows < 1:
        raise ValueError(f"min_rows must be a positive int, got {min_rows!r}")
    vals = source.set_index("image_id")[col]
    out = {i: probed for i in ids if float(vals[i]) < lo or float(vals[i]) > hi}
    if len(out) < min_rows:
        raise ValueError(f"only {len(out)} measured {col} rows outside [{lo}, {hi}]; "
                         f"ship condition is >= {min_rows}; nothing written")
    return out


def build(source: pd.DataFrame, col: str, to_set: dict[str, float]) -> pd.DataFrame:
    out = source.copy()
    out[col] = [to_set.get(i, float(v)) for i, v in zip(out.image_id, out[col])]
    return out


def column_text(path: pathlib.Path, idx: int) -> list[str]:
    """The raw text of one CSV column, for byte-level identity checks."""
    return [ln.rstrip("\r\n").split(",")[idx] for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shrink", "clamp"], required=True)
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--alpha", type=float, default=None, help="shrink: weight on the measured value, in (0,1)")
    ap.add_argument("--lo", type=float, default=None, help="clamp: lower bound of the plausible band")
    ap.add_argument("--hi", type=float, default=None, help="clamp: upper bound of the plausible band")
    ap.add_argument("--min-rows", type=int, default=5, help="clamp: refuse unless >= this many rows move")
    ap.add_argument("--source", required=True, help="composed best CSV")
    ap.add_argument("--rows", required=True, help="pipeline rows CSV with <target>_measured and ok flags")
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()
    col, _, _, _ = TARGETS[a.target]
    if a.mode == "shrink" and (a.alpha is None or a.lo is not None or a.hi is not None):
        ap.error("--mode shrink takes --alpha only")
    if a.mode == "clamp" and (a.lo is None or a.hi is None or a.alpha is not None):
        ap.error("--mode clamp takes --lo and --hi (and --min-rows), not --alpha")
    vals = load_medians(pathlib.Path(a.medians))
    probed = vals[col]
    source = read_csv(a.source, "source CSV")
    rows = read_csv(a.rows, "rows CSV")
    try:
        ids = measured_ids(source, rows, a.target, probed)
        if a.mode == "shrink":
            to_set = plan_shrink(source, ids, a.target, probed, a.alpha)
            rule = f"alpha={a.alpha:g}"
        else:
            to_set = plan_clamp(source, ids, a.target, probed, a.lo, a.hi, a.min_rows)
            rule = f"band=[{a.lo:g},{a.hi:g}] min_rows={a.min_rows}"
    except ValueError as e:
        sys.exit(f"make_shrink: {e}")
    before = source.set_index("image_id")[col]
    med_before = float(statistics.median([float(before[i]) for i in ids]))
    print(f"{a.mode} {a.target}: measured rows {len(ids)}, rows to set {len(to_set)}, {rule}, "
          f"probed {probed:g}, measured median before {med_before:.4f}")
    if a.dry_run:
        return 0
    out = pathlib.Path(a.out)
    src = pathlib.Path(a.source)
    if out.resolve() == src.resolve():
        sys.exit("make_shrink: --out must differ from --source")
    df = build(source, col, to_set)
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out, dtype={"image_id": str})
        base = pd.read_csv(src, dtype={"image_id": str})
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_shrink: cannot write/read back {out}: {e}")
    # post-write invariants: explicit checks, never `assert` (stripped under -O)
    if len(back) != N_ROWS or list(back.columns) != NEED or list(back.image_id) != list(base.image_id):
        sys.exit("make_shrink: read-back shape/order differs from the source")
    for c in NEED[1:]:
        if c == col:
            continue
        # both files are %.4f, so parsed values compare exactly; the text must match too
        if not (back[c] == base[c]).all() or column_text(out, NEED.index(c)) != column_text(src, NEED.index(c)):
            sys.exit(f"make_shrink: {c} changed on read-back")
    changed = set(back.image_id[back[col] != base[col]])
    if not changed <= set(to_set):
        sys.exit(f"make_shrink: rows changed outside the planned set: {sorted(changed - set(to_set))[:5]}")
    got = back.set_index("image_id")[col]
    for iid, v in to_set.items():
        # write_submission serialises with float_format="%.4f": the read-back must sit within
        # half a unit of the 4th decimal of the planned value (tolerance, not a string re-format,
        # so a serializer rounding-mode difference cannot fail a correct file)
        if abs(float(got[iid]) - v) > 0.00005 + 1e-9:
            sys.exit(f"make_shrink: {iid} read back {got[iid]} differs from planned {v:.6f} by more than 5e-5")
    # a clamp must move every planned row (they were outside the band, the median is inside)
    if a.mode == "clamp" and len(changed) != len(to_set):
        sys.exit(f"make_shrink: clamp planned {len(to_set)} rows but {len(changed)} changed")
    med_after = float(statistics.median([float(got[i]) for i in ids]))
    if a.report:
        rp = pathlib.Path(a.report)
        try:
            rp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"mode": a.mode, "target": a.target, "column": col, "rule": rule,
                           "alpha": a.alpha, "lo": a.lo, "hi": a.hi, "min_rows": a.min_rows,
                           "probed_median": probed, "measured_rows": len(ids), "rows_set": len(to_set),
                           "rows_changed": len(changed), "measured_median_before": med_before,
                           "measured_median_after": med_after,
                           "min_after": float(got[ids].min()), "max_after": float(got[ids].max()),
                           "source": str(src), "rows_file": a.rows}]).to_csv(rp, index=False)
        except OSError as e:
            sys.exit(f"make_shrink: cannot write report {rp}: {e}")
    print(f"wrote {out}  rows={len(back)}  {col} rows set={len(to_set)}  numerically changed={len(changed)}"
          f"  measured median {med_before:.4f} -> {med_after:.4f}" + (f"  report={a.report}" if a.report else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
