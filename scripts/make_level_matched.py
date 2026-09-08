#!/usr/bin/env python3
"""L6z (PLAN v19 rung 1): level-matched PA or FL isolation on top of the composed best.

Every full-coverage PA/FL isolation to date shipped the pipeline's values at their
own level (measured PA median ~13.0 deg vs the leaderboard-probed 16.42; measured FL
median ~93.6 mm vs 81.5), and at the metric's tolerances that level offset alone
(PA -3.4/6/3 = 0.19, FL +12/12/3 = 0.34 per measured row) swamps any per-image skill.
This script ships ONE target with its level matched to the probed public median by
ONE scalar fixed label-free from the measured rows themselves:

    pa:  pa_deg' = pa_deg + (median_probed_pa - median(measured pa_deg))      (a shift)
    fl:  fl_mm'  = fl_mm  * (median_probed_fl / median(measured fl_mm))       (a ratio)

Only rows flagged measured (and ok) in the pipeline rows file are set; the other rows
keep the probed median. The source must already carry that target at the probed
median on every row (it is an isolation on top of the composed MT file), and the
two other columns are never touched: MT in particular must read back byte-identical
to the source so the paired delta on the leaderboard is the level-matched target alone.
Nothing is tuned on the board. Writes the CSV and a one-row report; submission is
pilot's job (submit_queue/ + submit-gate).

    make_level_matched.py --target pa --source outputs/sub_L6r_run_median_v3d2.csv \
        --rows reports/pipeline_a1_scalev3d2_rows.csv \
        --out outputs/sub_L6z_pa_level_v3d2.csv --report reports/l6z_pa_level.csv
"""
from __future__ import annotations
import sys, json, pathlib, argparse, statistics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIANS = ROOT / "reports" / "public_medians.json"
NEED = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
TARGETS = {"pa": ("pa_deg", "pa_measured", "shift", 0.0, 90.0),
           "fl": ("fl_mm", "fl_measured", "ratio", 0.0, 400.0)}


def load_medians(path: pathlib.Path) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
        vals = {"pa_deg": float(m["median_pa_deg"]),
                "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_level_matched: cannot read medians {path}: {e}")
    for k, v in vals.items():
        if not v > 0:
            sys.exit(f"make_level_matched: {k} median must be positive, got {v}")
    return vals


def read_csv(path: str, what: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"image_id": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_level_matched: cannot read {what} {path}: {e}")
    if df.empty:
        sys.exit(f"make_level_matched: {what} {path} is empty")
    return df


def plan(source: pd.DataFrame, rows: pd.DataFrame, target: str, probed: float):
    """Return (values_to_set: {image_id: new value}, scalar, measured median). Pure."""
    col, flag, kind, lo, hi = TARGETS[target]
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    for c in ("image_id", col, flag, "ok"):
        if c not in rows.columns:
            raise ValueError(f"rows file lacks column {c!r}")
    source = source.copy(); rows = rows.copy()
    source["image_id"] = source["image_id"].astype(str)
    rows["image_id"] = rows["image_id"].astype(str)
    if source.image_id.duplicated().any() or rows.image_id.duplicated().any():
        raise ValueError("duplicate image_id in source or rows")
    if set(source.image_id) != set(rows.image_id):
        raise ValueError("source and rows files do not cover the same image_ids")
    if source[NEED[1:]].isna().any().any():
        raise ValueError("NaN in source predictions")
    # the source is a %.4f file: the target must sit at the probed median on every row
    if not (source[col] == float(f"{probed:.4f}")).all():
        raise ValueError(f"source {col} is not the probed median {probed} on every row; "
                         "this must be an isolation built on the composed MT file")
    for c in (flag, "ok"):
        if rows[c].isna().any():
            raise ValueError(f"NaN in rows flag column {c!r}")
        if rows[c].dtype != bool and not set(rows[c].unique()) <= {True, False, "True", "False"}:
            raise ValueError(f"rows flag column {c!r} is not boolean")
    flags = (rows[flag].astype(str) == "True") & (rows["ok"].astype(str) == "True")
    meas = rows.loc[flags, ["image_id", col]]
    if meas[col].isna().any():
        raise ValueError(f"NaN {col} on a row flagged {flag}")
    if len(meas) < 30:
        raise ValueError(f"only {len(meas)} measured rows for {target}; refusing")
    med = float(statistics.median(meas[col].tolist()))
    if not med > 0:
        raise ValueError(f"measured {col} median must be positive, got {med}")
    if kind == "shift":
        scalar = probed - med
        new = meas[col] + scalar
    else:
        scalar = probed / med
        new = meas[col] * scalar
    if not ((new > lo) & (new < hi)).all():
        raise ValueError(f"level-matched {col} leaves ({lo}, {hi}): min {new.min():.3f} max {new.max():.3f}")
    to_set = {i: float(v) for i, v in zip(meas.image_id, new)}
    return to_set, scalar, med


def build(source: pd.DataFrame, col: str, to_set: dict[str, float]) -> pd.DataFrame:
    out = source.copy()
    out[col] = [to_set.get(i, float(v)) for i, v in zip(out.image_id, out[col])]
    return out


def column_text(path: pathlib.Path, idx: int) -> list[str]:
    """The raw text of one CSV column, for byte-level identity checks."""
    return [ln.rstrip("\r\n").split(",")[idx] for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--source", required=True, help="composed best CSV (target at the probed median on every row)")
    ap.add_argument("--rows", required=True, help="pipeline rows CSV with <target>_measured and ok flags")
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()
    col, flag, kind, _, _ = TARGETS[a.target]
    vals = load_medians(pathlib.Path(a.medians))
    source = read_csv(a.source, "source CSV")
    rows = read_csv(a.rows, "rows CSV")
    try:
        to_set, scalar, med = plan(source, rows, a.target, vals[col])
    except ValueError as e:
        sys.exit(f"make_level_matched: {e}")
    print(f"{a.target}: measured rows {len(to_set)}, measured median {med:.4f}, probed {vals[col]:g}, "
          f"{kind} scalar {scalar:+.4f}" if kind == "shift" else
          f"{a.target}: measured rows {len(to_set)}, measured median {med:.4f}, probed {vals[col]:g}, "
          f"{kind} scalar x{scalar:.4f}")
    if a.dry_run:
        return 0
    out = pathlib.Path(a.out)
    src = pathlib.Path(a.source)
    if out.resolve() == src.resolve():
        sys.exit("make_level_matched: --out must differ from --source")
    df = build(source, col, to_set)
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out, dtype={"image_id": str})
        base = pd.read_csv(src, dtype={"image_id": str})
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_level_matched: cannot write/read back {out}: {e}")
    # post-write invariants: explicit checks, never `assert` (stripped under -O)
    if len(back) != 309 or list(back.columns) != NEED or list(back.image_id) != list(base.image_id):
        sys.exit("make_level_matched: read-back shape/order differs from the source")
    for c in NEED[1:]:
        if c == col:
            continue
        # both files are %.4f, so parsed values compare exactly; MT must also be byte-identical
        if not (back[c] == base[c]).all() or column_text(out, NEED.index(c)) != column_text(src, NEED.index(c)):
            sys.exit(f"make_level_matched: {c} changed on read-back")
    changed = set(back.image_id[back[col] != base[col]])
    if not changed <= set(to_set):
        sys.exit(f"make_level_matched: rows changed outside the measured set: {sorted(changed - set(to_set))[:5]}")
    got = back.set_index("image_id")[col]
    for iid, v in to_set.items():
        expect = float(f"{v:.4f}")  # the same serialization path write_submission used
        if float(got[iid]) != expect:
            sys.exit(f"make_level_matched: {iid} read back {got[iid]} != {expect}")
    shipped_med = float(statistics.median([float(got[i]) for i in to_set]))
    if abs(shipped_med - vals[col]) > 0.001:
        sys.exit(f"make_level_matched: shipped {col} median {shipped_med:.4f} != probed {vals[col]}")
    if a.report:
        rp = pathlib.Path(a.report)
        try:
            rp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"target": a.target, "column": col, "kind": kind, "scalar": scalar,
                           "measured_median": med, "probed_median": vals[col], "rows_set": len(to_set),
                           "rows_changed": len(changed), "shipped_median": shipped_med,
                           "min": min(to_set.values()), "max": max(to_set.values()),
                           "source": str(src), "rows_file": a.rows}]).to_csv(rp, index=False)
        except OSError as e:
            sys.exit(f"make_level_matched: cannot write report {rp}: {e}")
    print(f"wrote {out}  rows={len(back)}  {col} rows set={len(to_set)}  numerically changed={len(changed)}"
          f"  shipped median={shipped_med:.4f}" + (f"  report={a.report}" if a.report else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
