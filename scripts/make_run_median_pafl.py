#!/usr/bin/env python3
"""L7r (PLAN v21 rung 4): within-video-run median consensus for PA and/or FL on top of
the composed best, label-free, 0 compute.

The 161 test video frames come in runs (same probe placement, same scale, anatomy
fixed; run id = the `run` field of the scale manifest, NOT reports/groups.json). L6r
showed that on MT a run whose shipped values spread by more than 25 % of their median
is a frame-level fault, and setting every shipped frame in such a run to the run
median gave -0.005 on the board (threshold 0.15 fired the other way, so the rule is
confined to gross faults). This script applies the SAME rule to PA and FL on the
composed best: for each target, each video run with >= MIN_FRAMES measured frames
whose rel_range = (max - min) / median exceeds THRESHOLD has every measured frame of
that target set to the run median of the measured frames. ONE variable: within-run
median consensus at the L6r threshold. Targets are decided independently per run.

Rows not flagged <target>_measured and ok in the rows file must already sit at the
probed median in the source (the composed file puts them there) and are never
touched; columns not in --targets read back byte-identical. Nothing is tuned on the
board. Writes the CSV and a per-(target, run) report; submission is pilot's job.

    make_run_median_pafl.py --targets pa,fl --threshold 0.25 --min-frames 3 \
        --source outputs/sub_L6ob_flip_level_v3d2.csv \
        --rows reports/pipeline_a1_scalev3d2_flip_rows.csv \
        --scale reports/scale_v3d2.json \
        --out outputs/sub_L7r_run_median_pafl.csv --report reports/l7r_run_median_pafl.csv
"""
from __future__ import annotations
import sys, json, pathlib, argparse, statistics
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIANS = ROOT / "reports" / "public_medians.json"
NEED = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
TARGETS = {"pa": ("pa_deg", "pa_measured"),
           "fl": ("fl_mm", "fl_measured"),
           "mt": ("mt_mm", "mt_measured")}
N_ROWS = 309


def load_medians(path: pathlib.Path) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
        vals = {"pa_deg": float(m["median_pa_deg"]),
                "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_run_median_pafl: cannot read medians {path}: {e}")
    for k, v in vals.items():
        if not v > 0:
            sys.exit(f"make_run_median_pafl: {k} median must be positive, got {v}")
    return vals


def load_runs(path: pathlib.Path) -> tuple[dict[str, int], dict[str, str]]:
    try:
        s = json.loads(path.read_text())
        runs = {str(k): int(v["run"]) for k, v in s.items()}
        methods = {str(k): str(v.get("method", "")) for k, v in s.items()}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
        sys.exit(f"make_run_median_pafl: cannot read scale manifest {path}: {e}")
    return runs, methods


def read_csv(path: str, what: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"image_id": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_run_median_pafl: cannot read {what} {path}: {e}")
    if df.empty:
        sys.exit(f"make_run_median_pafl: {what} {path} is empty")
    return df


def validate(source: pd.DataFrame, rows: pd.DataFrame, runs: dict[str, int], targets: list[str]) -> None:
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    if len(source) != N_ROWS:
        raise ValueError(f"source has {len(source)} rows, expected {N_ROWS}")
    if source.image_id.duplicated().any() or rows.image_id.duplicated().any():
        raise ValueError("duplicate image_id in source or rows")
    if set(source.image_id) != set(rows.image_id):
        raise ValueError("source and rows files do not cover the same image_ids")
    if source[NEED[1:]].isna().any().any():
        raise ValueError("NaN in source predictions")
    missing = [i for i in source.image_id if i not in runs]
    if missing:
        raise ValueError(f"{len(missing)} source ids absent from the scale manifest, e.g. {missing[:3]}")
    for t in targets:
        flag = TARGETS[t][1]
        for c in (flag, "ok"):
            if c not in rows.columns:
                raise ValueError(f"rows file lacks column {c!r}")
            if rows[c].isna().any():
                raise ValueError(f"NaN in rows flag column {c!r}")
            if rows[c].dtype != bool and not set(rows[c].astype(str).unique()) <= {"True", "False"}:
                raise ValueError(f"rows flag column {c!r} is not boolean")


def measured_ids(rows: pd.DataFrame, source: pd.DataFrame, target: str, probed: float) -> set[str]:
    """image_ids flagged <target>_measured and ok. Every OTHER row must sit at the
    probed median in the source (the composed-file contract)."""
    col, flag = TARGETS[target]
    flags = (rows[flag].astype(str) == "True") & (rows["ok"].astype(str) == "True")
    ids = {str(i) for i in rows.loc[flags, "image_id"]}
    if len(ids) < 30:
        raise ValueError(f"only {len(ids)} measured rows for {target}; refusing")
    other = source.loc[~source.image_id.isin(ids), col]
    if not (other == float(f"{probed:.4f}")).all():
        raise ValueError(f"{col} is not the probed median {probed} on every non-measured row; "
                         "this must be built on the composed file")
    # the source must carry per-image values on the measured rows (an isolation of another
    # target has this column at the median everywhere and is the wrong baseline)
    carried = int((source.loc[source.image_id.isin(ids), col] != float(f"{probed:.4f}")).sum())
    if carried < 30:
        raise ValueError(f"only {carried} measured {col} rows differ from the probed median {probed}; "
                         "the source does not carry measured values for this target")
    return ids


def plan(source: pd.DataFrame, measured: dict[str, set[str]], runs: dict[str, int],
         methods: dict[str, str], method: str, threshold: float, min_frames: int):
    """Return ({target: {image_id: run_median}}, report rows). Pure: touches nothing."""
    vals = source.set_index("image_id")
    to_set: dict[str, dict[str, float]] = {t: {} for t in measured}
    report = []
    for t, ids in measured.items():
        col = TARGETS[t][0]
        by_run: dict[int, list[str]] = defaultdict(list)
        for iid in source.image_id:          # source order, deterministic
            if iid in ids and (not method or methods[iid] == method):
                by_run[runs[iid]].append(iid)
        for run, lst in sorted(by_run.items()):
            if len(lst) < min_frames:
                continue
            v = [float(vals.at[i, col]) for i in lst]
            med = statistics.median(v)
            if not med > 0:
                raise ValueError(f"{t} run {run}: non-positive run median {med}")
            rel = (max(v) - min(v)) / med
            smoothed = rel > threshold
            report.append({"target": t, "run": run, "n_measured": len(lst), "run_median": round(med, 4),
                           "rel_range": round(rel, 4), "smoothed": smoothed,
                           "method": ",".join(sorted({methods[i] for i in lst})), "ids": " ".join(lst)})
            if smoothed:
                for i in lst:
                    to_set[t][i] = med
    return to_set, report


def build(source: pd.DataFrame, to_set: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = source.copy()
    for t, m in to_set.items():
        col = TARGETS[t][0]
        out[col] = [m.get(i, float(v)) for i, v in zip(out.image_id, out[col])]
    return out


def column_text(path: pathlib.Path, idx: int) -> list[str]:
    """The raw text of one CSV column, for byte-level identity checks."""
    return [ln.rstrip("\r\n").split(",")[idx] for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="pa,fl", help="comma list from pa,fl,mt (default pa,fl)")
    ap.add_argument("--threshold", type=float, default=0.25, help="rel_range = (max-min)/median above which a run is smoothed")
    ap.add_argument("--min-frames", type=int, default=3, help="runs with fewer measured frames are left alone")
    ap.add_argument("--method", default="video", help="scale-manifest method to restrict runs to ('' = all)")
    ap.add_argument("--source", required=True, help="composed best CSV (baseline)")
    ap.add_argument("--rows", required=True, help="pipeline rows CSV with <target>_measured and ok flags")
    ap.add_argument("--scale", required=True, help="scale manifest JSON with `run` and `method` per image")
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()
    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    if not targets or any(t not in TARGETS for t in targets) or len(set(targets)) != len(targets):
        sys.exit(f"make_run_median_pafl: --targets must be a unique comma list from {sorted(TARGETS)}, got {a.targets!r}")
    if not (a.threshold > 0) or a.min_frames < 2:
        sys.exit("make_run_median_pafl: need threshold > 0 and min-frames >= 2")
    src = pathlib.Path(a.source)
    out = pathlib.Path(a.out)
    if out.resolve() == src.resolve():
        sys.exit("make_run_median_pafl: --out must differ from --source")
    medians = load_medians(pathlib.Path(a.medians))
    runs, methods = load_runs(pathlib.Path(a.scale))
    source = read_csv(a.source, "source CSV")
    rows = read_csv(a.rows, "rows CSV")
    try:
        validate(source, rows, runs, targets)
        measured = {t: measured_ids(rows, source, t, medians[TARGETS[t][0]]) for t in targets}
        to_set, report = plan(source, measured, runs, methods, a.method, a.threshold, a.min_frames)
    except ValueError as e:
        sys.exit(f"make_run_median_pafl: {e}")
    n_set = sum(len(m) for m in to_set.values())
    for t in targets:
        sm = [r for r in report if r["target"] == t and r["smoothed"]]
        n_runs = sum(1 for r in report if r["target"] == t)
        print(f"{t}: {n_runs} {a.method or 'all'} runs with >= {a.min_frames} measured frames; smoothed (rel_range > {a.threshold}): "
              f"{len(sm)} runs / {len(to_set[t])} rows")
        for r in sm:
            print(f"  run {r['run']}: n={r['n_measured']} median={r['run_median']:.2f} rel_range={r['rel_range']:.3f} [{r['method']}]")
    if n_set == 0:
        sys.exit("make_run_median_pafl: no run exceeds the threshold for any target; nothing to isolate (refusing to write a copy)")
    if a.dry_run:
        return 0
    df = build(source, to_set)
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out, dtype={"image_id": str})
        base = pd.read_csv(src, dtype={"image_id": str})
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_run_median_pafl: cannot write/read back {out}: {e}")
    # post-write invariants: explicit checks, never `assert` (stripped under -O)
    if len(back) != N_ROWS or list(back.columns) != NEED or list(back.image_id) != list(base.image_id):
        sys.exit("make_run_median_pafl: read-back shape/order differs from the source")
    touched_cols = {TARGETS[t][0] for t in targets}
    for c in NEED[1:]:
        if c in touched_cols:
            continue
        # both files are %.4f, so parsed values compare exactly; the text must match too
        if not (back[c] == base[c]).all() or column_text(out, NEED.index(c)) != column_text(src, NEED.index(c)):
            sys.exit(f"make_run_median_pafl: {c} changed on read-back")
    n_changed = 0
    for t in targets:
        col = TARGETS[t][0]
        changed = set(back.image_id[back[col] != base[col]])
        if not changed <= set(to_set[t]):
            sys.exit(f"make_run_median_pafl: {col} rows changed outside the planned set: {sorted(changed - set(to_set[t]))[:5]}")
        n_changed += len(changed)
        got = back.set_index("image_id")[col]
        for iid, v in to_set[t].items():
            # write_submission serialises with float_format="%.4f": the read-back must sit within
            # half a unit of the 4th decimal of the planned value (tolerance, not a string re-format)
            if abs(float(got[iid]) - v) > 0.00005 + 1e-9:
                sys.exit(f"make_run_median_pafl: {iid} {col} read back {got[iid]} differs from planned {v:.6f} by more than 5e-5")
    if a.report:
        rp = pathlib.Path(a.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report).to_csv(rp, index=False)
    print(f"wrote {out}  rows={len(back)}  rows set to run median={n_set}  rows numerically changed={n_changed}"
          + (f"  report={a.report}" if a.report else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
