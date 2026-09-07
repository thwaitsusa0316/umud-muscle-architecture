#!/usr/bin/env python3
"""L6r (PLAN v17 rung 2): within-run MT median smoothing on an MT isolation.

The 161 test video frames come in runs (same probe placement, same scale, anatomy
fixed; run id = the `run` field of the scale manifest, NOT reports/groups.json whose
numbering differs). On a shipped MT isolation, a run whose shipped MT values spread
by more than THRESHOLD of their median is a frame-level pairing fault, not a scale
fault. This script replaces the MT of EVERY shipped frame in such a run by that run's
median (of its shipped frames). ONE variable: within-run median smoothing.

Rows that were not shipped (MT at the probed median) are never touched; PA/FL are
never touched. Writes the CSV and a per-run report; submission is pilot's job.

    make_run_median.py --source outputs/sub_L6g_isolate_mt_v3d2.csv \
        --scale reports/scale_v3d2.json --threshold 0.25 \
        --out outputs/sub_L6r_run_median_v3d2.csv --report reports/l6r_run_median.csv
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


def load_medians(path: pathlib.Path) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
        vals = {"pa_deg": float(m["median_pa_deg"]),
                "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_run_median: cannot read medians {path}: {e}")
    if not vals["mt_mm"] > 0:
        sys.exit(f"make_run_median: mt median must be positive, got {vals['mt_mm']}")
    return vals


def load_runs(path: pathlib.Path) -> tuple[dict[str, int], dict[str, str]]:
    try:
        s = json.loads(path.read_text())
        runs = {k: int(v["run"]) for k, v in s.items()}
        methods = {k: str(v.get("method", "")) for k, v in s.items()}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_run_median: cannot read scale manifest {path}: {e}")
    return runs, methods


def plan(source: pd.DataFrame, runs: dict[str, int], methods: dict[str, str],
         mt_median: float, threshold: float, min_frames: int):
    """Return (rows_to_set: {image_id: new_mt}, report rows) without touching source."""
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    source = source.copy()
    source["image_id"] = source["image_id"].astype(str)
    if source.image_id.duplicated().any():
        raise ValueError("duplicate image_id in source")
    missing = [i for i in source.image_id if i not in runs]
    if missing:
        raise ValueError(f"{len(missing)} source ids absent from the scale manifest, e.g. {missing[:3]}")
    if source.mt_mm.isna().any():
        raise ValueError("NaN in source mt_mm")
    # the source is a %.4f file, so parsed values are exact 4-dp decimals: compare exactly
    shipped = source[source.mt_mm != mt_median]
    by_run: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for iid, mt in zip(shipped.image_id, shipped.mt_mm):
        by_run[runs[iid]].append((iid, float(mt)))
    to_set: dict[str, float] = {}
    report = []
    for run, lst in sorted(by_run.items()):
        if len(lst) < min_frames:
            continue
        vals = [v for _, v in lst]
        med = statistics.median(vals)
        rel = (max(vals) - min(vals)) / med
        smoothed = rel > threshold
        report.append({"run": run, "n_shipped": len(lst), "median_mt": round(med, 4),
                       "rel_range": round(rel, 4), "smoothed": smoothed,
                       "method": ",".join(sorted({methods[i] for i, _ in lst})),
                       "ids": " ".join(i for i, _ in lst)})
        if smoothed:
            for iid, _ in lst:
                to_set[iid] = med
    return to_set, report


def build(source: pd.DataFrame, to_set: dict[str, float]) -> pd.DataFrame:
    out = source.copy()
    out["mt_mm"] = [to_set.get(i, float(m)) for i, m in zip(out.image_id, out.mt_mm)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="shipped MT isolation CSV (baseline)")
    ap.add_argument("--scale", required=True, help="scale manifest JSON with a `run` field per image")
    ap.add_argument("--threshold", type=float, default=0.25, help="rel_range = (max-min)/median above which a run is smoothed")
    ap.add_argument("--min-frames", type=int, default=3, help="runs with fewer shipped frames are left alone")
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()
    if not (a.threshold > 0) or a.min_frames < 2:
        sys.exit("make_run_median: need threshold > 0 and min-frames >= 2")
    vals = load_medians(pathlib.Path(a.medians))
    runs, methods = load_runs(pathlib.Path(a.scale))
    try:
        source = pd.read_csv(a.source, dtype={"image_id": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_run_median: cannot read source CSV {a.source}: {e}")
    if source.empty:
        sys.exit(f"make_run_median: source CSV {a.source} is empty")
    try:
        to_set, report = plan(source, runs, methods, vals["mt_mm"], a.threshold, a.min_frames)
    except ValueError as e:
        sys.exit(f"make_run_median: {e}")
    sm = [r for r in report if r["smoothed"]]
    print(f"runs with >= {a.min_frames} shipped frames: {len(report)}; smoothed (rel_range > {a.threshold}): "
          f"{len(sm)} runs / {sum(r['n_shipped'] for r in sm)} rows")
    for r in sm:
        print(f"  run {r['run']}: n={r['n_shipped']} median={r['median_mt']:.2f} rel_range={r['rel_range']:.3f} [{r['method']}]")
    if not sm:
        sys.exit("make_run_median: no run exceeds the threshold; nothing to isolate (refusing to write a copy)")
    if a.dry_run:
        return 0
    df = build(source, to_set)
    out = pathlib.Path(a.out)
    if out.resolve() == pathlib.Path(a.source).resolve():
        sys.exit("make_run_median: --out must differ from --source")
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out, dtype={"image_id": str})
        base = pd.read_csv(a.source, dtype={"image_id": str})
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_run_median: cannot write/read back {out}: {e}")
    # post-write invariants: explicit checks, never `assert` (stripped under -O)
    if len(back) != 309 or list(back.columns) != NEED or list(back.image_id) != list(base.image_id):
        sys.exit("make_run_median: read-back shape/order differs from the source")
    # base and out are both %.4f files, so parsed values compare exactly; never mix
    # %.4f (round-half-away) with .round(4) (round-half-even) in these checks
    for col in ("pa_deg", "fl_mm"):
        if not (back[col] == base[col]).all():
            sys.exit(f"make_run_median: {col} changed on read-back")
    changed = set(back.image_id[back.mt_mm != base.mt_mm])
    if not changed <= set(to_set):
        sys.exit(f"make_run_median: rows changed outside the smoothed runs: {sorted(changed - set(to_set))[:5]}")
    got_mt = back.set_index("image_id").mt_mm
    for iid, mt in to_set.items():
        expect = float(f"{mt:.4f}")  # the same serialization path write_submission used
        if float(got_mt[iid]) != expect:
            sys.exit(f"make_run_median: {iid} read back {got_mt[iid]} != run median {expect}")
    if a.report:
        rp = pathlib.Path(a.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report).to_csv(rp, index=False)
    print(f"wrote {out}  rows={len(back)}  rows set to run median={len(to_set)}  rows numerically changed={len(changed)}"
          + (f"  report={a.report}" if a.report else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
