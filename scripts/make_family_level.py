#!/usr/bin/env python3
"""L8f (PLAN v24): per-family level matching of PA or FL on top of the composed best,
label-free, 0 compute.

The composed best matches ONE global level (L6z: one scalar over all 297 measured
rows) and then shrinks toward the probed public median (L7f/L7p, alpha 0.5). But the
297 measured rows come from three image families whose shipped medians differ after
that global match: on the L8 file FL sits at tif 78.58 / png 68.71 / video 84.15 mm
against the probed 81.5, PA at tif 17.18 / png 19.14 / video 15.71 deg against 16.42.
This script moves each family to the probed median with ONE additive shift per family
fixed label-free from that family's own measured rows:

    x' = x + (median_probed - median(measured x in the family))

A shift (not a ratio) is used for both targets so the within-family dispersion the
shrinkage alpha already set is left untouched: the single variable is the per-family
level. Family = 'video' when the scale manifest method is video, else the file
extension ('tif' / 'png'). Only rows flagged <target>_measured and ok in the rows
file are set; the other rows must already sit at the probed median in the source and
are never touched; the two other columns read back byte-identical. Nothing is tuned
on the board. Writes the CSV and a per-family report; submission is pilot's job.

    make_family_level.py --target fl --source outputs/sub_L8_compose_a05.csv \
        --rows reports/pipeline_a1_scalev3d2_flip_rows.csv --scale reports/scale_v3d2.json \
        --out outputs/sub_L8f_family_fl_level.csv --report reports/l8f_family_fl_level.csv
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
TARGETS = {"pa": ("pa_deg", "pa_measured", 0.0, 90.0),
           "fl": ("fl_mm", "fl_measured", 0.0, 400.0)}
FAMILIES = ("tif", "png", "video")
MIN_FAMILY_ROWS = 10
N_ROWS = 309


def load_medians(path: pathlib.Path) -> dict[str, float]:
    try:
        m = json.loads(path.read_text())
        vals = {"pa_deg": float(m["median_pa_deg"]),
                "fl_mm": float(m["median_fl_mm"]),
                "mt_mm": float(m["median_mt_mm"])}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        sys.exit(f"make_family_level: cannot read medians {path}: {e}")
    for k, v in vals.items():
        if not v > 0:
            sys.exit(f"make_family_level: {k} median must be positive, got {v}")
    return vals


def load_families(path: pathlib.Path) -> dict[str, str]:
    """image_id -> family from the scale manifest ('video' by method, else extension)."""
    try:
        s = json.loads(path.read_text())
        fam = {}
        for k, v in s.items():
            method = str(v.get("method", ""))
            ext = str(k).rsplit(".", 1)[-1].lower() if "." in str(k) else ""
            if method == "video":
                fam[str(k)] = "video"
            elif ext in ("tif", "tiff"):
                fam[str(k)] = "tif"
            elif ext == "png":
                fam[str(k)] = "png"
            else:
                raise ValueError(f"{k}: method {method!r} / extension {ext!r} maps to no family")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
        sys.exit(f"make_family_level: cannot read scale manifest {path}: {e}")
    return fam


def read_csv(path: str, what: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"image_id": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_family_level: cannot read {what} {path}: {e}")
    if df.empty:
        sys.exit(f"make_family_level: {what} {path} is empty")
    return df


def plan(source: pd.DataFrame, rows: pd.DataFrame, families: dict[str, str], target: str,
         probed: float):
    """Return (values_to_set: {image_id: new value}, per-family report rows). Pure."""
    col, flag, lo, hi = TARGETS[target]
    if list(source.columns) != NEED:
        raise ValueError(f"source columns must be {NEED}, got {list(source.columns)}")
    for c in ("image_id", flag, "ok"):  # the rows file supplies flags only; predictions come from the source
        if c not in rows.columns:
            raise ValueError(f"rows file lacks column {c!r}")
    source = source.copy(); rows = rows.copy()
    source["image_id"] = source["image_id"].astype(str)
    rows["image_id"] = rows["image_id"].astype(str)
    if len(source) != N_ROWS:
        raise ValueError(f"source has {len(source)} rows, expected {N_ROWS}")
    if source.image_id.duplicated().any() or rows.image_id.duplicated().any():
        raise ValueError("duplicate image_id in source or rows")
    if set(source.image_id) != set(rows.image_id):
        raise ValueError("source and rows files do not cover the same image_ids")
    missing = set(source.image_id) - set(families)
    if missing:
        raise ValueError(f"{len(missing)} source image_ids missing from the scale manifest: {sorted(missing)[:5]}")
    if source[NEED[1:]].isna().any().any():
        raise ValueError("NaN in source predictions")
    for c in (flag, "ok"):
        if rows[c].isna().any():
            raise ValueError(f"NaN in rows flag column {c!r}")
        if rows[c].dtype != bool and not set(rows[c].unique()) <= {True, False, "True", "False"}:
            raise ValueError(f"rows flag column {c!r} is not boolean")
    flags = (rows[flag].astype(str) == "True") & (rows["ok"].astype(str) == "True")
    measured_ids = set(rows.loc[flags, "image_id"])
    src = source.set_index("image_id")[col]
    probed4 = float(f"{probed:.4f}")
    # every unmeasured row must already sit at the probed median: the source is the composed file
    unmeasured = [i for i in src.index if i not in measured_ids]
    off = [i for i in unmeasured if float(src[i]) != probed4]
    if off:
        raise ValueError(f"{len(off)} unmeasured rows not at the probed median {probed4} (e.g. {off[:3]}); "
                         "the source must be the composed file")
    by_fam: dict[str, list[str]] = defaultdict(list)
    for i in src.index:
        if i in measured_ids:
            by_fam[families[i]].append(i)
    if set(by_fam) - set(FAMILIES):
        raise ValueError(f"unexpected family {set(by_fam) - set(FAMILIES)}")
    to_set: dict[str, float] = {}
    report = []
    for fam in FAMILIES:
        ids = by_fam.get(fam, [])
        if len(ids) < MIN_FAMILY_ROWS:
            raise ValueError(f"family {fam} has only {len(ids)} measured rows (< {MIN_FAMILY_ROWS}); refusing")
        vals = [float(src[i]) for i in ids]
        med = float(statistics.median(vals))
        if not med > 0:
            raise ValueError(f"family {fam} {col} median must be positive, got {med}")
        shift = probed - med
        new = [v + shift for v in vals]
        if not all(lo < v < hi for v in new):
            raise ValueError(f"family {fam}: shifted {col} leaves ({lo}, {hi}): min {min(new):.3f} max {max(new):.3f}")
        for i, v in zip(ids, new):
            to_set[i] = float(v)
        report.append({"target": target, "column": col, "family": fam, "rows": len(ids),
                       "source_median": med, "probed_median": probed, "shift": shift,
                       "min_before": min(vals), "max_before": max(vals),
                       "min_after": min(new), "max_after": max(new)})
    if len(to_set) != len(measured_ids):
        raise ValueError("measured rows were not all assigned a family")
    if all(abs(r["shift"]) < 1e-6 for r in report):
        raise ValueError("every family already sits at the probed median; nothing to shift "
                         "(is the source an isolation file rather than the composed best?)")
    return to_set, report


def build(source: pd.DataFrame, col: str, to_set: dict[str, float]) -> pd.DataFrame:
    out = source.copy()
    out["image_id"] = out["image_id"].astype(str)
    out[col] = [to_set.get(i, float(v)) for i, v in zip(out.image_id, out[col])]
    return out


def column_text(path: pathlib.Path, idx: int) -> list[str]:
    """The raw text of one CSV column, for byte-level identity checks."""
    return [ln.rstrip("\r\n").split(",")[idx] for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--source", required=True, help="composed best CSV (target measured on the flagged rows, probed median elsewhere)")
    ap.add_argument("--rows", required=True, help="pipeline rows CSV with <target>_measured and ok flags")
    ap.add_argument("--scale", required=True, help="scale manifest JSON with `method` per image")
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()
    col, flag, _, _ = TARGETS[a.target]
    vals = load_medians(pathlib.Path(a.medians))
    families = load_families(pathlib.Path(a.scale))
    source = read_csv(a.source, "source CSV")
    rows = read_csv(a.rows, "rows CSV")
    try:
        to_set, report = plan(source, rows, families, a.target, vals[col])
    except ValueError as e:
        sys.exit(f"make_family_level: {e}")
    for r in report:
        print(f"{a.target} {r['family']:5s}: rows {r['rows']:3d}  source median {r['source_median']:.4f}  "
              f"probed {r['probed_median']:g}  shift {r['shift']:+.4f}  after [{r['min_after']:.2f}, {r['max_after']:.2f}]")
    if a.dry_run:
        return 0
    out = pathlib.Path(a.out)
    src = pathlib.Path(a.source)
    if out.resolve() == src.resolve():
        sys.exit("make_family_level: --out must differ from --source")
    df = build(source, col, to_set)
    try:
        submit.write_submission(df, out)
        back = pd.read_csv(out, dtype={"image_id": str})
        base = pd.read_csv(src, dtype={"image_id": str})
    except (OSError, AssertionError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        sys.exit(f"make_family_level: cannot write/read back {out}: {e}")
    # post-write invariants: explicit checks, never `assert` (stripped under -O)
    if len(back) != N_ROWS or list(back.columns) != NEED or list(back.image_id) != list(base.image_id):
        sys.exit("make_family_level: read-back shape/order differs from the source")
    for c in NEED[1:]:
        if c == col:
            continue
        if not (back[c] == base[c]).all() or column_text(out, NEED.index(c)) != column_text(src, NEED.index(c)):
            sys.exit(f"make_family_level: {c} changed on read-back")
    changed = set(back.image_id[back[col] != base[col]])
    if not changed <= set(to_set):
        sys.exit(f"make_family_level: rows changed outside the measured set: {sorted(changed - set(to_set))[:5]}")
    got = back.set_index("image_id")[col]
    for iid, v in to_set.items():
        expect = float(f"{v:.4f}")  # the same serialization path write_submission used
        if abs(float(got[iid]) - expect) > 5e-5:
            sys.exit(f"make_family_level: {iid} read back {got[iid]} != {expect}")
    for r in report:
        ids = [i for i in to_set if families[i] == r["family"]]
        shipped = float(statistics.median([float(got[i]) for i in ids]))
        r["shipped_median"] = shipped
        r["rows_changed"] = len(changed & set(ids))
        if abs(shipped - vals[col]) > 0.001:
            sys.exit(f"make_family_level: family {r['family']} shipped {col} median {shipped:.4f} "
                     f"is not the probed {vals[col]}")
    if a.report:
        rp = pathlib.Path(a.report)
        try:
            rp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report).assign(source=str(src), rows_file=a.rows, scale_file=a.scale).to_csv(rp, index=False)
        except OSError as e:
            sys.exit(f"make_family_level: cannot write report {rp}: {e}")
    print(f"wrote {out}  rows={len(back)}  {col} rows set={len(to_set)}  numerically changed={len(changed)}"
          + (f"  report={a.report}" if a.report else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
