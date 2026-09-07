#!/usr/bin/env python3
"""Write a constant-prediction submission CSV from the leaderboard-probed medians.

Rung L1 of PLAN.md: one CSV, every test image gets PA/FL/MT = the public-set medians
recovered by scripts/probe.py (reports/public_medians.json). The separable-constant
model of the UMUD metric predicts 1.0374 for this file; the pre-registered falsifier
is |score - 1.0374| > 0.01. Writes the file only — submission is pilot's job
(submit_queue/ + submit-gate), never this script's.

Rung L4d (bracketing, PLAN v16): ``--set pa_deg=0`` / ``--set pa_deg=200`` overrides ONE
target with a constant outside the support of the truth (0 below every value, 200 above
every value); the other two targets stay at the medians. Because the UMUD metric is
separable, score(c=0) + score(c=200) = 200/(3*tol) + 2*(share of the other two targets),
which recovers the exact per-target split of the constant floor. Overrides may be 0
(the medians themselves must stay positive).
"""
from __future__ import annotations
import sys, json, pathlib, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIANS = ROOT / "reports" / "public_medians.json"


def load_medians(path: pathlib.Path = MEDIANS) -> dict[str, float]:
    m = json.loads(path.read_text())
    vals = {"pa_deg": float(m["median_pa_deg"]),
            "fl_mm": float(m["median_fl_mm"]),
            "mt_mm": float(m["median_mt_mm"])}
    for k, v in vals.items():
        if not (v > 0):
            raise ValueError(f"{k} median must be positive, got {v}")
    return vals


def apply_overrides(vals: dict[str, float], sets: list[str]) -> dict[str, float]:
    """Apply ``key=value`` overrides (key in pa_deg/fl_mm/mt_mm, value >= 0, finite)."""
    out = dict(vals)
    for item in sets:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in out:
            raise ValueError(f"--set key must be one of {sorted(out)}, got {k!r}")
        x = float(v)
        if not (x >= 0) or x != x or x in (float("inf"),):
            raise ValueError(f"--set {k} must be a finite number >= 0, got {v!r}")
        out[k] = x
    return out


def build(vals: dict[str, float]) -> pd.DataFrame:
    ids = submit.test_ids()
    return pd.DataFrame({"image_id": ids,
                         "pa_deg": vals["pa_deg"],
                         "fl_mm": vals["fl_mm"],
                         "mt_mm": vals["mt_mm"]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sub_L1_probed_median.csv"))
    ap.add_argument("--medians", default=str(MEDIANS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one target with a constant (pa_deg|fl_mm|mt_mm), e.g. --set pa_deg=0")
    a = ap.parse_args()
    vals = apply_overrides(load_medians(pathlib.Path(a.medians)), a.set)
    df = build(vals)
    out = submit.write_submission(df, pathlib.Path(a.out))
    back = pd.read_csv(out)
    assert len(back) == 309 and list(back.columns) == ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    for k, v in vals.items():
        assert (back[k].round(4) == round(v, 4)).all(), f"{k} not constant on read-back"
    print(f"wrote {out}  rows={len(back)}  pa={vals['pa_deg']:g} fl={vals['fl_mm']:g} mt={vals['mt_mm']:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
