#!/usr/bin/env python3
"""Write a constant-prediction submission CSV from the leaderboard-probed medians.

Rung L1 of PLAN.md: one CSV, every test image gets PA/FL/MT = the public-set medians
recovered by scripts/probe.py (reports/public_medians.json). The separable-constant
model of the UMUD metric predicts 1.0374 for this file; the pre-registered falsifier
is |score - 1.0374| > 0.01. Writes the file only — submission is pilot's job
(submit_queue/ + submit-gate), never this script's.
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
    a = ap.parse_args()
    vals = load_medians(pathlib.Path(a.medians))
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
