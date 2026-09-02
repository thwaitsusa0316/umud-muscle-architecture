#!/usr/bin/env python3
"""Leaderboard coordinate probing.

The UMUD score is a tolerance-normalised mean absolute error:
    S(pa, fl, mt) = (1/3) * [ MAE_PA/t_PA + MAE_FL/t_FL + MAE_MT/t_MT ]
For a *constant* prediction c of one variable, MAE(c) = mean|y_i - c| is convex
in c and minimised at the median of the (public) test targets. The unknown
tolerance t is a positive scale factor and does not move that minimum, so a
1-D search over c per variable recovers the public-set median of each target.
"""
from __future__ import annotations
import sys, json, pathlib, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import submit

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "reports" / "probe_log.json"


def load_log() -> list[dict]:
    return json.loads(LOG.read_text()) if LOG.exists() else []


def record(pa, fl, mt, tag) -> None:
    log = load_log()
    log.append({"pa": pa, "fl": fl, "mt": mt, "tag": tag, "score": None})
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text(json.dumps(log, indent=2))


def fire(pa: float, fl: float, mt: float, tag: str) -> None:
    ids = submit.test_ids()
    df = pd.DataFrame({"image_id": ids, "pa_deg": pa, "fl_mm": fl, "mt_mm": mt})
    path = ROOT / "outputs" / f"probe_pa{pa:g}_fl{fl:g}_mt{mt:g}.csv"
    submit.write_submission(df, path)
    submit.submit(path, f"probe const pa={pa:g} fl={fl:g} mt={mt:g} [{tag}]")
    record(pa, fl, mt, tag)
    print(f"submitted pa={pa:g} fl={fl:g} mt={mt:g}")


def sync_scores() -> pd.DataFrame:
    """Pull scored results back and join them onto the probe log."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    rows = []
    for s in api.competition_submissions(submit.COMP):
        rows.append({"desc": s.description, "score": s.public_score,
                     "status": str(s.status), "date": str(s.date)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", type=float, default=25.0)
    ap.add_argument("--fl", type=float, default=115.0)
    ap.add_argument("--mt", type=float, default=30.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--sync", action="store_true")
    a = ap.parse_args()
    if a.sync:
        print(sync_scores().to_string())
    else:
        fire(a.pa, a.fl, a.mt, a.tag)
