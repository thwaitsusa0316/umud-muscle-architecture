"""The official UMUD score, transcribed from the host's published scorer.

Source: https://www.kaggle.com/code/paulritsche/umud-score (linked from the
competition Overview). Reproduced here so we can score locally instead of
spending leaderboard submissions to learn anything.

    S = (1/3) * [ MAE_PA / 6.0 deg  +  MAE_FL / 12.0 mm  +  MAE_MT / 3.0 mm ]
        + 1e-6 * (same with median absolute error)
        + 1e-9 * (same with RMSE)

The tolerances are the host's, "based on literature MDCs". Weights are equal and
each row (image) counts once. The tie-breaker terms are ~7 orders of magnitude
below the primary term and cannot affect ranking except at a near-exact tie.

What the tolerances imply for effort allocation
-----------------------------------------------
One tolerance-unit of error costs the same in each target, so the targets are
worth 6 deg of PA == 12 mm of FL == 3 mm of MT. Muscle thickness is the tightest
in absolute terms and is also the easiest quantity to measure (the perpendicular
distance between two aponeuroses, no fascicle extrapolation), which makes it the
best value per unit of work. Fascicle length is the loosest and by far the hardest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ("pa_deg", "fl_mm", "mt_mm")
TOLERANCES = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
WEIGHTS = {"pa_deg": 1.0, "fl_mm": 1.0, "mt_mm": 1.0}
EPS_SECONDARY = 1e-6
EPS_TERTIARY = 1e-9


def umud_score(solution: pd.DataFrame, submission: pd.DataFrame,
               row_id_column_name: str = "image_id") -> float:
    """Official score. Lower is better."""
    merged = solution.merge(submission, on=row_id_column_name, how="inner",
                            suffixes=("_true", "_pred"))
    if len(merged) != len(solution):
        raise ValueError(f"submission is missing {len(solution) - len(merged)} row ids")
    primary = secondary = tertiary = 0.0
    wsum = float(sum(WEIGHTS[c] for c in TARGETS))
    for c in TARGETS:
        yt = merged[f"{c}_true"].to_numpy(dtype=float)
        yp = merged[f"{c}_pred"].to_numpy(dtype=float)
        tau, w = TOLERANCES[c], WEIGHTS[c]
        err = np.abs(yp - yt)
        primary += w * (err.mean() / tau)
        secondary += w * (np.median(err) / tau)
        tertiary += w * (np.sqrt(np.mean((yp - yt) ** 2)) / tau)
    return float(primary / wsum + EPS_SECONDARY * secondary / wsum
                 + EPS_TERTIARY * tertiary / wsum)


def report(solution: pd.DataFrame, submission: pd.DataFrame,
           row_id_column_name: str = "image_id") -> pd.DataFrame:
    """Per-target MAE, its normalised contribution, and bias.

    `contribution` is what that target adds to the final score, so it says
    directly where the remaining points are.
    """
    merged = solution.merge(submission, on=row_id_column_name, how="inner",
                            suffixes=("_true", "_pred"))
    rows = []
    for c in TARGETS:
        yt = merged[f"{c}_true"].to_numpy(dtype=float)
        yp = merged[f"{c}_pred"].to_numpy(dtype=float)
        err = yp - yt
        mae = float(np.abs(err).mean())
        rows.append({"target": c, "tolerance": TOLERANCES[c], "mae": mae,
                     "medae": float(np.median(np.abs(err))),
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "bias": float(err.mean()),
                     "contribution": mae / TOLERANCES[c] / len(TARGETS)})
    return pd.DataFrame(rows)


def budget(target_score: float) -> pd.DataFrame:
    """Per-target MAE budget if a target score were met with equal effort.

    Useful as a sanity check on whether a leaderboard position is even physically
    plausible given known inter-rater disagreement.
    """
    per = target_score * len(TARGETS) / sum(WEIGHTS.values())
    return pd.DataFrame([{"target": c, "tolerance": TOLERANCES[c],
                          "allowed_mae": per * TOLERANCES[c]} for c in TARGETS])
