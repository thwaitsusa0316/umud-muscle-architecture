"""Local validation harness built on the OSF expert benchmark.

35 muscle-architecture images, each scored independently by seven expert raters
using the same protocol the challenge used for its own test labels (three MT lines
at left/middle/right, three fascicles, three pennation angles, averaged). The sheet
also carries ground-truth `Scale_pixel_per_cm` and DL_Track's own predictions.

Source: https://osf.io/xbawc "Expert Analysed Benchmark Image Datasets".
This is external data under the competition rules and its use must be declared.
Note the licence conflict flagged in the README (OSF says CC BY 4.0, the bundled
usage policy says non-commercial, the host has not resolved it) -- treat as
validation-only until that is answered.

Why this matters: it turns model development into a measurable loop. Before it we
could only rank ideas by taste or by spending leaderboard submissions; now any
candidate pipeline gets an honest score, and we know what the score means because
the same harness prices the human raters and DL_Track on the same scale.
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

from .metric import TOLERANCES, TARGETS

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "external" / "benchmark_dataset_architecture_v0.1.0"
SHEET = BENCH / "Results_benchmark_architecture_v0.1.0.xlsx"
RATERS = [f"R{i}" for i in range(1, 8)]
SUFFIX = {"pa_deg": "PA", "fl_mm": "FL", "mt_mm": "MT"}


def load() -> pd.DataFrame:
    """Benchmark table: image path, ground-truth scale, and consensus targets.

    The consensus is the mean over all seven raters, mirroring how the challenge
    builds its own labels ("annotations from researchers were averaged for final
    per image architectural score").
    """
    df = pd.read_excel(SHEET)
    out = pd.DataFrame({
        "image_id": df["ImageID"],
        "path": [str(BENCH / f"{i}.tif") for i in df["ImageID"]],
        "px_per_cm": df["Scale_pixel_per_cm"].astype(float),
    })
    for col, suf in SUFFIX.items():
        out[col] = df[[f"{r}_{suf}" for r in RATERS]].mean(axis=1)
        out[f"{col}_sd"] = df[[f"{r}_{suf}" for r in RATERS]].std(axis=1)
    for col, suf in SUFFIX.items():
        out[f"dltrack_{col}"] = df[f"DLTrack_{suf}"].astype(float)
    return out


def score(pred: pd.DataFrame, truth: pd.DataFrame | None = None) -> float:
    """UMUD score of a prediction table against the expert consensus."""
    truth = load() if truth is None else truth
    m = truth.merge(pred, on="image_id", suffixes=("_true", "_pred"))
    if len(m) == 0:
        raise ValueError("no image_ids in common")
    return float(sum(np.abs(m[f"{c}_pred"] - m[f"{c}_true"]).mean() / TOLERANCES[c]
                     for c in TARGETS) / len(TARGETS))


def report(pred: pd.DataFrame, truth: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-target error and its share of the total score."""
    truth = load() if truth is None else truth
    m = truth.merge(pred, on="image_id", suffixes=("_true", "_pred"))
    rows = []
    for c in TARGETS:
        err = m[f"{c}_pred"] - m[f"{c}_true"]
        mae = float(err.abs().mean())
        rows.append({"target": c, "n": len(m), "mae": mae, "bias": float(err.mean()),
                     "rmse": float(np.sqrt((err ** 2).mean())),
                     "contribution": mae / TOLERANCES[c] / len(TARGETS)})
    d = pd.DataFrame(rows)
    d["share"] = d.contribution / d.contribution.sum()
    return d


def noise_floor() -> pd.DataFrame:
    """Each rater scored against the consensus of the other six.

    The mean of this is the practical ceiling: a model that reproduces an expert
    consensus better than the average expert does is either exceptional or fitted
    to the evaluation set.
    """
    df = pd.read_excel(SHEET)
    rows = []
    for r in RATERS:
        others = [o for o in RATERS if o != r]
        s = 0.0
        rec = {"rater": r}
        for col, suf in SUFFIX.items():
            cons = df[[f"{o}_{suf}" for o in others]].mean(axis=1)
            mae = float((df[f"{r}_{suf}"] - cons).abs().mean())
            rec[f"mae_{col}"] = mae
            s += mae / TOLERANCES[col]
        rec["umud"] = s / len(TARGETS)
        rows.append(rec)
    return pd.DataFrame(rows)


def baselines() -> dict[str, float]:
    """Reference scores everything else should be judged against."""
    t = load()
    dl = t[["image_id"] + [f"dltrack_{c}" for c in TARGETS]].rename(
        columns={f"dltrack_{c}": c for c in TARGETS})
    out = {"dltrack": score(dl, t), "expert_mean": float(noise_floor().umud.mean()),
           "expert_best": float(noise_floor().umud.min())}
    # best possible constant: predict each target's median
    const = pd.DataFrame({"image_id": t.image_id})
    for c in TARGETS:
        const[c] = t[c].median()
    out["best_constant"] = score(const, t)
    return out
