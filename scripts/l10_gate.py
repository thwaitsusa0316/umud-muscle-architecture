#!/usr/bin/env python3
"""L10 label-free go/no-go (PLAN v27) for the fascicle seed ensemble.

Gate 1 (benchmark, real GT): the per-image MEDIAN FL over the ensemble members must beat
the single pw6 model's 5.95 mm MAE by >= 10 % -> MAE < --mae-max (5.35 mm).
Gate 2 (test frames, label-free): per-image FL must agree across seeds, mean pairwise
Spearman over the commonly measured test rows > --rho-min (0.2); below that the detectors
are not measuring the same structure and a median is not noise-cancelling.

Inputs (written by scripts/l10_train_seeds.sh):
  reports/l10/bench_rows_<tag>.csv                          per-member benchmark rows
  reports/l10/pipeline_a1_scalev3d2_flip_<tag>_rows.csv     per-member test rows (seeds)
  reports/pipeline_a1_scalev3d2_flip_rows.csv               the shipped pw6 member (base)
Outputs: reports/l10/gate.json, reports/l10/ensemble_fl_rows.csv (image_id, fl_mm_median,
n_members, fl_mm_base). Exit 0 = GO, 1 = NO-GO, 2 = inputs missing.
"""
from __future__ import annotations
import sys, json, glob, pathlib, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_members(paths: list[str], key: str, need: str | None) -> dict[str, pd.Series]:
    out = {}
    for p in paths:
        d = pd.read_csv(p)
        if need is not None and need in d.columns:
            d = d[d[need].astype(bool)]
        s = pd.to_numeric(d[key], errors="coerce")
        s.index = d["image_id"].astype(str).values
        out[pathlib.Path(p).stem] = s.dropna()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l10-dir", default=str(ROOT / "reports" / "l10"))
    ap.add_argument("--base-rows", default=str(ROOT / "reports" / "pipeline_a1_scalev3d2_flip_rows.csv"))
    ap.add_argument("--mae-max", type=float, default=5.35)
    ap.add_argument("--rho-min", type=float, default=0.2)
    ap.add_argument("--min-members", type=int, default=3, help="members needed per image for a median")
    a = ap.parse_args()
    L = pathlib.Path(a.l10_dir)

    bench_paths = sorted(glob.glob(str(L / "bench_rows_*.csv")))
    seed_paths = sorted(glob.glob(str(L / "pipeline_a1_scalev3d2_flip_pw6s*_rows.csv")))
    if len(bench_paths) < 2 or len(seed_paths) < 1 or not pathlib.Path(a.base_rows).is_file():
        print(f"ERROR: need >=2 bench files (have {len(bench_paths)}), >=1 seed rows file "
              f"(have {len(seed_paths)}) and base rows {a.base_rows}", file=sys.stderr)
        return 2

    # ---- gate 1: benchmark ensemble-median FL MAE vs the expert consensus
    from umud import validation as V
    truth = V.load().set_index(V.load()["image_id"].astype(str))
    bench = load_members(bench_paths, "fl_mm", None)
    B = pd.DataFrame(bench)
    n_b = B.notna().sum(axis=1)
    med_b = B.median(axis=1)[n_b >= a.min_members]
    common = med_b.index.intersection(truth.index)
    err = (med_b.loc[common] - truth.loc[common, "fl_mm"])
    mae_ens = float(err.abs().mean()) if len(common) else float("nan")
    single = {k: float((v.reindex(common) - truth.loc[common, "fl_mm"]).abs().mean()) for k, v in bench.items()}

    # ---- gate 2: inter-seed Spearman of per-image FL on the measured test rows
    seeds = load_members(seed_paths, "fl_mm", "fl_measured")
    seeds["base_pw6"] = load_members([a.base_rows], "fl_mm", "fl_measured")["pipeline_a1_scalev3d2_flip_rows"]
    T = pd.DataFrame(seeds)
    names = list(T.columns)
    rhos = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = T[[names[i], names[j]]].dropna()
            if len(pair) >= 10:
                rhos[f"{names[i]}|{names[j]}"] = float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0])
    rho_mean = float(np.mean(list(rhos.values()))) if rhos else float("nan")
    rho_min = float(np.min(list(rhos.values()))) if rhos else float("nan")

    n_t = T.notna().sum(axis=1)
    med_t = T.median(axis=1)
    ens = pd.DataFrame({"image_id": T.index, "fl_mm_median": med_t.values, "n_members": n_t.values,
                        "fl_mm_base": T["base_pw6"].reindex(T.index).values})
    ens.loc[ens.n_members < a.min_members, "fl_mm_median"] = np.nan
    ens.to_csv(L / "ensemble_fl_rows.csv", index=False)

    go1 = bool(np.isfinite(mae_ens) and mae_ens < a.mae_max)
    go2 = bool(np.isfinite(rho_mean) and rho_mean > a.rho_min)
    res = dict(members_bench=list(bench), members_test=names, n_bench_images=int(len(common)),
               bench_fl_mae_ensemble_median=mae_ens, bench_fl_mae_single=single, mae_max=a.mae_max,
               gate1_pass=go1, test_pairwise_spearman=rhos, rho_mean=rho_mean, rho_min=rho_min,
               rho_min_required=a.rho_min, gate2_pass=go2,
               n_test_rows_with_median=int((ens.n_members >= a.min_members).sum()),
               n_test_rows_total=int(len(ens)), verdict="GO" if (go1 and go2) else "NO-GO")
    (L / "gate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nVERDICT {res['verdict']}: bench median-FL MAE {mae_ens:.3f} mm (< {a.mae_max}? {go1}); "
          f"inter-seed rho mean {rho_mean:.3f} min {rho_min:.3f} (> {a.rho_min}? {go2})")
    return 0 if res["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
