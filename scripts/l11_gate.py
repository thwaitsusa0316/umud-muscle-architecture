#!/usr/bin/env python3
"""L11 label-free go/no-go (PLAN v29) for ONE architecture-swapped fascicle detector.

Single-model gate (no ensemble): the new arch's checkpoint must
  gate 1 (benchmark, real GT): per-image FL MAE on the 35-image expert benchmark
          < --mae-max (5.35 mm; the shipped pw6 U-Net scores 5.724 on the same rows);
  gate 2 (test frames, label-free): fl_measured coverage >= --min-cover (280 of 309);
  gate 3 (test frames, label-free): Spearman of per-image fl_mm vs the shipped pw6 rows
          over the commonly measured test rows > --rho-min (0.2) -- below that the new
          detector is not measuring the same structure and the composed FL cannot be trusted.

Inputs (written by scripts/l11_train_arch.sh for --arch):
  reports/l11/bench_rows_<arch>.csv                          benchmark rows (l10_bench_rows.py)
  reports/l11/pipeline_a1_scalev3d2_flip_<arch>_rows.csv     production test rows
  reports/pipeline_a1_scalev3d2_flip_rows.csv                the shipped pw6 rows (--base-rows)
Output: reports/l11/gate_<arch>.json.  Exit 0 = GO, 1 = NO-GO, 2 = inputs missing.
"""
from __future__ import annotations
import sys, json, pathlib, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def fl_series(path: pathlib.Path, need: str | None) -> pd.Series:
    d = pd.read_csv(path)
    if need is not None and need in d.columns:
        d = d[d[need].astype(bool)]
    s = pd.to_numeric(d["fl_mm"], errors="coerce")
    s.index = d["image_id"].astype(str).values
    return s.dropna()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["fpn", "segformer", "unetpp", "l6t"],
                    help="file-name key under --l11-dir (l6t = the L6t-b self-trained unet, files under reports/l6t)")
    ap.add_argument("--l11-dir", default=str(ROOT / "reports" / "l11"))
    ap.add_argument("--base-rows", default=str(ROOT / "reports" / "pipeline_a1_scalev3d2_flip_rows.csv"))
    ap.add_argument("--mae-max", type=float, default=5.35)
    ap.add_argument("--min-cover", type=int, default=280)
    ap.add_argument("--rho-min", type=float, default=0.2)
    a = ap.parse_args()
    L = pathlib.Path(a.l11_dir)
    bench_p = L / f"bench_rows_{a.arch}.csv"
    test_p = L / f"pipeline_a1_scalev3d2_flip_{a.arch}_rows.csv"
    base_p = pathlib.Path(a.base_rows)
    missing = [str(p) for p in (bench_p, test_p, base_p) if not p.is_file()]
    if missing:
        print("ERROR: missing inputs: " + ", ".join(missing), file=sys.stderr)
        return 2

    # ---- gate 1: benchmark FL MAE vs the expert consensus (same loader as l10_gate.py)
    from umud import validation as V
    truth = V.load()
    truth = truth.set_index(truth["image_id"].astype(str))
    bench = fl_series(bench_p, None)
    common = bench.index.intersection(truth.index)
    mae = float((bench.loc[common] - truth.loc[common, "fl_mm"]).abs().mean()) if len(common) else float("nan")

    # ---- gate 2: measured-FL coverage on the 309 test frames
    test_all = pd.read_csv(test_p)
    n_total = int(len(test_all))
    cover = int(test_all["fl_measured"].astype(bool).sum()) if "fl_measured" in test_all.columns else 0

    # ---- gate 3: Spearman of per-image FL vs the shipped pw6 rows
    new = fl_series(test_p, "fl_measured")
    base = fl_series(base_p, "fl_measured")
    pair = pd.concat([new.rename("new"), base.rename("base")], axis=1).dropna()
    rho = float(spearmanr(pair["new"], pair["base"])[0]) if len(pair) >= 10 else float("nan")
    med_new = float(new.median()) if len(new) else float("nan")
    med_base = float(base.median()) if len(base) else float("nan")

    go1 = bool(np.isfinite(mae) and mae < a.mae_max)
    go2 = bool(cover >= a.min_cover)
    go3 = bool(np.isfinite(rho) and rho > a.rho_min)
    res = dict(arch=a.arch, n_bench_images=int(len(common)), bench_fl_mae=mae, mae_max=a.mae_max,
               gate1_pass=go1, fl_measured_cover=cover, n_test_rows=n_total, min_cover=a.min_cover,
               gate2_pass=go2, n_common_test_rows=int(len(pair)), spearman_vs_pw6=rho,
               rho_min=a.rho_min, gate3_pass=go3, test_fl_median_new=med_new,
               test_fl_median_pw6=med_base, verdict="GO" if (go1 and go2 and go3) else "NO-GO")
    (L / f"gate_{a.arch}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nVERDICT {res['verdict']}: bench FL MAE {mae:.3f} mm (< {a.mae_max}? {go1}); "
          f"coverage {cover}/{n_total} (>= {a.min_cover}? {go2}); "
          f"Spearman vs pw6 {rho:.3f} on {len(pair)} rows (> {a.rho_min}? {go3})")
    return 0 if res["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
