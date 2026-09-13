#!/bin/zsh
# L10 (PLAN v27): 4 further seeds of the pw6 fascicle U-Net recipe, sequential on MPS,
# then the production test pipeline and the benchmark rows for each checkpoint.
# Recipe = the shipped pw6 run (ledger 2026-09-04T03:10Z: pos_weight 6, dilate 2, 16 epochs;
# reports/train_fasc_unet_pw6.json). Only the seed varies (single-variable rung).
# Launch detached:  nohup zsh scripts/l10_train_seeds.sh > reports/l10/chain.log 2>&1 &
# Progress markers: reports/l10/<tag>.train.done / .pred.done / .bench.done ; reports/l10/ALL.done
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/reports/l10"; mkdir -p "$OUT"
EPOCHS=16; POSW=6; DILATE=2; THR=0.9
SEEDS=(${=L10_SEEDS:-1 2 3 4})
echo "L10 chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) seeds=${SEEDS[*]} epochs=$EPOCHS pos_weight=$POSW dilate=$DILATE thr=$THR"
fail=0
for s in "${SEEDS[@]}"; do
  tag="pw6s${s}"
  ckpt="$ROOT/checkpoints/fasc_unet_${tag}_best.pt"
  if [[ ! -f "$OUT/${tag}.train.done" ]]; then
    echo "== train $tag start $(date -u +%H:%M:%SZ)"
    "$PY" scripts/train_fascicle.py --arch unet --epochs $EPOCHS --pos-weight $POSW --dilate $DILATE \
        --seed "$s" --tag "_${tag}" > "$OUT/train_${tag}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 || ! -f "$ckpt" ]]; then echo "!! train $tag FAILED rc=$rc (no $ckpt)"; fail=1; continue; fi
    touch "$OUT/${tag}.train.done"; echo "== train $tag done $(date -u +%H:%M:%SZ)"
  fi
  if [[ ! -f "$OUT/${tag}.pred.done" ]]; then
    echo "== predict $tag start $(date -u +%H:%M:%SZ)"
    "$PY" scripts/make_submission.py --aspect 1 --fasc-ckpt "$ckpt" --fasc-thr $THR \
        --scale reports/scale_v3d2.json --flip-positive \
        --out "outputs/sub_L10_${tag}_raw.csv" --report "$OUT/pipeline_a1_scalev3d2_flip_${tag}_rows.csv" \
        > "$OUT/pred_${tag}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 || ! -f "$OUT/pipeline_a1_scalev3d2_flip_${tag}_rows.csv" ]]; then echo "!! predict $tag FAILED rc=$rc"; fail=1; continue; fi
    touch "$OUT/${tag}.pred.done"; echo "== predict $tag done $(date -u +%H:%M:%SZ)"
  fi
  if [[ ! -f "$OUT/${tag}.bench.done" ]]; then
    echo "== bench $tag start $(date -u +%H:%M:%SZ)"
    "$PY" scripts/l10_bench_rows.py --ckpt "$ckpt" --thr $THR --out "$OUT/bench_rows_${tag}.csv" \
        > "$OUT/bench_${tag}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then echo "!! bench $tag FAILED rc=$rc"; fail=1; continue; fi
    touch "$OUT/${tag}.bench.done"; echo "== bench $tag done $(date -u +%H:%M:%SZ)"
  fi
done
# the shipped pw6 checkpoint's benchmark rows (5th ensemble member), same thr
if [[ ! -f "$OUT/pw6.bench.done" ]]; then
  "$PY" scripts/l10_bench_rows.py --ckpt checkpoints/fasc_unet_pw6_best.pt --thr $THR --out "$OUT/bench_rows_pw6.csv" \
      > "$OUT/bench_pw6.log" 2>&1 && touch "$OUT/pw6.bench.done" || { echo "!! bench pw6 FAILED"; fail=1; }
fi
if [[ $fail -eq 0 ]]; then touch "$OUT/ALL.done"; echo "L10 chain ALL done $(date -u +%Y-%m-%dT%H:%M:%SZ)"; else echo "L10 chain finished WITH FAILURES $(date -u +%Y-%m-%dT%H:%M:%SZ)"; fi
exit $fail
