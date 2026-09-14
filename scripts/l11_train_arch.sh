#!/bin/zsh
# L11 (PLAN v29): ONE architecture swap of the pw6 fascicle recipe (single-variable rung: --arch),
# then the production test pipeline and the benchmark rows for the new checkpoint.
# Recipe = the shipped pw6 run (ledger 2026-09-04T03:10Z: pos_weight 6, dilate 2, 16 epochs, default seed;
# reports/train_fasc_unet_pw6.json). Only the arch varies (unet -> $L11_ARCH; smp names: fpn | segformer | unetpp).
# Launch detached:  L11_ARCH=fpn nohup zsh scripts/l11_train_arch.sh > reports/l11/chain_fpn.log 2>&1 &
# Progress markers: reports/l11/<arch>.train.done / .pred.done / .bench.done ; reports/l11/<arch>.ALL.done
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/reports/l11"; mkdir -p "$OUT"
EPOCHS=16; POSW=6; DILATE=2; THR=0.9
ARCH="${L11_ARCH:-fpn}"
case "$ARCH" in fpn|segformer|unetpp) ;; *) echo "!! unknown L11_ARCH=$ARCH (fpn|segformer|unetpp)"; exit 2 ;; esac
tag="pw6"                                   # trainer names the ckpt fasc_<arch><tag>_best.pt -> no clash with the unet pw6
ckpt="$ROOT/checkpoints/fasc_${ARCH}_${tag}_best.pt"
echo "L11 chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) arch=$ARCH epochs=$EPOCHS pos_weight=$POSW dilate=$DILATE thr=$THR"
fail=0
if [[ ! -f "$OUT/${ARCH}.train.done" ]]; then
  echo "== train $ARCH start $(date -u +%H:%M:%SZ)"
  "$PY" scripts/train_fascicle.py --arch "$ARCH" --epochs $EPOCHS --pos-weight $POSW --dilate $DILATE \
      --tag "_${tag}" > "$OUT/train_${ARCH}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 || ! -f "$ckpt" ]]; then echo "!! train $ARCH FAILED rc=$rc (no $ckpt)"; fail=1; fi
  [[ $fail -eq 0 ]] && { touch "$OUT/${ARCH}.train.done"; echo "== train $ARCH done $(date -u +%H:%M:%SZ)"; }
fi
if [[ $fail -eq 0 && ! -f "$OUT/${ARCH}.pred.done" ]]; then
  echo "== predict $ARCH start $(date -u +%H:%M:%SZ)"
  "$PY" scripts/make_submission.py --aspect 1 --fasc-ckpt "$ckpt" --fasc-thr $THR \
      --scale reports/scale_v3d2.json --flip-positive \
      --out "outputs/sub_L11_${ARCH}_raw.csv" --report "$OUT/pipeline_a1_scalev3d2_flip_${ARCH}_rows.csv" \
      > "$OUT/pred_${ARCH}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 || ! -f "$OUT/pipeline_a1_scalev3d2_flip_${ARCH}_rows.csv" ]]; then echo "!! predict $ARCH FAILED rc=$rc"; fail=1; fi
  [[ $fail -eq 0 ]] && { touch "$OUT/${ARCH}.pred.done"; echo "== predict $ARCH done $(date -u +%H:%M:%SZ)"; }
fi
if [[ $fail -eq 0 && ! -f "$OUT/${ARCH}.bench.done" ]]; then
  echo "== bench $ARCH start $(date -u +%H:%M:%SZ)"
  "$PY" scripts/l10_bench_rows.py --ckpt "$ckpt" --thr $THR --out "$OUT/bench_rows_${ARCH}.csv" \
      > "$OUT/bench_${ARCH}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "!! bench $ARCH FAILED rc=$rc"; fail=1; fi
  [[ $fail -eq 0 ]] && { touch "$OUT/${ARCH}.bench.done"; echo "== bench $ARCH done $(date -u +%H:%M:%SZ)"; }
fi
if [[ $fail -eq 0 ]]; then touch "$OUT/${ARCH}.ALL.done"; echo "L11 $ARCH ALL done $(date -u +%Y-%m-%dT%H:%M:%SZ)"; else echo "L11 $ARCH finished WITH FAILURES $(date -u +%Y-%m-%dT%H:%M:%SZ)"; fi
exit $fail
