#!/bin/zsh
# L6t-b (PLAN v32, human gate 3 / A18): pseudo-label self-training of the pw6 fascicle U-Net on the
# 309 test frames. ONE variable vs the shipped pw6 run: the training set gains the model's own
# pseudo-labelled test frames (scripts/l6t_make_pseudo.py). Recipe otherwise identical to pw6
# (ledger 2026-09-04T03:10Z: unet, pos_weight 6, dilate 2, 16 epochs, seed 0, thr 0.9, scale_v3d2,
# flip-positive). Then the production test pipeline, the benchmark rows and the label-free gate
# (scripts/l11_gate.py --arch l6t: bench FL MAE < 5.35, coverage >= 280, Spearman vs pw6 > 0.2).
# DO NOT launch while campaign.json pilot.blocked_on is A18: this script refuses unless
# L6T_A18_PERMITTED=1 is set by the tick that read `pilot.py unblock` in the ledger.
# Launch detached:  L6T_A18_PERMITTED=1 nohup zsh scripts/l6t_self_train.sh > reports/l6t/chain.log 2>&1 &
# Progress markers: reports/l6t/pseudo.done / train.done / pred.done / bench.done / ALL.done
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/reports/l6t"; mkdir -p "$OUT"
EPOCHS=16; POSW=6; DILATE=2; THR=0.9; SEED=0
ARCH=unet; TAG="_l6t"
BASE_CKPT="$ROOT/checkpoints/fasc_unet_pw6_best.pt"
PSEUDO="$ROOT/data/pseudo/l6t"
ckpt="$ROOT/checkpoints/fasc_${ARCH}${TAG}_best.pt"
if [[ "${L6T_A18_PERMITTED:-0}" != "1" ]]; then
  echo "!! refusing: L6T_A18_PERMITTED is not 1 (human gate 3 / A18 not recorded as permitted)"; exit 3
fi
if [[ ! -f "$BASE_CKPT" ]]; then echo "!! missing base checkpoint $BASE_CKPT"; exit 2; fi
echo "L6t-b chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) arch=$ARCH epochs=$EPOCHS pos_weight=$POSW dilate=$DILATE thr=$THR seed=$SEED"
fail=0
if [[ ! -f "$OUT/pseudo.done" ]]; then
  echo "== pseudo start $(date -u +%H:%M:%SZ)"
  "$PY" scripts/l6t_make_pseudo.py --ckpt "$BASE_CKPT" --arch "$ARCH" --thr $THR --out "$PSEUDO" \
      --summary "$OUT/pseudo_summary.json" > "$OUT/pseudo.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 || ! -d "$PSEUDO/masks" ]]; then echo "!! pseudo FAILED rc=$rc"; fail=1; fi
  [[ $fail -eq 0 ]] && { touch "$OUT/pseudo.done"; echo "== pseudo done $(date -u +%H:%M:%SZ)"; }
fi
if [[ $fail -eq 0 && ! -f "$OUT/train.done" ]]; then
  echo "== train start $(date -u +%H:%M:%SZ)"
  "$PY" scripts/train_fascicle.py --arch "$ARCH" --epochs $EPOCHS --pos-weight $POSW --dilate $DILATE \
      --seed $SEED --tag "$TAG" --pseudo-dir "$PSEUDO" > "$OUT/train.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 || ! -f "$ckpt" ]]; then echo "!! train FAILED rc=$rc (no $ckpt)"; fail=1; fi
  [[ $fail -eq 0 ]] && { touch "$OUT/train.done"; echo "== train done $(date -u +%H:%M:%SZ)"; }
fi
if [[ $fail -eq 0 && ! -f "$OUT/pred.done" ]]; then
  echo "== predict start $(date -u +%H:%M:%SZ)"
  "$PY" scripts/make_submission.py --aspect 1 --fasc-ckpt "$ckpt" --fasc-thr $THR --fasc-arch "$ARCH" \
      --scale reports/scale_v3d2.json --flip-positive \
      --out "outputs/sub_L6t_selftrain_raw.csv" --report "$OUT/pipeline_a1_scalev3d2_flip_l6t_rows.csv" \
      > "$OUT/pred.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 || ! -f "$OUT/pipeline_a1_scalev3d2_flip_l6t_rows.csv" ]]; then echo "!! predict FAILED rc=$rc"; fail=1; fi
  [[ $fail -eq 0 ]] && { touch "$OUT/pred.done"; echo "== predict done $(date -u +%H:%M:%SZ)"; }
fi
if [[ $fail -eq 0 && ! -f "$OUT/bench.done" ]]; then
  echo "== bench start $(date -u +%H:%M:%SZ)"
  "$PY" scripts/l10_bench_rows.py --ckpt "$ckpt" --thr $THR --arch "$ARCH" --out "$OUT/bench_rows_l6t.csv" \
      > "$OUT/bench.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "!! bench FAILED rc=$rc"; fail=1; fi
  [[ $fail -eq 0 ]] && { touch "$OUT/bench.done"; echo "== bench done $(date -u +%H:%M:%SZ)"; }
fi
if [[ $fail -eq 0 ]]; then
  echo "== gate $(date -u +%H:%M:%SZ)"
  "$PY" scripts/l11_gate.py --arch l6t --l11-dir "$OUT" > "$OUT/gate.log" 2>&1; grc=$?
  echo "gate exit $grc (0 GO / 1 NO-GO / 2 inputs missing) -> $OUT/gate_l6t.json"
  case $grc in
    0) verdict=GO ;;
    1) verdict=NO-GO ;;
    *) verdict=ERROR; fail=1 ;;
  esac
  if [[ $fail -eq 0 && -f "$OUT/gate_l6t.json" ]]; then
    # 0 and 1 are both a completed decision; only then is the chain marked ALL.done.
    echo "$verdict rc=$grc" > "$OUT/ALL.done"
    echo "L6t-b ALL done verdict=$verdict $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    [[ ! -f "$OUT/gate_l6t.json" ]] && echo "!! gate wrote no $OUT/gate_l6t.json"
    fail=1; echo "!! gate FAILED rc=$grc (inputs missing or crash); no ALL.done"
  fi
fi
if [[ $fail -ne 0 ]]; then
  echo "L6t-b finished WITH FAILURES $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
exit $fail
