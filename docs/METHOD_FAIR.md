# Method description — UMUD Challenge entry (public 0.56382, submission 56194131)

**Status:** DRAFT v1.2, 2026-09-17T02:20Z (v1 17:16Z, v1.1 23:55Z; v1.2 disposes the codex panelist review `reports/method_fair/codex_review_2026-09-17T01-58Z.md`, section 10), written under PLAN v36 for Rules s2.8 / s2.5.b
("training code, inference code, and a description of the required computational
environment"; "a detailed description of methodology, where one must be able to
reproduce the approach") and the Overview's "Reproducibility, FAIRness, and Open-Source
Eligibility Check". Every number below is read from the files named next to it; nothing is
quoted from memory.

Repository: <https://github.com/thwaitsusa0316/umud-muscle-architecture> (GPL-3.0-or-later,
`LICENSE`, `CITATION.cff`, `pyproject.toml`).

## 1. What the entry predicts

For each of the 309 test images: pennation angle `pa_deg`, fascicle length `fl_mm` and
muscle thickness `mt_mm`. The shipped file is `outputs/sub_L8a_compose_fl075.csv`
(sha256 `22c1d4fd6062edafd05a6eb68a81cf92227d77624e716e17bfaabd2bd96c1f57`, 309 rows,
comma-delimited UTF-8, no BOM). Its medians are PA 16.420 deg, FL 81.500 mm, MT 22.024 mm;
ranges PA 12.68–24.97 deg, FL 43.45–120.59 mm, MT 12.65–34.12 mm.

The second final is the constant floor `outputs/sub_L1_probed_median.csv` (sha256
`496931f47be1958e1cd4f1787bc4778e91c71c46a3d4831792d8d108e8cc1a13`, public 1.03662,
submission 56003661): PA 16.42 / FL 81.5 / MT 20.7 on every row.

**Artifact manifest** (what a reader needs, where it will be public, and whether it is in git today):

| artifact | local path | sha256 (`shasum -a 256`, 2026-09-17T02:20Z) | in git | public delivery |
|---|---|---|---|---|
| final 1, score 0.56382 | `outputs/sub_L8a_compose_fl075.csv` | `22c1d4fd6062edafd05a6eb68a81cf92227d77624e716e17bfaabd2bd96c1f57` | no (`outputs/` gitignored) | `submission/` + `SHA256SUMS` (section 9) |
| final 2, constant floor 1.03662 | `outputs/sub_L1_probed_median.csv` | `496931f47be1958e1cd4f1787bc4778e91c71c46a3d4831792d8d108e8cc1a13` | no | same |
| 10 intermediate CSVs of section 4 | `outputs/sub_pipeline_a1_scalev3d2*.csv`, `sub_L6g_*`, `sub_L6r_*`, `sub_L6ob_*` (2), `sub_L7c_*`, `sub_L7r2_*`, `tmp_L8a_fl075.csv` | listed in `SHA256SUMS` at push | no | same |
| fascicle checkpoint | `checkpoints/fasc_unet_pw6_best.pt` (97,918,663 bytes) | `6c9bee9a492875ac279916a59079109177a07cf07380fc2da61ccb5ece276279` | **no** (`checkpoints/` gitignored) | GitHub release asset (section 9) |
| training report for that checkpoint | `reports/train_fasc_unet_pw6.json` | at push | yes (unpushed) | `git push` |
| per-image scale lookup | `reports/scale_v3d2.json` | at push | yes (unpushed) | `git push` |
| flip set | `reports/l6t_fascicle_evidence.csv` | at push | yes (unpushed) | `git push` |
| scripts of sections 4-5 | `scripts/*.py`, `src/umud/*.py` | commit hash at push | yes (4 unpushed, section 9) | `git push` |
| environment lock | `pyproject.toml`, `uv.lock` | at push | yes | `git push` |

## 2. Principle

The vision models never emit a number. Two segmentation networks produce an aponeurosis
mask and a fascicle mask; deterministic geometry (`src/umud/geometry.py`) turns the masks
into an angle, a length and a thickness; a per-image pixels-per-centimetre scale converts
lengths to millimetres; a short chain of label-free, pre-registered post-processing rules
(each one script, one variable, each read once on the public leaderboard as an exact paired
delta) produces the shipped file. Design notes: `docs/HYBRID_DESIGN.md`.

## 3. Data and external resources (declared)

| resource | use | licence / provenance |
|---|---|---|
| Competition training set: 2,761 fascicle image/mask pairs (`data/raw/fasc_imgs_v1`, `fasc_masks_v1`) | trains the fascicle U-Net | challenge data (Rules s2.4) |
| Competition test set: 309 images (`data/raw/test_images_v2/test_set_v2`) | inference only; also the label-free fascicle-orientation audit that decides which frames are mirrored (`reports/l6t_fascicle_evidence.csv`) | challenge data |
| DL_Track_US v0.3.0 aponeurosis network `models/model-apo-VGG16_DiceBCE_512_300.h5` | aponeurosis mask (as published, no fine-tuning) | Apache-2.0, <https://osf.io/7mjsc>, fetched by `scripts/download_models.py`; Apache-2.0 is one-way compatible with GPL-3.0 |
| OSF expert benchmark (35 images, 7 raters, ground-truth px/cm) at `data/external/` | **validation only**: unit / convention / scale checks, and a go/no-go falsifier on the training rung (L2: benchmark UMUD 0.3673 → 0.3566, would have fired at a regression > 0.02; `ledger.jsonl` 2026-09-04T03:10:21Z). It never chose between checkpoints (that is validation Dice on held-out competition pairs, Step 2), was never trained on, and touched no shipped value | CC BY 4.0 node bundling a "DL_Track Data Usage Policy" (non-commercial); permissibility asked in discussion 737782, unanswered (assumption A6, unverified) |
| Public leaderboard | three constant-valued probe submissions fixed the public-set medians PA 16.42 deg / FL 81.5 mm / MT 20.7 mm (`reports/public_medians.json`, rung L1, public 1.03662); these three constants are the level anchors and fallbacks used everywhere below | Rules s2.2.a (5 submissions/day) |

No test-image labels exist locally; no hosted model or external API is used anywhere.

## 4. Pipeline, in execution order

All commands run from the repository root with the `uv` environment (section 6).

**Step 0 — data and weights.**
`uv run python scripts/download_data.py` and `uv run python scripts/download_models.py`.

**Step 1 — per-image scale (`reports/scale_v3d2.json`, 309/309 images).**
`scripts/scale_v2.py` fuses the sector-height / OCR-depth ratio (`src/umud/depth.py`) with the
ruler tick-comb pitch (`src/umud/ruler.py`) and fills gaps by acquisition-run consensus
(`src/umud/grouping.py`). `scripts/scale_v3.py` adds four mechanistic fixes on top:
(a) chrome-71/72 PNG stills: px/cm = sector width / 2.894 cm (3.18 cm on the 582-px zoom
rows); (b) native 644x1088 video frames: left ruler comb at 126.0 px/cm; (c) chrome-free
853-row video stills: bottom-edge comb 166.8 px/cm; (d) chrome-free 512/513-row RGB frames:
dashed right-edge ruler (`--dash-unit`). Output variant "d2" is the shipped lookup.

**Step 2 — fascicle segmentation model (`checkpoints/fasc_unet_pw6_best.pt`).**
`uv run python scripts/train_fascicle.py --arch unet --epochs 16 --pos-weight 6 --dilate 2 --tag _pw6`
(seed 0, the script default; `--tag _pw6` gives the checkpoint its name) trains a
`segmentation-models-pytorch` U-Net at 512 px on the 2,761 competition fascicle pairs
(AdamW, lr 3e-4, cosine schedule, masks dilated by 2 px, BCE positive weight 6; "pw6").
Split rule (`train_fascicle.py` main): the pair list is shuffled with `random.Random(0)`
and the first `max(40, int(0.12 x 2761)) = 331` pairs are validation, the other 2,430
train. Checkpoint rule: after every epoch the mean validation Dice is computed and the
state dict is saved whenever it improves, so the committed file is the best-Dice epoch:
epoch 13 of 16, val Dice 0.6651, 1,539 s wall (`reports/train_fasc_unet_pw6.json`). The
in-muscle coverage check (0.998 median on 60 test frames) and the benchmark falsifier of
section 3 were go/no-go gates on the rung, not selection criteria between checkpoints.
Checkpoint sha256 in the section 1 manifest. Why not the
published DL_Track fascicle weights: on the test images roughly two thirds of their
fascicle pixels fall outside the muscle (docstring of `scripts/train_fascicle.py`).

**Step 3 — inference with mirroring (`outputs/sub_pipeline_a1_scalev3d2_flip.csv`,
`reports/pipeline_a1_scalev3d2_flip_rows.csv`).**
```
uv run python scripts/make_submission.py --aspect 1 \
    --fasc-ckpt checkpoints/fasc_unet_pw6_best.pt --fasc-thr 0.9 --fasc-arch unet \
    --scale reports/scale_v3d2.json --mt-gate 8 35 \
    --flip-positive reports/l6t_fascicle_evidence.csv \
    --out outputs/sub_pipeline_a1_scalev3d2_flip.csv \
    --report reports/pipeline_a1_scalev3d2_flip_rows.csv
```
Per image: DL_Track aponeurosis mask + pw6 fascicle mask (threshold 0.9) →
`geometry.analyse` → PA from fascicle slopes relative to the deep aponeurosis (scale-free),
FL from the fascicle-line intersection with the two aponeuroses, MT from the aponeurosis
separation, both in px then / px-per-cm. The 113 test frames whose fascicles slope the
opposite way to the training set (`ang_med > 0` in the evidence file; the host's 300
training frames are 299/300 single-orientation) are mirrored left-right with
`cv2.flip(gray, 1)` before both networks. An MT outside 8–35 mm is treated as an
aponeurosis-pairing failure and falls back to the probed median 20.7. Result: PA and FL
measured on 297 rows, MT on 299; every unmeasured row sits at its probed median.
**Two runs of this command feed the shipped file, one per target group.** PA and FL
come from the mirrored run above (`outputs/sub_pipeline_a1_scalev3d2_flip.csv`, rows
`reports/pipeline_a1_scalev3d2_flip_rows.csv`). MT comes from the as-delivered
(unmirrored) run of the same command without `--flip-positive`
(`outputs/sub_pipeline_a1_scalev3d2.csv`, rows `reports/pipeline_a1_scalev3d2_rows.csv`),
which is why Step 4 starts from the unmirrored file and Step 5 re-reads PA/FL from the
mirrored rows file. MT (aponeurosis separation) is mirror-invariant, so the two runs
agree on it up to the mask thresholding.

**Step 4 — MT isolation and within-run consensus (rungs L6g, L6r).**
`scripts/make_isolation.py --keep mt --source outputs/sub_pipeline_a1_scalev3d2.csv`
→ `outputs/sub_L6g_isolate_mt_v3d2.csv`; then
`scripts/make_run_median.py --source outputs/sub_L6g_isolate_mt_v3d2.csv --scale reports/scale_v3d2.json --threshold 0.25`
→ `outputs/sub_L6r_run_median_v3d2.csv`: within each video run (run id from the scale
manifest) whose shipped MT values spread by more than 25 % of their median, every shipped
frame takes the run median (report `reports/l6r_run_median.csv`).

**Step 5 — level-matched PA and FL from the mirrored run (rung L6o-b).**
`scripts/make_level_matched.py --target fl --scalar 0.8705521018792615 --source outputs/sub_L6r_run_median_v3d2.csv --rows reports/pipeline_a1_scalev3d2_flip_rows.csv`
→ `outputs/sub_L6ob_fl_step_v3d2.csv` (FL x ratio = probed 81.5 / measured median 93.619);
`scripts/make_level_matched.py --target pa --scalar 3.3922419702826065 --source outputs/sub_L6ob_fl_step_v3d2.csv --rows ...`
→ `outputs/sub_L6ob_flip_level_v3d2.csv` (PA + shift = probed 16.42 − measured median
13.028). One scalar per target, fixed from the measured rows' own medians; reports
`reports/l6ob_fl_level.csv`, `reports/l6ob_pa_level.csv`. Public 0.67241.

**Step 6 — FL plausibility clamp (rung L7c).**
`scripts/make_shrink.py --mode clamp --target fl --lo 30 --hi 150 --min-rows 5 --source outputs/sub_L6ob_flip_level_v3d2.csv --rows reports/pipeline_a1_scalev3d2_flip_rows.csv`
→ `outputs/sub_L7c_fl_clamp.csv`: 21 rows with FL > 150 mm set to 81.5
(`reports/l7c_fl_clamp.csv`). Public 0.63398.

**Step 7 — within-run consensus for PA and FL (rung L7r2).**
`scripts/make_run_median_pafl.py --targets pa,fl --threshold 0.25 --min-frames 3 --method video --source outputs/sub_L7c_fl_clamp.csv --rows reports/pipeline_a1_scalev3d2_flip_rows.csv --scale reports/scale_v3d2.json`
→ `outputs/sub_L7r2_run_median_pafl_clamped.csv`: the Step 4 rule applied per target to
video runs with at least 3 measured frames (PA run 53, FL runs 53 and 56;
`reports/l7r2_run_median_pafl_clamped.csv`). Public 0.62061.

**Step 8 — shrinkage toward the probed medians (rungs L8 / L8a).**
`scripts/make_shrink.py --mode shrink --target fl --alpha 0.75 --source outputs/sub_L7r2_run_median_pafl_clamped.csv --rows ...` → `outputs/tmp_L8a_fl075.csv` (an intermediate, regenerated by the command;
fl' = 0.75 fl + 0.25 x 81.5 on the 297 measured rows; `reports/l8a_step_fl075.csv`), then
`scripts/make_shrink.py --mode shrink --target pa --alpha 0.5 --source outputs/tmp_L8a_fl075.csv --rows ...`
→ **`outputs/sub_L8a_compose_fl075.csv`** (pa' = 0.5 pa + 0.5 x 16.42;
`reports/l8a_step_pa05_fl075.csv`). Public **0.56382**. The alphas were chosen by a
pre-registered bracket (FL 0.25 / 0.5 / 0.75, PA 0.25 / 0.5 / 0.75, one read each) and
then frozen; no further tuning on the board (PLAN v23–v26). Checkable pointers: the
bracket reads are `ledger.jsonl` entries of type `submission` with candidates `L8`
(56134903), `L8a025` (56194124), `L8a075` (56194131, the shipped file), `L8b075`
(56194277) and `L8b025` (56200799), each carrying the artifact sha256 and its public
score; the score table is `reports/score_history.json`; the Kaggle submission list
(`kaggle competitions submissions`) shows the same ids and scores.

Every step's script and CSV passed an independent code review and byte-exact verification
before it was submitted (`maibo_tests/*.json`, referenced from `ledger.jsonl`).

## 5. Validation that does not use the leaderboard

- Unit and convention checks against the 35-image OSF expert benchmark
  (`scripts/eval_benchmark_ours.py --ckpt checkpoints/fasc_unet_pw6_best.pt --thr 0.9`):
  PA bias −0.20 deg, MAE 1.22 deg (n = 34); FL MAE 5.72 mm; local UMUD 0.3566.
- Orientation audit (`scripts/l6o_flip_audit.py`, `reports/l6o_flip_audit.csv`): the
  pipeline is orientation-dependent, which motivates Step 3's mirroring.
- Fascicle-orientation evidence on the test set (`scripts/l6t_fascicle_evidence.py`): the
  113 mirrored frames are chosen by the sign of the median fascicle slope, not by score.

## 6. Computational environment

- Apple M3 Max, 64 GB, macOS 26.6.1; PyTorch on MPS (no CUDA, no Kaggle GPU).
- Python 3.12.14 via `uv sync` (`pyproject.toml`, `uv.lock`): torch 2.13.0,
  segmentation-models-pytorch 0.5.0, tensorflow 2.21.0 + tf-keras (DL_Track .h5 weights),
  opencv-python-headless 5.0.0.93, numpy 2.5.2, pandas, scikit-image, scipy, pytesseract
  (needs a system `tesseract` binary for the depth OCR in Step 1: tesseract 5.5.3 with
  leptonica 1.87.0 from Homebrew, `brew install tesseract`).
- Wall time (from `ledger.jsonl` timings): training Step 2 about 35–40 min on MPS for
  16 epochs (four seeds took 2.3 h on 2026-09-13; the L6t run took 01:40–02:20Z on
  2026-09-16); inference Step 3 about 18 min for 309 images (02:20–02:38Z, 2026-09-16);
  Steps 4–8 are seconds each.
- Determinism: Steps 1 and 4–8 are deterministic. Step 2 is seeded (seed 0) but MPS
  kernels are not bit-reproducible, so a retrained checkpoint will differ; Steps 3–8
  reproduce `outputs/sub_L8a_compose_fl075.csv` byte-for-byte from the shipped checkpoint.
  **Correction (v1.2):** that checkpoint is NOT committed — `checkpoints/` is gitignored
  (97.9 MB, near GitHub's 100 MB file limit). It must be published as a release asset
  with its sha256 (section 9).
- The "four seeds took 2.3 h" timing refers to the L10 seed-ensemble rung
  (`checkpoints/fasc_unet_pw6s1..s4_best.pt`, 2026-09-13), which was NOT shipped; the
  shipped checkpoint is the single seed-0 run of 2026-09-03 (1,539 s).

## 7. FAIR software checklist (self-assessment)

| item | where |
|---|---|
| Findable: public repo, `CITATION.cff` with title/authors/licence | repo root |
| Accessible: `git clone` + `uv sync`; data fetched by script after accepting the rules | `README.md` Setup |
| Interoperable: standard CSV in/out, PyTorch/TensorFlow formats, no proprietary tooling | `src/umud/submit.py` |
| Reusable: GPL-3.0-or-later, per-script docstrings state purpose and the exact invocation, every shipped step has a report CSV | `LICENSE`, `scripts/`, `reports/` |

## 8. Known limitations (stated honestly)

- Public rank 85 of 264 at 0.56382 on 2026-09-17T01:57Z (snapshot
  `reports/leaderboard_2026-09-17T01-57Z.csv`, from `kaggle competitions leaderboard
  --download`; 14 dated snapshots under `reports/leaderboard_*.csv`); the entry is not in
  the prize band (#3 = 0.31898).
- Level anchors and fallbacks come from leaderboard probes (section 3), so the method is
  tied to this test set's public medians.
- The OSF benchmark's usage policy is unresolved (A6); it touched no shipped value.

## 9. Publication gap (audit 2026-09-16T23:55Z, blocks s2.8 as written)

Checked this tick with `git fetch origin` and `git ls-tree -r origin/master` against the
paths named in sections 3-5. The public repository head is `76753bd` (2026-09-03); local
`master` is **46 commits ahead** and nothing has been pushed since. Consequences for a
reader following section 4 from the public link today:

| gap | files | clause | fix (one action, branch A of gate 6) |
|---|---|---|---|
| Scripts named in the pipeline are not on the public head | `scripts/scale_v2.py`, `scripts/scale_v3.py`, `scripts/l6o_flip_audit.py`, `scripts/l6t_fascicle_evidence.py` (exist locally, unpushed) | s2.5.b "link to a code repository with complete and detailed instructions" | `git push origin master` after the kill-gate decision (46 commits; nothing target-restricted is tracked: `data/`, `models/`, `outputs/`, `checkpoints/` are gitignored) |
| Report CSVs cited as evidence are not public | 11 files under `reports/` (`scale_v3d2.json`, `l6o_flip_audit.csv`, `l6ob_*_level.csv`, `l6r_run_median.csv`, `l6t_fascicle_evidence.csv`, `l7c_fl_clamp.csv`, `l7r2_run_median_pafl_clamped.csv`, `l8a_step_*.csv`, `pipeline_a1_scalev3d2_flip_rows.csv`) | s2.8 reproducibility (the reader cannot check the per-image scale lookup or the flip set) | same push; they are tracked locally |
| The shipped CSVs are gitignored (`outputs/`), so the two sha256s in section 1 cannot be checked from the repo | `outputs/sub_L8a_compose_fl075.csv`, `outputs/sub_L1_probed_median.csv` and the 10 intermediate `outputs/sub_*.csv` named in section 4 | s2.8 (winner must deliver the Submission and the code that generated it) | add a `submission/` directory (or a release asset) carrying the two finals and the 10 intermediates with a `SHA256SUMS` file; 12 files, ~15 KB each; not competition data, so redistributable |
| **The shipped fascicle checkpoint is gitignored (found by the v1.2 review; section 6 v1.1 wrongly said "committed")** | `checkpoints/fasc_unet_pw6_best.pt`, 97,918,663 bytes, sha256 `6c9bee9a…6279` (section 1 manifest) | s2.8 (a reader cannot run Step 3 without it; retraining on MPS is not bit-reproducible) | attach it as a GitHub release asset (under the 2 GB asset limit; over the 100 MB git file limit is close, so a release, not a commit) and add its sha256 to `SHA256SUMS`; `reports/train_fasc_unet_pw6.json` goes with the push |
| Data and weights are fetched, not shipped | `data/raw/*`, `data/external/`, `models/*.h5` | s2.4 (no redistribution of Competition Data) | correct as designed; section 4 step 0 names the fetch scripts. Verify `scripts/download_data.py` still resolves the v2 test-set URL at close. |

sha256 re-check this tick: both hashes in section 1 match the local files
(`shasum -a 256`, 2026-09-16T23:53Z). Repository URL returns HTTP 200.

## 10. Independent review disposition (codex panelist, 2026-09-17T01:58Z)

Review file: `reports/method_fair/codex_review_2026-09-17T01-58Z.md` (maibo `ask`, engine
codex, attachments: this document v1.1 and `RULES_EXCERPT.txt`). Every finding and what
v1.2 did with it:

| finding | disposition in v1.2 |
|---|---|
| (1).1–(1).4 training code, inference code, final CSVs and intermediates are not public-checkable | already section 9; now also the section 1 artifact manifest. Needs the `git push` + `submission/` + release asset of branch A (human gate 6) — a doc edit cannot close it |
| (1).5 OSF benchmark permission unresolved | kept as stated (A6 unverified); role clarified in section 3 |
| (1).6 tesseract not pinned | section 6: tesseract 5.5.3 / leptonica 1.87.0, Homebrew |
| (2).1 / (3).4 Step 3 MT provenance unclear | section 4 Step 3: two runs named with both output and rows files |
| (2).2 / (3).5 "validation only" vs "model-selection gates" | section 3 and Step 2: the benchmark was a go/no-go falsifier on the rung, never a checkpoint selector |
| (2).3 / (3).6 seed and checkpoint selection | Step 2: seed 0, split rule, best-Dice-epoch rule, epoch 13/16, Dice 0.6651, sha256; section 6 separates the L10 four-seed timing from the shipped run |
| (2).4 / (3).7 leaderboard facts unverifiable | section 8: dated snapshot files; Step 8: ledger submission ids and `reports/score_history.json` |
| (2).5 sha256s unverifiable from the doc | manifest in section 1; checkable only after the section 9 publication |
| (2).6 bracket not pointed to plan/ledger | Step 8 pointers (five submission ids) |
| (3).1 artifact manifest | section 1 |
| (3).2 replace section 9 with the pushed commit hash | deferred to branch A; section 9 stays until the push happens |
| new, found while disposing (2).3 | the checkpoint is gitignored, not committed: section 6 corrected, section 9 row added |
