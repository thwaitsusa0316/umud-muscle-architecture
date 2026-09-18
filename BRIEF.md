> **Target Kaggle:** [https://www.kaggle.com/competitions/ultrasound-muscle-segmentation](https://www.kaggle.com/competitions/ultrasound-muscle-segmentation) *(Click to open or copy URL)*

# UMUD Challenge: Muscle Architecture in Ultrasound Data — BRIEF (P0)

Written 2026-09-03 by pilot (tick 1). Every line carries its source. Re-run P0 only when rules or scope change.

Platform: **Kaggle**, Community prediction competition. Slug: `umud-challenge-muscle-architecture-in-ultrasound-data`.
Host: Paul Ritsche, University of Basel (with Gerardo Romney, Oliver Faude). Sponsor: Swissuniversities, Bern.
Sources read this tick: Rules page and Overview page (logged-in browser, 2026-09-03 20:05 UTC); `kaggle competitions submissions/leaderboard/list` (20:02 UTC); README.md, docs/HYBRID_DESIGN.md, reports/, src/umud/submit.py, git log.

## What pays
| tier | rank | $ | score it currently takes (public) | source |
|---|---|---|---|---|
| 1st | 1 on **private** LB | CHF 1500 | 0.29358 (Patryk Wolny) | Rules §1.5; leaderboard 2026-09-03 |
| 2nd | 2 | CHF 1000 | 0.29974 (PatrickAIForFun) | same |
| 3rd | 3 | CHF 500 | **0.31898** (suguuuuu) — paying cutoff | same |
| next | 4 | nothing | 0.35134 | same |

No Kaggle points or medals ("Does not award Points or Medals" — Overview). Prizes paid in CHF; fees/taxes on recipient.

**Compliance gate on the money** (Overview → Evaluation, "Reproducibility, FAIRness, and Open-Source Eligibility Check"): the provisional private top-3 must supply a public repo that is (a) OSI-licensed, (b) passes the FAIR Software Checklist, (c) reproducible by the organisers from a notebook / script / package / installer with environment spec, (d) documented for inference on the challenge data. A failing team is dropped and the next eligible team moves up. Decision three weeks after the deadline. Winner licence GPL-3.0 (Rules §1.6) — the repo already carries GPL-3.0-or-later.

`what_pays` (campaign.json): rank ≤ 3 on the private leaderboard (≈207 of 309 images) by UMUD score, after the FAIR/open-source check; public cutoff today 0.31898.

Our position: **rank 121 / 198 teams**, best public 1.15267 (`kaggle competitions list -s umud`, userRank 121). 807 entrants, 5,333 submissions total (Overview).

## Metric and split
- Scorer: **UMUD score** = (1/3)·[MAE_PA/6° + MAE_FL/12 mm + MAE_MT/3 mm] + 1e-6·(same with MedAE) + 1e-9·(same with RMSE); lower is better. Host notebook <https://www.kaggle.com/code/paulritsche/umud-score>, mirrored exactly in `src/umud/metric.py`; confirmed by the FL constant sweep (MAE_FL(115) − MAE_FL(80) = 16.46 mm, consistent with a 12 mm tolerance and a median FL ≈ 81.5 mm).
- Public / private: **33 % / 67 %** (~102 / ~207 images) — README (host statement). Test set: 309 images, 251 `.tif` + 58 `.png`.
- Does public track private? **Unknown.** The scorer is deterministic, so the only noise is sampling: benchmark per-image UMUD SD = 0.54 (n = 34, our pipeline) → SE ≈ 0.053 on 102 images, ≈ 0.038 on 207; a 95 % public-vs-private drift for the same method ≈ 0.14. Leader (0.29358) has 252 submissions against 102 public images and sits below the mean-expert noise floor (0.3032), so the top of the public board is probably overfit (README).
- **Noise floor:** resubmission variance is 0 (deterministic scorer; the constant 2.41344 appears identically for several teams). Minimum interpretable public delta between two different methods ≈ **0.10** (2 SE). Because the metric is separable across targets and constant predictions, the score of any constant triple is predictable from the three sweeps already run; submitting the probed-median constant (predicted **1.0374**) is a 1-submission check of that model and banks the true constant floor (rung L1).

## Dates (UTC)
| event | date | notes |
|---|---|---|
| start | ~April 2026 | Overview: "Start 5 months ago" |
| entry / team merge | not shown on the pages read | Rules §2.3: on Overview → Timeline; check next tick |
| final submission | **2026-11-14 23:00:00 UTC** | `kaggle competitions list` deadline = deadline_utc (campaign.json says 23:59; tightened to 23:00) |
| compliance decision | deadline + 3 weeks | Overview |

72 days remain.

## Submission rules
| limit | value | source |
|---|---|---|
| per day | **5** | Rules §2.2.a "You may submit a maximum of five (5) Submissions per day." |
| per week | 35 (5 × 7) | derived |
| total | 5 × days running (only binds team mergers) | Rules §2.1.b |
| final selections | **2** | Rules §2.2.b "You may select up to two (2) Final Submissions for judging." |
| run time (hours) | n/a — CSV upload, no kernel | Overview "submit them through the Kaggle competition interface" |
| GPU hours / week (account) | not used: all compute is local (M3 Max, 64 GB) | README |
| team size | 10 | Rules §2.1.a |
| internet / external data | allowed if public, equally accessible, minimal cost, and documented in the final method description; pretrained models allowed | Rules §2.6; Overview "Training Constraints" |
| hand labelling of test data | **forbidden** | Foundational Rule 4.b "Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records." |
| accounts | one only | Foundational 5.a "You may make Submissions only under one, unique Kaggle.com account." |
| **automated submission** | **silent** | No clause addresses scripted or API submission. Closest text: Foundational 5.a "You may submit up to the maximum number of Submissions per day as specified on the Competition Website."; Overview "Participants generate predictions for the test dataset and submit them through the Kaggle competition interface. Multiple submissions are allowed during the competition period." → `auto_allowed` = false pending Stephen (pilot rule: silence parks). Note the 13 existing submissions were all made through the official Kaggle API (`src/umud/submit.py`, KaggleApi.competition_submit). |

Mechanism (read): `.venv/bin/kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data`; `... leaderboard --show`.
Mechanism (write, when authorised): `.venv/bin/kaggle competitions submit -c umud-challenge-muscle-architecture-in-ultrasound-data -f <csv> -m "<msg>"`.

### Submission-format traps (encoded in `src/umud/submit.py`, both confirmed by a rejected submission)
1. The host's `sample_submission.csv` is **semicolon-delimited with a BOM**; the grader wants **plain comma-delimited UTF-8**. The semicolon file fails with `ID column image_id not found in submission` (submission 55920456, ERROR, 2026-08-31).
2. `image_id` must carry the **true per-file extension** — 251 `.tif` + 58 `.png`; assuming `.tif` throughout is rejected.
3. Columns exactly `image_id,pa_deg,fl_mm,mt_mm`, 309 rows, no NaNs; `write_submission()` asserts all of this.

## Ground truth already banked (13 submissions, 12 scored)
| file | public | what it taught |
|---|---|---|
| sub_00_const_midpoint.csv (PA25/FL115/MT30) | **2.41344** | format valid; same value as other teams' constants |
| FL sweep 60/80/100 | 2.15750 / 1.95614 / 2.10414 | public median FL ≈ 81.5 mm |
| MT sweep 15/20/25 | 1.61484 / 1.33274 / 1.48875 | public median MT ≈ 20.7 mm |
| PA sweep 20/15/10 (last with MT 20) | 1.72313 / 1.66079 / 1.26409 | public median PA ≈ 16.4° |
| sub_pipeline_a1.35.csv (full pipeline) | 1.46532 | **worse than a constant** — transfer failure |
| diag_measuredPA_constFLMT.csv | 1.15267 (our best) | measured PA alone costs 0.115 vs the median |

`reports/probe_log.json` shows null scores only because it was never back-filled; the Kaggle submissions list is authoritative.

## Resources and baselines
- Hardware: Apple M3 Max, 64 GB, PyTorch/MPS; no hosted model anywhere in the pipeline (FAIR re-run requirement).
- Data: 2.75 GB archive; 2,761 fascicle image/mask pairs for training; test 309 images from Siemens Acuson Juniper, Telemed ArtUS, Philips Lumify; 27 runs of exactly 5 video frames (135 images).
- External: OSF expert benchmark (35 images, 7 raters, ground-truth px/cm) at `data/external/`; DL_Track_US pretrained U-Nets (Apache-2.0).
- Public notebooks/models: **none at 2026-09-03**; stale by 2026-09-16 — 28 public notebooks, incl. `dreaddevelopment/vera-seg-centerline-mt-correction-lb-0-45134` (2026-09-02, a hard-coded 309-row CSV that scores exactly 0.45134; 21 teams sit on it by 09-16, copy at `reference/kernels/vera_lb045134.csv`), `phuongncn/lb-0-76704-no-train-anatomy-calibrated-dltrack`, and `lamhuy8904/variational-ultrasound-kinematics-ensemble` (16 votes, GPU). The 0.45134 cluster is a shared file, not necessarily stock DL_Track.
- Local benchmark: our geometry 0.3349, DL_Track 0.3306, mean expert 0.3032, best expert 0.2459. On the test set the same pipeline scores 1.465 — the benchmark cannot see the console-screenshot domain shift.

## Eligibility / legal
- Standard Kaggle eligibility; US resident fine; W-9 for prize. Competition Data CC BY-NC-SA, non-commercial use only (Rules §1.7, §2.4).
- OSF benchmark: CC BY 4.0 node bundling a "DL_Track Data Usage Policy" (non-commercial); host asked in discussion 737782, unanswered. Must be declared as external data in the final method description.
- Winner obligations: full training + inference code, environment description, FAIR checklist, reproducible run (Rules §2.8; Overview).

## Feasibility estimate
Paying position = rank 3 = public ≈ 0.319 today (private unknown; realistic honest ceiling ≈ 0.30–0.35 given the expert noise floor).
Best banked = 1.153 (diagnostic); best real pipeline on test = 1.465; best constant (predicted) = 1.037. Gap ≈ 0.7–0.85 UMUD.
Structural diagnosis (docs/HYBRID_DESIGN.md): the fascicle detector puts two thirds of its pixels outside the muscle on test images; pair selection is not the binding constraint; the VLM stage 3 is a measured negative and not shipped.
The gap must be closed to at least **public ≤ 0.45134 (stock-DL_Track level) by 2026-10-12** (→ `feasibility_gate.by_date`) or the money objective is declared unreachable and the plan drops to "publishable open benchmark method / reallocate quota to a sibling engagement" by a numbered revision.
