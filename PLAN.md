# UMUD Challenge — PLAN v1

Supersedes: none. Rests on BRIEF.md (2026-09-03). Revisions bump the version and say why.

## The campaign reduces to one number
**Private-leaderboard UMUD score ≤ ~0.32 (today's public #3 = 0.31898) at 2026-11-14 23:00 UTC**, with a FAIR-compliant public GPL-3.0 repo behind it.
Today's best banked: 1.15267 (a diagnostic, measured PA only). Best real pipeline: 1.46532. Predicted best constant: 1.0374. Gap ≈ 0.7–0.85 UMUD, of which the expert noise floor says ~0.30 is the honest ceiling.
The gap is not a measurement problem: the pipeline scores 0.3349 on native-resolution benchmark images and fails on the test domain because the fascicle detector puts two thirds of its pixels outside the muscle (docs/HYBRID_DESIGN.md). Every rung below attacks that transfer failure or measures it.

## Day 1 — bank the floor (rung L0) — ALREADY DONE
`outputs/sub_00_const_midpoint.csv` (PA 25 / FL 115 / MT 30) was submitted 2026-08-31 (Kaggle submission 55920465) and scored **2.41344**; the semicolon/BOM variant (55920456) ERRORed, which is what validated the format. The submit-and-read-back loop has since closed 12 times. Re-submitting the identical file would burn a slot on a deterministic scorer for zero information, so L0 is recorded as banked, not queued.

## The ladder
| rung | single variable | expected gain | cost (GPU h / subs / days) | falsifier (pre-registered) | status |
|---|---|---|---|---|---|
| L0 | constant midpoint (format + first score) | banks 2.41344 | 0 / 1 / 0 | n/a | **done 2026-08-31** |
| L1 | probed-median constant PA 16.42 / FL 81.5 / MT 20.7 (`reports/public_medians.json`) — checks metric separability and banks the true constant floor | predicted **1.0374** (→ from 1.153, rank ↑) | 0 / 1 / 0 | if \|score − 1.0374\| > 0.01, the separable-constant model or a probed median is wrong → re-derive from sweeps before any pipeline is judged against it | queued (needs the CSV written and a candidate file; parked until auto-submit is authorised) |
| L2 | fascicle U-Net retrained on the 2,761 competition pairs (`scripts/train_fascicle.py`), ranked by held-out Dice AND label-free in-muscle coverage on test | coverage 0.33 → ≥ 0.6; the prerequisite for everything above | ~6 local GPU h / 0 / 3 | if median in-muscle coverage on the 20 gated test images < 0.50 (vs 0.33 now) or benchmark UMUD regresses > 0.02, the retrain is dead at this architecture → try SegFormer / nnU-Net before abandoning | next |
| L3 | full pipeline with the L2 detector, sector-cropped, aspect 1.0, probed-median fallback | first real score below the constant floor | 0 / 1 / 1 | if public > 1.0374 − 0.10 = 0.937 the pipeline is still worse than a constant → back to L2 diagnostics (checkpoint 2026-09-24) | queued |
| L4 | isolation: measured MT only, PA/FL at probed medians (MT is 44 % of the error budget and needs no fascicles) | −0.10 to −0.25 vs constant | 0 / 1 / 0 | if public > 1.0374 − 0.10, aponeurosis pair selection is wrong on test → aponeurosis retrain / sector crop rung inserted | queued |
| L5 | per-run consensus over the 27 five-frame video runs (scale + targets pooled) | −0.01 to −0.03; variance | 0 / 1 / 1 | paired sub (same pipeline ± consensus): if delta ≥ 0 the rung is dead (no sampling noise between paired subs) | queued |
| L6 | scale as a lookup over device configurations (image shape → px/cm), validated on the 35 benchmark GT scales | fixes the 160 cropped frames | 0.5 / 0 / 2 | if lookup accuracy on benchmark < 90 % of images within 3 %, dead | queued |
| L7 | port DL_Track's fascicle-length routine onto our PA/MT (benchmark says 0.3148 if FL matches DL_Track) | −0.05 on benchmark | 2 / 1 / 3 | benchmark FL MAE not < 5.0 mm → dead | queued |
| L8 | SAM2 prompted aponeurosis masks + video propagation | replaces CNN pair step | 4 / 1 / 4 | benchmark UMUD not better than the CNN route by ≥ 0.01 and no coverage gain on test → dead | queued |
| L9 | final-selection strategy: 2 finals = best public + most conservative honest pipeline | protects private | 0 / 0 / 1 | n/a — human gate | queued (P4) |

Killed before v1: VLM band identification (stage 3, net +1/20), VLM direct measurement (1.42–1.77), anisotropy 1.35, per-target calibration gains, manual test annotation (Foundational 4.b).

## Assumption register (mirrored in campaign.json `pilot.assumptions`)
| id | claim | evidence | verify every | last verified | status |
|---|---|---|---|---|---|
| A1 | retrained fascicle net closes the transfer gap | untested; coverage diagnosis | 7 d | — | unverified |
| A2 | local CV (35-image benchmark) tracks LB | 0.3349 bench vs 1.465 LB | 14 d | 2026-09-03 | **broken** — benchmark valid for geometry conventions only |
| A3 | submission encoding correct | 12 COMPLETE, 1 ERROR (semicolon) | 30 d | 2026-09-03 | verified |
| A4 | public tracks private | unmeasured; est. drift ~0.14 | 30 d | — | unverified |
| A5 | metric mirror exact | host notebook + FL-sweep arithmetic | 30 d | 2026-09-03 | verified |
| A6 | OSF benchmark is permissible external data | licence conflict, discussion 737782 open | 14 d | — | unverified |
| A7 | public top is overfit; honest 0.30–0.35 places | leader below expert floor | 30 d | — | unverified |
| A8 | Kaggle CLI submission permitted | rules silent; 13 accepted | 30 d | — | unverified — Stephen |
| A9 | VLM band identification beats geometry | +1/20 | — | 2026-09-03 | dead |
| A10 | scale recoverable for 160 cropped frames | OCR 0/35; shape→scale lookup untested | 14 d | — | unverified |

## Submission calendar
5/day, 2 finals. Used 13 so far (2 today). Policy: at most 2 submissions per day are spent on the ladder (one rung reading + one isolation/paired control); 3 held for a same-day fix or a probe. No submission without a `submit_queue/<id>.json` justification written against the current plan version and a `maibo_test` record for the code that produced it. Constant/probe CSVs need no model run and are the cheapest measurements available; a pipeline submission is only worth a slot once L2's coverage gate passes.

## Human gates
1. **Auto-submit authorisation** (blocked now): the rules are silent on scripted submission; the mechanism is Kaggle's official CLI. Until Stephen says yes, candidates park in `submit_queue/` and are pushed.
2. External-data declaration: the OSF benchmark licence conflict (A6) must be resolved or the benchmark confined to development-only use before the final method description.
3. Final selection of the 2 submissions (P4) and the FAIR repo release.
4. Any change to `what_pays` (e.g. dropping from prize to publishable-method tier at the feasibility gate).

## Feasibility / kill gate
metric: best public UMUD of a real (non-constant) pipeline submission · threshold: **≤ 0.45134** (stock DL_Track cluster) · by **2026-10-12** · consequence: money objective declared unreachable, numbered revision + push, drop to publishable-method tier and recommend reallocating quota. Interim checkpoint 2026-09-24: a pipeline submission beats the constant floor 1.0374 by ≥ 0.10 (rung L3 falsifier).

## Review ledger
maibo method review not yet run (tick 1 was P0/P1 reading only; no code touched). Queued for tick 2: `maibo panel` on this plan's measurement design — specifically whether L2's coverage gate (label-free, on test images) is a sufficient proxy before a submission is spent.
