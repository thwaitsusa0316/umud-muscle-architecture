# UMUD Challenge — PLAN v2

Supersedes: v1 (2026-09-03). Rests on BRIEF.md (2026-09-03). Revisions bump the version and say why.

**Why v2 (2026-09-04T03:10Z):** three results moved rungs. (1) L1 scored **1.03662** vs 1.0374 predicted: the separable-constant metric model and the probed medians are validated to 0.001, and every pipeline falsifier now anchors on the *measured* floor 1.0366. (2) L2 passed its gate (val Dice 0.0047 → 0.6651; median in-muscle coverage on 60 test images 0.27 → 0.998) yet (3) L3 fired: the retrained pipeline scored 1.44645 > 0.937. Coverage was necessary, not sufficient; assumption A1 is broken. 176 of 309 test images already fall back to constants, so the 133 images we measure average *worse* than guessing. The Gemini bias-corrected test read (n=12) matched neither our numbers nor the probed medians and is retired as uninformative.

## The campaign reduces to one number
**Private-leaderboard UMUD score ≤ ~0.32 (public #3 = 0.31898, #1 0.29358 on 2026-09-04) at 2026-11-14 23:00 UTC**, with a FAIR-compliant public GPL-3.0 repo behind it.
Best banked: **1.03662** (L1 constant floor, rank 108/200; leaderboard median team 0.977). Best real pipeline: 1.44645. Gap to the cutoff ≈ 0.72 UMUD; the expert noise floor says ~0.30 is the honest ceiling.
Diagnosis after L3: the measured values are wrong in a way the benchmark cannot see (A2 broken). The only instrument that spans the test distribution is the leaderboard, and the cheapest way to use it is **per-target isolation**: submit one measured target with the other two held at the probed medians. Each isolation costs one slot, no model run, and attributes the 0.41 excess over the floor to PA, FL, or MT. Nothing is retrained until the isolations say which target to fix.

## Day 1 — floor (L0) — done 2026-08-31 (2.41344); true floor L1 — done 2026-09-04 (1.03662)

## The ladder
| rung | single variable | expected gain | cost (GPU h / subs / days) | falsifier (pre-registered) | status |
|---|---|---|---|---|---|
| L0 | constant midpoint (format + first score) | banks 2.41344 | 0 / 1 / 0 | n/a | **done 2026-08-31** |
| L1 | probed-median constant PA 16.42 / FL 81.5 / MT 20.7 | predicted 1.0374 | 0 / 1 / 0 | \|score − 1.0374\| > 0.01 | **done 2026-09-04: 1.03662, not fired** |
| L2 | fascicle U-Net retrained on 2,761 competition pairs | coverage 0.33 → ≥ 0.6 | 6 local GPU h / 0 / 1 | coverage < 0.50 or bench regress > 0.02 | **done 2026-09-03: coverage 0.998, Dice 0.6651, bench 0.3566; not fired** |
| L3 | full pipeline with the L2 detector, aspect 1.0, probed-median fallback | first score below the floor | 0 / 1 / 1 | public > 0.937 | **fired 2026-09-03: 1.44645** → isolations (L4) before any retrain |
| L4a | **isolation: measured MT only** (`mt_mm` from `outputs/sub_pipeline_a1.csv`), PA/FL at probed medians. MT is 44 % of the constant error budget and needs no fascicles | −0.10 to −0.25 vs 1.0366 if the aponeurosis pair is right | 0 / 1 / 0 | if public > 1.0366 the measured MT is worse than the median on test → aponeurosis pair selection / scale is the fault; insert a sector-crop + scale-lookup rung (L6) before MT is used again | **queued (submit_queue/L4a-isolate-mt.json)** — next tick submits first |
| L4b | isolation: measured FL only | attributes FL share | 0 / 1 / 0 | public > 1.0366 → FL routine is dead on test; L7 (DL_Track FL port) moves up | queued after L4a |
| L4c | isolation: measured PA only with the L2 detector (the 2026-09-03 PA-only read with DL_Track fascicles was 1.15267, i.e. worse than the floor) | attributes PA share | 0 / 1 / 0 | public > 1.0366 → retrained fascicle angles are still wrong on test → in-muscle angle QC / L8 SAM2 | queued after L4b |
| L5 | per-run consensus over the 27 five-frame video runs | −0.01 to −0.03 | 0 / 1 / 1 | paired sub delta ≥ 0 → dead | queued |
| L6 | scale as a lookup over device configurations (image shape → px/cm), validated on the 35 benchmark GT scales; also closes A10 | fixes the 160 cropped frames | 0.5 / 0 / 2 | lookup < 90 % of benchmark images within 3 % → dead | queued; promoted if L4a fires |
| L7 | port DL_Track's fascicle-length routine onto our PA/MT | −0.05 on benchmark | 2 / 1 / 3 | benchmark FL MAE not < 5.0 mm → dead | queued; promoted if L4b fires |
| L8 | SAM2 prompted aponeurosis masks + video propagation | replaces CNN pair step | 4 / 1 / 4 | not better than CNN route by ≥ 0.01 and no coverage gain → dead | queued |
| L9 | final-selection: 2 finals = best public + most conservative honest pipeline | protects private | 0 / 0 / 1 | n/a — human gate | queued (P4) |

Killed: VLM band identification (+1/20), VLM direct measurement (1.42–1.77), Gemini bias-corrected test read (n=12, matches neither), anisotropy 1.35, per-target calibration gains, manual test annotation (Foundational 4.b), and — as of v2 — "retrain the detector and the score follows" (A1).

## Assumption register (mirrored in campaign.json `pilot.assumptions`)
| id | claim | evidence | verify every | last verified | status |
|---|---|---|---|---|---|
| A1 | retrained fascicle net closes the transfer gap | coverage 0.998 but L3 1.44645 | 7 d | 2026-09-04 | **broken** — coverage necessary, not sufficient |
| A2 | local CV (35-image benchmark) tracks LB | 0.3566 bench vs 1.446 LB | 14 d | 2026-09-03 | **broken** — geometry conventions only |
| A3 | submission encoding correct | 14 COMPLETE, 1 ERROR (semicolon) | 30 d | 2026-09-03 | verified |
| A4 | public tracks private | not measurable before 2026-11-14; est. drift ~0.14 | 70 d | 2026-09-04 | unverified — reassess at L9 |
| A5 | metric mirror exact / separable-constant model | L1 1.03662 vs 1.0374 predicted | 30 d | 2026-09-04 | verified |
| A6 | OSF benchmark is permissible external data | licence conflict; discussion 737782 — see review ledger | 14 d | 2026-09-04 | unverified — human gate 2 |
| A7 | public top is overfit; honest 0.30–0.35 places | leader 0.29358 < expert floor 0.3032 | 70 d | 2026-09-04 | unverified — reassess at L9 |
| A8 | Kaggle CLI submission permitted | Stephen yes 2026-09-03; 15 accepted | 30 d | 2026-09-04 | verified |
| A9 | VLM band identification beats geometry | +1/20 | — | 2026-09-03 | dead |
| A10 | scale recoverable for 160 cropped frames | OCR 0/35; shape→scale lookup untested | 14 d | — | unverified — closed by L6 |

## Submission calendar
5/day (UTC), 2 finals. Used 15 (1 on 2026-09-04). Policy unchanged: ≤ 2 ladder submissions per UTC day, 3 held for a same-day fix or probe. Isolation CSVs (L4a–c) need no model run: L4a next tick, L4b and L4c the following ticks, one per tick so each result is read before the next is spent. No submission without `submit_queue/<id>.json` justified against **v2** and a `maibo_test` record whose sha256 matches the CSV.

## Human gates
1. ~~Auto-submit authorisation~~ — granted 2026-09-03 (A8 verified).
2. External-data declaration (A6): resolve the OSF licence conflict or confine the benchmark to development-only use before the final method description.
3. Final selection of the 2 submissions (P4) and the FAIR repo release.
4. Any change to `what_pays`.

## Feasibility / kill gate (unchanged dates, re-anchored)
metric: best public UMUD of a real (non-constant) pipeline submission · threshold: **≤ 0.45134** (stock DL_Track cluster) · by **2026-10-12** · consequence: money objective declared unreachable, numbered revision + push, drop to publishable-method tier. Interim checkpoint **2026-09-24**: a pipeline submission beats the measured floor by ≥ 0.10, i.e. **≤ 0.9366**.

## Review ledger
- maibo method review of v2 launched detached 2026-09-04 (`reports/maibo_method_review_v2.md`); question: are per-target isolations on the public board a sufficient diagnostic given 33 % scoring and SE ≈ 0.053, and what would make L4's falsifier uninformative. Findings disposed next tick.
- A6: discussion 737782 re-read 2026-09-04 (result recorded in ledger).
