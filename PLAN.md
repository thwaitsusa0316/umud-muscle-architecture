# UMUD Challenge — PLAN v4

Supersedes: v3 (2026-09-04T10:05Z; v3 text retained below as the base, v4 changes listed first).

**Why v4 (2026-09-04T17:45Z):** L6 (scale) ran and its falsifier did **not** fire. `scripts/scale_v2.py` (maibo_test PASS `maibo_tests/9bfdf2bd705ad8ac.json` after 3 FAIL rounds) → `reports/scale_v2.json`: **269/309** scaled (was 133), fallback **40/309** (< 60 kill line). Three mechanistic findings, 0 slots:
1. The 156 "cropped" video-mode frames are full Siemens Juniper screenshots whose sector finder returned the whole frame; they carry a left ruler 0..50 mm (1 cm major ticks) and, on 1200-wide frames, a panel `De 50 mm`. Left-ruler read: **148.1 px/cm** on 110 frames (comb and panel agree), 172.4 on 19, 148.2 on 10; **50 frames at 644×1088 still unscaled** (ticks sit on the spine at column 0, dimmer than the bg+25 threshold; label OCR drops the 5 of "50") — next tick.
2. Unit bug in `estimate_scale.py`: Siemens draws 2 mm ticks at 3.0 cm depth; the unit set {0.25, 0.5, 1, 2} forced 0.25 → **127.6 instead of 159.5 px/cm (20 % low)** on 23 fused images and their chrome-consensus dependants. This is the +13–16 % FL/MT bias fingerprint of v3. Chrome-consensus itself is unsound across depth settings (chrome does not change with depth): 74 images moved > 3 %, e.g. IMG_00010 135.4 → 152.0.
3. Siemens still layout has a constant **5.75 cm** lateral field at every depth seen (3.0/3.5/4.0/4.5/6.0/6.5 cm → 5.74–5.76), so `px/cm = sector_w / 5.75` fills 56 stills and cross-checks the rest: median |diff| 0.1 %, 1/34 > 4 %.
Open checks before L6b ships: within-run spread 90th pct **7.6 %** (6/43 runs > 2 %, all should be 0) — audit those runs; the 19 non-Siemens singletons (853×1069, ~513×465) have no cue. The v3 "benchmark unit check" is void: the 35 benchmark images are cropped (no chrome), so the checks are comb-vs-panel agreement and the footprint cross-check instead. **A10 → verified (partial: 269/309).** Coverage ceiling with 40 fallback rows ≈ (40/309) × 0.88 × 1.03662 ≈ **0.12 — below the 0.45134 gate; measurement rungs are unblocked.**
Rung moves: L6 → **done (falsifier not fired)**; **L6b promoted to rung 1**: point `predict_test.py` at `scale_v2.json` (code change → maibo_test), re-run the a1 pipeline, ship MT-only isolation with plausibility gate MT ∈ [8, 35] mm (1 slot, 2026-09-05); L6a' (644×1088 comb + run-spread audit) rung 2; L4d bracketing stays on idle slots.

---
*(v3 text follows.)*

Supersedes: v2 (2026-09-04T03:10Z, archived at reports/PLAN_v2_archive.md). Rests on BRIEF.md (2026-09-03). Revisions bump the version and say why.

**Why v3 (2026-09-04T10:05Z):** two things moved rungs. (1) **L4a fired**: measured-MT isolation scored **1.25464** vs the floor 1.03662, Δ_MT = **+0.21802**; undiluted per-measured-row damage 0.218 × 309/131 = **0.514**, i.e. the 131 measured MT values carry ≈ 4.6 mm more MAE than the constant 20.7. Measured MT is not shipped until it beats the median. (2) The maibo method review of v2 (`reports/maibo_method_review_v2.md`, 3 vendors, 34 numbered items) was read and its checkable claims were run this tick:
- **E1 confirmed**: fallback is per-target, not uniform — PA falls back on **16** rows (293 measured), FL on **180**, MT on **178**; FL and MT fall back on the *same* 178 rows; 14 rows fall back on all three. v2's "176 / 133" is retired. Dilution: 0.95 for PA, 0.42 for FL/MT.
- **E5 confirmed**: `outputs/diag_measuredPA_constFLMT.csv` carries a1.35's PA column (IMG_00001 = 22.1196, not a1's 16.0670). The banked 1.15267 is Δ_PA for the DL_Track pipeline, not the retrained one.
- **MT is invariant** between a1 and a1.35 on all 131 measured rows (checked), so L4a completes *both* decompositions: a1.35: Δ_PA 0.11605 + Δ_MT 0.21802 + Δ_FL **0.09463** = 0.42870 ✓. a1: Δ_MT 0.21802, Δ_PA + Δ_FL = 0.19181 (L4b resolves it).
- **Scale is the fallback**: `reports/scale_final.json` has a px/cm for **133** images and `none` for **176**; the 178 FL/MT fallback rows are the no-scale rows (+2 geometry failures). **None of the 138 video-run frames has a scale.** Coverage of FL/MT is a scale problem, full stop.
- **Bias fingerprint (label-free, 0 slots)**: on the rows we measure, median MT 23.58 vs probed 20.7 (+14 %), median FL 92.37 vs 81.5 (+13 %), median PA 13.01 vs 16.42 (−21 %). FL and MT high by the same factor is the signature of a px/cm that is ~13 % too low; MT p90 = 45 mm (mean 26.6) says pairing outliers add gross error on top.
- **Within-run dispersion (label-free)**: 27 video runs × 5 frames; PA within-run MAD = 0.13° (median), max 0.40°. PA's excess is **bias**, not noise — per-run averaging (old L5) cannot help PA, and FL/MT have zero measured video frames. L5 as written is dead.
- **Coverage ceiling (E4)**: with FL and MT unavailable on 178/309 rows, even perfect measurement gives S ≥ (178/309) × (FL + MT share of the 1.03662 floor). The per-target split of the floor is *not yet measured* (E2); the panel's reconstruction puts PA near 0.16 of the floor, which would set the ceiling near 0.50 > the 0.45134 kill gate. **No rung that leaves coverage unchanged can reach the gate; L6 (scale) moves to rung 1, unconditionally.**
- **Two noise constants** replace v2's single 0.10: δ_public = **0** (same public rows, deterministic scorer: paired deltas are exact) and δ_transfer = **0.10** (public → private, 2 SE). Promotion additionally needs a pre-registered *minimum interpretable delta* of **±0.05** on an isolation (guards against reading a ~44-public-image effect as a roadmap).
- **A7 retired as reasoning**: "leader 0.29358 < expert mean 0.3032 ⇒ overfit" does not follow (different data, different annotators; a single-protocol GT is learnable below inter-rater noise). Observation kept, comfort removed: 0.29 is probably real skill.
- **Direct image→(PA, FL, MT) regressor (panel item 11) is dead**: `data/raw` carries only image/mask pairs (apo, fascicle), no scalar labels and no px/cm field; test labelling is forbidden (Foundational 4.b).

## The campaign reduces to one number
**Private-leaderboard UMUD score ≤ ~0.32 (public #3 = 0.31898, #1 0.29358 on 2026-09-04) at 2026-11-14 23:00 UTC**, with a FAIR-compliant public GPL-3.0 repo behind it.
Three references, kept apart (panel): constant-class floor **1.03662** (measured); leaderboard target **0.31898** (public #3, dated 2026-09-04, moves); external benchmark reference 0.3032 (OSF expert mean, *non-binding* for the test distribution).
Best banked: **1.03662** (L1). Best real pipeline: 1.44645. Best partial: 1.15267 (a1.35 PA only). Score trend: 1.03662 → 1.25464 → 1.20053 (L4a, L4b: probes, both red vs the floor, both informative). **a1 decomposition complete: PA +0.028 / FL +0.164 / MT +0.218 = 0.410.**

## Day 1 — floor (L0) — done 2026-08-31 (2.41344); true floor L1 — done 2026-09-04 (1.03662)

## The ladder
| rung | single variable | expected gain | cost (GPU h / subs / days) | falsifier (pre-registered) | status |
|---|---|---|---|---|---|
| L0–L3 | as v2 | — | — | — | done (L3 fired 1.44645) |
| L4a | isolation: measured MT (a1), PA/FL at medians | −0.10 to −0.25 if pairing/scale right | 0 / 1 / 0 | Δ > +0.05 harmful; < −0.05 helpful; else uninformative; report undiluted Δ×309/131 | **fired 2026-09-04: 1.25464, Δ +0.218, undiluted +0.514** → scale/pairing at fault |
| L4b | isolation: measured FL (a1, 129 rows), PA/MT at medians (`outputs/sub_L4_isolate_fl.csv`, sha b9ed0430…) | resolves Δ_FL vs Δ_PA of the a1 excess 0.19181 | 0 / 1 / 0 | Δ > +0.05 → FL routine harmful on test (undiluted ×309/129); < −0.05 helpful; else uninformative. Δ_PA(a1) = 0.19181 − Δ_FL by subtraction | **fired 2026-09-04: 1.20053, Δ_FL +0.164, undiluted +0.393** → Δ_PA(a1) = +0.028 (neutral). Scale-fault fingerprint confirmed: both mm targets harmful, PA neutral |
| L4c | isolation: measured PA (a1, 293 rows) — now a **checksum**, not a measurement | verifies CSV provenance | 0 / 1 / 0 | \|Δ_PA + Δ_FL + 0.21802 − 0.40983\| > 0.002 → the CSVs are not what we think (A11 broken); everything downstream suspended | **done 2026-09-04: 1.06452 = predicted 1.06452 exactly**; Σ deltas = 0.40983 residual 0.00000 → A11 verified; Δ_PA(a1) = +0.028 neutral |
| L4d | **bracketing probes** (E2): per target, constant at 0 and far above support (PA {0, 200}, FL {0, 400}, MT {0, 200}), others at medians → absolute MAE_t(median) and the exact per-target split of the floor | fixes the coverage ceiling exactly | 0 / 6 / 2 | the three shares must sum to 1.03662 ± 0.002 or the separable model (A5) is broken | queued 2026-09-05/06 on idle slots |
| **L6** | **scale for the 176 no-scale images**: lookup over device configurations (image shape / chrome cluster → px/cm; `scale_final.json` already shows ≤ 12 discrete values, 135 px/cm on 42 images), with per-run sharing for the 27 video runs; unit/convention check on the 35 benchmark GT scales (legitimate use of the benchmark) | FL/MT coverage 131 → ≥ 250 rows; moves the ceiling below the gate | 0.5 / 0 / 2 | FL/MT fallback rows not < **60/309** → dead; benchmark scale within 3 % on < 90 % of 35 → unit/convention wrong, fix before use | **done 2026-09-04T17:45Z: 269/309 scaled, fallback 40 < 60 — not fired** (v4) |
| **L6b** | re-run pipeline with L6 scale (`reports/scale_v2.json`); **MT isolation again** (new rows, plausibility gate MT ∈ [8, 35] mm else median — mechanistic, not public-tuned) | Δ_MT from +0.218 to ≤ 0 | 0 / 1 / 1 | Δ_MT(undiluted ×309/n) > 0 → scale is not the MT fault; pairing is → L8 | after L6 |
| L6c | confidence-gated routing (panel 10): rank measured rows by geometric residual \|MT − FL·sin PA\|, plausibility, and run-consistency; ship measured values only for the top decile, medians elsewhere | first pipeline below the floor | 0 / 1 / 1 | not < 1.03662 → the branch has no usable subset; retire the geometric branch, do not retrain | after L6b |
| L7 | port DL_Track's FL routine onto our PA/MT | Δ_FL ≤ −0.05 | 2 / 1 / 3 | **LB isolation** Δ_FL(undiluted) not < 0 → dead (benchmark used for convention only; A2) | after L4b if FL fires |
| L8 | SAM2 prompted aponeurosis masks + video propagation | replaces pairing | 4 / 1 / 4 | LB isolation Δ_MT not < 0 by ≥ 0.05 → dead | after L6b if pairing fires |
| L9 | finals, **pre-registered now**: (i) the constant floor L1; (ii) the pipeline chosen on *mechanism* (passes L6/L6b/L6c falsifiers), not on best public score | protects private | 0 / 0 / 1 | human gate 3 | P4 |

Killed (cumulative): VLM band identification, VLM direct measurement, Gemini test read, anisotropy 1.35, per-target calibration, manual test annotation, "retrain and the score follows" (A1), **L5 per-run consensus** (PA MAD 0.13°: nothing to average; FL/MT have no measured video frames), **direct scalar regressor** (no labels), **A7 as a ceiling argument**.

## Coverage-ceiling gate (new, pre-registered)
`S_ceiling = Σ_t (n_fallback_t / 309) × share_t(floor)`, recomputed after every pipeline change (shares from L4d once measured; until then the panel reconstruction FL 0.52 / MT 0.36 / PA 0.16 is the *placeholder*, labelled as such). **If S_ceiling > 0.45134, no rung that leaves coverage unchanged may be scheduled.** Today (178/309 on FL+MT ≈ 0.88 of the floor): S_ceiling ≈ 0.51 — bad, above the gate; L6 is the only rung that moves it.

## Decision table for the isolations (panel codex #8)
| result | next discriminating experiment (not a retrain) |
|---|---|
| MT harmful (**fired**) | scale vs pairing: L6 fixes scale, L6b re-isolates MT; if still harmful → pairing (L8) |
| FL harmful | scale vs fascicle geometry: L6 first; if still harmful after L6 → endpoint selection (L7) |
| FL uninformative / helpful | FL routine kept; excess 0.192 is PA → angle convention / in-muscle QC (PA is precise, MAD 0.13°, so it is a *bias* — one additive/multiplicative correction, checked mechanistically on the benchmark conventions, not tuned on public) |
| checksum L4c fails | A11 broken; suspend all attributions, rebuild CSVs from one generator |

## Assumption register (mirrored in campaign.json `pilot.assumptions`)
| id | claim | evidence | verify every | last verified | status |
|---|---|---|---|---|---|
| A1 | retrained fascicle net closes the transfer gap | coverage 0.998 but L3 1.44645; L4a shows MT (no fascicles) alone carries +0.218 | 7 d | 2026-09-04 | **broken** |
| A2 | local CV (35-image benchmark) tracks LB | 0.3566 bench vs 1.446 LB | 14 d | 2026-09-04 | **broken** — conventions/units/scale only |
| A3 | submission encoding correct | 16 COMPLETE, 1 ERROR | 30 d | 2026-09-04 | verified |
| A4 | public tracks private | not measurable before 2026-11-14; δ_transfer 0.10 | 70 d | 2026-09-04 | unverified — L9 |
| A5 | metric mirror exact / separable | L1 1.03662 vs 1.0374; separability is algebraic (metric.py); L4d tests the interpolation part | 30 d | 2026-09-04 | verified |
| A6 | OSF benchmark permissible external data | discussion 737782 unreadable headlessly | 14 d | 2026-09-04 | unverified — human gate 2 (Stephen) |
| A7 | public top is overfit | argument does not follow (panel); observation only | — | 2026-09-04 | **retired as reasoning** |
| A8 | Kaggle CLI submission permitted | 16 accepted | 30 d | 2026-09-04 | verified |
| A9 | VLM band identification | +1/20 | — | 2026-09-03 | dead |
| A10 | scale recoverable for the cropped frames | scale_v2.json 2026-09-04T17:45Z: 269/309 (video frames read from the left ruler, 148.1 px/cm on 110); 50 at 644×1088 + 19 other-device singletons open | 14 d | 2026-09-04 | verified (partial) |
| A11 | fallback rows carry exactly the L1 triple and all CSVs share id order | grep: 178 rows `,81.5000,20.7000`, 16 rows `,16.4200,`; L1/a1/L4a/L4b id order identical | 14 d | 2026-09-04 | verified |
| A12 | coverage ceiling: the architecture cannot reach 0.45134 while FL/MT fall back on ≥ 178 rows | arithmetic above; shares placeholder until L4d | after every pipeline change | 2026-09-04 | verified (arithmetic) |

## Submission calendar
5/day (UTC), 2 finals. Used 16 (2 on 2026-09-04: L1, L4a). **Policy v3 (panel 13):** mechanistic probes (isolations, bracketing, checksums) are exact and carry no selection risk → exempt from the 2/day rationing; keep **≥ 1 slot/day in reserve**. Selection is pre-registered separately at L9. No submission without `submit_queue/<id>.json` justified against **v3** and a `maibo_test` record whose sha256 matches the CSV.

## Human gates
1. ~~Auto-submit~~ — granted 2026-09-03.
2. External-data declaration (A6): Stephen reads https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/discussion/737782 in a browser.
3. Final selection (pre-registered above) and the FAIR repo release.
4. Any change to `what_pays`.

## Feasibility / kill gate (unchanged)
metric: best public UMUD of a real (non-constant) pipeline submission · threshold **≤ 0.45134** · by **2026-10-12** · consequence: money objective declared unreachable, numbered revision + push. Interim checkpoint **2026-09-24**: a pipeline submission **≤ 0.9366**. Coverage-ceiling gate above is the early-warning: if L6 fails its falsifier by 2026-09-10, the kill gate is effectively decided and the revision is written then, not on 10-12.

## Review ledger
- maibo method review of v2 — **disposed 2026-09-04T10:05Z** in this version (E1, E5 confirmed by test; E3 adopted; E4 adopted as the coverage-ceiling gate; E2 queued as L4d; items 1 (hold L4a) overruled — L4a was already gated and exact, and its result agrees with the panel's scale diagnosis; 3, 4 (scale sweep) folded into L6/L6b; 5, 6, 7, 8, 9, 10, 12, 13, 14 adopted; 11 dead (no labels); codex 2/4/8/9 adopted; grok 1/2/5/7/8/9/10 adopted).
- L4b read 2026-09-04T09:57Z: 1.20053 (Δ_FL +0.164). Decision table row 'FL harmful' → L6 first, L7 only if FL still harmful after L6. No rung moved; v3 stands.
- L4c read 2026-09-04T10:02Z: 1.06452, checksum exact. Attribution of a1's 0.40983 excess is closed: PA +0.028 / FL +0.164 / MT +0.218. Next slot use: L6 pipeline (scale) then L6b MT re-isolation; L4d bracketing on idle slots 2026-09-05/06.
- A6: unchanged, owner Stephen.
