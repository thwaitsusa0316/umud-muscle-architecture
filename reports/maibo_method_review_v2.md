panel: 3 of 3 answered, 0 refused, families: anthropic, openai, xai

--- claude (anthropic) ok ---
Provenance first: `python` execution and file writes were blocked in this session, so everything below comes from `head`/`grep` over the CSVs in `outputs/`, the scorer source in `src/umud/metric.py`, and hand arithmetic on the sweep table in BRIEF.md. The grep counts are checkable; the per-target decomposition of the floor is my reconstruction, not a measurement, and I flag it as such.

## (1) Is the isolation design sound

The algebra is sound and stronger than the plan claims. `src/umud/metric.py` computes each target's MAE, MedAE and RMSE independently and sums them, so the score is **exactly** separable across targets *and* across rows. An isolation delta is therefore not an estimate — it is an exact linear functional of the per-row errors, with zero sampling noise on the public set. The plan does not need A5 to be "verified" for this; separability is algebraic. A5's empirical part (predicting a *new* constant's score by interpolating MAE(c)) is a different and genuinely approximate claim, and the plan conflates the two.

What is not sound is the falsifier and the basis.

The fallback structure is not what PLAN v2 says. From `outputs/sub_pipeline_a1.csv` (309 rows):

| target | fallback rows | measured |
|---|---|---|
| PA (=16.4200) | 16 | **293** |
| FL (=81.5000) | 180 | 129 |
| MT (=20.7000) | 178 | 131 |
| FL and MT together | **178** | 131 |
| all three | 14 | — |

FL and MT fall back on the *same* 178 rows; PA is measured on 293. So there is no single "133 measured images". The dilution factor differs by target: **0.95 for PA, 0.42 for FL/MT** (neutral facts, but they invert the design). Consequences:

- The falsifier "public > 1.0366" is a sign test on a deterministic number whose sum is already known to be +0.40983 (= 1.44645 − 1.03662). At least one target must fire. It cannot discriminate.
- A magnitude that matters per-row is hidden by dilution: for FL/MT the true per-measured-row damage is Δ × 309/131 = **2.36 × Δ**. An MT that is meaningfully worse per row can still show Δ_MT ≈ 0.03 and "pass".
- **Targets are not the causal basis.** A wrong px/cm scale is one fault that appears in FL and MT proportionally and never in PA — which is exactly the fingerprint of the 178-row set (mm-valued targets fail together; the dimensionless one does not). L4a and L4b would both fire and you would conclude two routines are dead when one scalar is wrong.
- The L4c baseline is mismatched: `outputs/diag_measuredPA_constFLMT.csv` carries `IMG_00001 = 22.1196`, which is **a1.35's** PA column, not a1's (`16.0670`). The 1.15267 read describes the DL_Track pipeline, not the retrained one.
- Two of the three isolations are redundant. Δ_PA(a1.35) = 1.15267 − 1.03662 = **0.11605** is already banked, and Δ_FL + Δ_MT(a1.35) = 1.46532 − 1.03662 − 0.11605 = **0.31265** follows for free. `IMG_00001` has MT = 25.3141 in *both* a1 and a1.35, suggesting MT is invariant to the retrain — if so, one MT isolation completes both decompositions and the third slot is a checksum, not a measurement.
- Uncheckable residual: whether the 178 fallback rows are proportionally represented in the public 33%. If they are not, the dilution differs between splits and the isolation magnitude does not transfer. You cannot test this; name it and stop.

## (2) Is the floor/ceiling framing right

The floor is right and safer than the plan treats it. Re-deriving the medians from the sweeps: the local-linear median estimator gives PA 16.42° (F(12.5)=0.092, F(17.5)=0.612) and MT 20.72 mm (F(17.5)=0.246, F(22.5)=0.640) — both reproduce the plan's constants exactly, so that method is correct. The public-fitted constant overfits the private split by roughly 1 SE of a sample median (≈3.2 mm FL, ≈0.63 mm MT), which costs ~0.004 in score — negligible. The constant submission is a legitimate insurance final.

The ceiling framing is wrong in two places.

**The real ceiling is architectural, not human.** With FL/MT falling back on 178 of 309 rows, and *perfect* FL/MT on the 131 it measures and *perfect* PA everywhere:

S_min = (178/309) × (FL_share + MT_share) = 0.576 × (1.03662 − PA_share)

For S_min ≤ 0.319 (paying) you would need PA_share ≥ 0.483 — i.e. PA alone would have to be 47% of the constant floor, which contradicts the plan's own claim that MT is 44%. **A paying position is arithmetically unreachable under the current architecture regardless of how good the measurement branch gets.** For the 2026-10-12 gate of 0.45134 you need PA_share ≥ 0.253; my reconstruction puts it near 0.16, so that gate is probably unreachable too — but that one turns on a number nobody has measured, which is why it is item 1 below.

**A7 is bad reasoning.** "Leader 0.29358 < expert mean 0.3032, therefore overfit" does not follow. 0.3032 is inter-rater disagreement on a different dataset with a different annotation convention; if the competition GT comes from one annotator or one protocol, a model can learn that protocol's systematic bias and beat inter-rater noise. Nothing here is evidence of overfitting. Removing that comfort matters: it means 0.29 is real skill and the leaders are almost certainly not running a geometric pipeline with a scale fallback.

## (3) Failure modes that burn the gate without learning why

- Spending slots attributing among three targets when the coverage arithmetic already forecloses the objective, and the arithmetic is free.
- Diagnosing along the target axis when the fault axis is `{scale, angle convention, aponeurosis pairing}` — scale is one parameter visible in two targets.
- Falsifiers for L7 ("benchmark FL MAE < 5.0 mm") and L8 ("better by ≥ 0.01") written against the 35-image benchmark, which A2 declares non-tracking. Those two rungs cannot currently be killed or passed.
- Treating "minimum interpretable delta ≈ 0.10" as applying to method comparison. Paired deltas on the same public set are **exact** (the plan itself records resubmission variance = 0). 0.10 is the public→private transfer bound only. As written it will discard real signal — L5's stated −0.01 to −0.03 would be declared uninterpretable when it is exactly measurable.
- Rationing 2 ladder submissions/day. ~40 days to the gate × 5 = ~200 slots against a 6-submission decomposition. The scarce resource is not slots; it is *independent mechanistic hypotheses*.
- No rung on the ladder changes the 178-row fallback rate except L6, which sits sixth and is gated on L4a firing.

---

## TEST PROTOCOLS

**E1 — the plan's fallback count and dilution are wrong.**
CLAIM: PLAN v2 says "176 of 309 test images already fall back to constants … the 133 images we measure"; in fact 178 rows fall back on FL *and* MT, only 16 fall back on PA, and 293 rows carry a measured PA.
TEST (shell, from repo root):
`grep -c ',16\.4200,' outputs/sub_pipeline_a1.csv; grep -c ',81\.5000,20\.7000' outputs/sub_pipeline_a1.csv; grep -c ',16\.4200,81\.5000,20\.7000' outputs/sub_pipeline_a1.csv`
EXPECTED IF I AM RIGHT: `16`, `178`, `14`.
EXPECTED IF I AM WRONG: the first count ≈ 176 and the second ≈ 176 (a uniform per-row fallback).
SCOPE: PLAN.md §"Why v2" and rows L4a/L4b/L4c; `campaign.json pilot.assumptions.A1`; any code computing a `fallback` flag or `coverage` in `src/umud/`; `reports/maibo_method_review_v2.md`, which was briefed on the 176/133 numbers.
FIXED WHEN: PLAN.md states per-target fallback counts (16 / 180 / 178) and every L4 delta is reported as Δ_t × 309 / n_measured_t.

**E2 — the per-target split of the 1.03662 floor is asserted, not measured.**
CLAIM: "MT is 44% of the constant error budget" is not derivable from the sweeps in hand — the sweeps give only *differences* MAE_t(c₁) − MAE_t(c₂), never an absolute per-target MAE. My reconstruction (anchoring each MAE at a constant above the support, where slope saturates at +1) puts the split near FL 0.52 / MT 0.36 / PA 0.16, i.e. FL is the largest share, not MT.
TEST (2 leaderboard submissions per target, 6 total; MAE is piecewise-linear in c with slope 2F(c)−1, so at c far above the support MAE(c) = c − μ and at c = 0, MAE = μ):
submit `(PA=0, FL=81.5, MT=20.7)` → S₀ and `(PA=200, FL=81.5, MT=20.7)` → S₂₀₀; then μ_PA = (200 − 18·(S₂₀₀ − S₀))/2, and MAE_PA(16.42) = μ_PA − 18·(S₀ − 1.03662). Repeat with FL ∈ {0, 400} (factor 36) and MT ∈ {0, 200} (factor 9).
EXPECTED IF I AM RIGHT: MAE_FL(81.5)/12/3 > MAE_MT(20.7)/3/3, with MT's share in 0.30–0.40 of 1.03662.
EXPECTED IF I AM WRONG: MT's share ≥ 0.44 and exceeds FL's.
SCOPE: L4a's ordering rationale in PLAN.md; the docstring claim in `src/umud/metric.py` that "muscle thickness … is the best value per unit of work"; `budget()` in the same file; any effort allocation in `campaign.json`.
FIXED WHEN: the three absolute MAEs are recorded in `reports/` and sum, via /6, /12, /3 and ÷3, to 1.03662 ± 0.002.

**E3 — "minimum interpretable public delta ≈ 0.10" is a category error for paired comparison.**
CLAIM: on the same public rows the scorer is deterministic, so the difference between two submissions has zero sampling error; 0.053 is the SE of the *public→private* transfer, not of a paired public delta.
TEST: resubmit `outputs/sub_L1_probed_median.csv` unchanged and compare to the banked 1.03662.
EXPECTED IF I AM RIGHT: identical to all printed digits.
EXPECTED IF I AM WRONG: a difference of order 0.05.
SCOPE: BRIEF.md "Noise floor" paragraph; the L5 falsifier ("−0.01 to −0.03"); the L8 falsifier ("≥ 0.01"); any decision rule in `campaign.json` citing 0.10.
FIXED WHEN: BRIEF.md carries two named constants — δ_public = 0 (exact) and δ_transfer = 0.10 — and each falsifier cites the right one.

**E4 — the architecture cannot reach the paying cutoff.**
CLAIM: with FL and MT unavailable on 178/309 rows, S ≥ (178/309)·(FL_share + MT_share) even at zero measurement error, which exceeds 0.319 unless PA is ≥ 47% of the floor.
TEST: hand computation, all inputs stated — 178/309 = 0.5761; floor = 1.03662; required FL+MT share for S = 0.319 is 0.319/0.5761 = 0.5537; required PA share = 1.03662 − 0.5537 = 0.4829, i.e. 46.6% of the floor. Compare against the measured PA share from E2.
EXPECTED IF I AM RIGHT: measured PA share < 0.25, so the bound is > 0.45 and the cutoff is unreachable.
EXPECTED IF I AM WRONG: PA share ≥ 0.483.
SCOPE: PLAN.md "The campaign reduces to one number", the whole L4–L8 ladder, and `feasibility_gate` in `campaign.json`.
FIXED WHEN: PLAN.md carries an explicit "coverage ceiling" line recomputed from the current fallback count, and no rung is scheduled whose best case sits above it.

**E5 — L4c's stated baseline came from a different pipeline.**
CLAIM: the 1.15267 read used a1.35's PA column, so it is not the comparison point for an L2-detector PA isolation.
TEST: `head -2 outputs/diag_measuredPA_constFLMT.csv outputs/sub_pipeline_a1.35.csv outputs/sub_pipeline_a1.csv`
EXPECTED IF I AM RIGHT: the diagnostic's `IMG_00001` PA = 22.1196 = a1.35's, ≠ a1's 16.0670.
EXPECTED IF I AM WRONG: it equals 16.0670.
SCOPE: PLAN.md L4c row; `reports/score_history.json test_set_history`; `submit_queue/*.json`.
FIXED WHEN: every score_history entry names the sha256 of the CSV that produced it and the full constant triple used for fallback.

---

## Numbered changes

1. **Do not submit L4a next.** Submit the six bracketing probes from E2 first (two per target, `c = 0` and `c` above the support, other two at probed medians). They give the exact per-target decomposition of the 1.03662 floor, which is the input to every feasibility statement in the plan and is currently a guess.
2. **Add a pre-registered coverage-ceiling gate to PLAN.md**, recomputed after every pipeline change: `S_ceiling = (n_fallback/309) × (share of the floor held by the fallen-back targets)`. Pre-register: if S_ceiling > 0.45134, no rung that leaves coverage unchanged may be scheduled.
3. **Promote L6 (scale lookup) to rung 1, unconditionally.** It is the only queued rung that moves the 178. Retarget its falsifier from the benchmark to the thing that matters: fallback rows for FL/MT must drop below 60 of 309, or it is dead.
4. **Re-basis the diagnostic from `{PA, FL, MT}` to `{scale, angle, pairing}`.** The first probe should be a multiplicative scale sweep on measured FL and MT jointly — `k ∈ {0.8, 0.9, 1.0, 1.15, 1.3}`, medians elsewhere. MAE is convex in a multiplicative correction, so five slots locate the minimum, and a scale fault is mechanistic (physical, device-linked) rather than a public-set tuning. If the curve's minimum is at k ≈ 1.0, scale is exonerated and the fault is pairing or convention.
5. **Cut the isolations from three to two plus a checksum.** Δ_PA(a1.35) = 0.11605 is banked. Run isolate-MT (it is likely invariant across a1/a1.35, so it completes both decompositions) and isolate-FL; take Δ_PA(a1) by subtraction from 0.40983; then spend the third slot only as a checksum, pre-registering that |ΣΔ − 0.40983| > 0.002 means the CSVs are not what you think they are.
6. **Report every isolation delta undiluted**: Δ_t × 309/n_measured_t, and state the falsifier on that number, not on the sign of the raw score.
7. **Replace the 0.10 constant with two constants** — δ_public = 0 (exact, deterministic) and δ_transfer = 0.10 (public→private only) — and re-audit every falsifier in the ladder for which one it should cite. L5 and L8 currently cite the wrong one.
8. **Strike the benchmark-based falsifiers on L7 and L8.** A2 says the instrument is broken for transfer claims. Keep the benchmark only for what it can still do: unit, convention and physical-scale checks with real ground truth (which is why L6's benchmark falsifier is legitimate and L7's is not).
9. **Build three label-free instruments on the test set itself, before spending more slots.** (a) *Within-run dispersion*: 27 runs × 5 frames where the true values are near-constant; per-target within-run MAD is a lower bound on that target's MAE requiring no labels and no submissions, and it splits the 0.41 excess into variance (fixable by averaging) and bias (not). (b) *Geometric residual* `r = MT − FL·sin(PA)` per image — the probed medians satisfy it to 11% (81.5·sin 16.42° = 23.04 vs 20.7), so gross unit errors are already excluded, but per-image violations are a free routing signal. (c) *Bias check*: the median of your **measured rows only** per target versus the probed public medians. If your measured MT median is off by 3 mm, that alone is 0.33 of score and a constant shift fixes it today for zero slots. None of these are in the plan and all three are free.
10. **Add a confidence-gated routing rung, ahead of any retrain.** Given the measurement branch is worse than the median on the rows it covers, the highest-value question is not "which target" but "on which rows is measured better than median". Rank rows by a confidence score built from (9a)–(9c), route only the top decile to measured, and spend one slot. If that beats 1.03662, the ordering is real and the threshold is tunable; if it does not, the branch has no usable subset and should be retired rather than retrained.
11. **Open a non-geometric rung now, not after L8.** Every rung on the ladder is a geometric pipeline with a scale dependency, and the coverage arithmetic forecloses that family. A direct image→(PA, FL, MT) regressor learns px/cm implicitly from device appearance and has no fallback rows. I do not know from the supplied material whether the 2,761 training pairs carry scalar PA/FL/MT labels or a px/cm field — that is the blocking question, and it should be answered before the ladder is re-sequenced, not assumed either way.
12. **Retire A7 as reasoning**, keeping the observation. "Leader below the inter-rater floor" is expected under single-annotator ground truth and is not evidence of overfitting. Downstream, this removes the justification for treating ~0.30 as a ceiling and for the L9 assumption that the public top will regress on private.
13. **Relax the 2/day rationing for measurement submissions and pre-register the selection rule separately.** Mechanistic probes (scale sweeps, bracketing constants, isolations) are exact and carry no selection risk; only *choosing* among tunings by public score does. Pre-register now that the two finals will be (i) the constant floor and (ii) the pipeline chosen on mechanism, not on best public score — then the daily limit stops being a constraint at all.
14. **Record, for every submission, the sha256 of the CSV and the full constant triple used for fallback**, in `reports/score_history.json`. Only 5 rows are identical between `sub_pipeline_a1.csv` and `sub_pipeline_a1.35.csv`, and the PA-only diagnostic silently belongs to the older one; without this field the checksum in change 5 is unverifiable.

---
Our score, last three: 1.46532 → 1.44645 → **1.03662** (latest is the best banked, green; still 0.72 above the 0.31898 paying cutoff — bad).
Running: nothing (python execution and file writes were blocked this session, so no artifact was produced). Next update: on request. Trigger: a decision on whether to hold L4a and run the E2 bracketing probes first.

--- codex (openai) ok ---
Reading additional input from stdin...
OpenAI Codex v0.150.1
--------
workdir: /Users/smt2/data/bounty-hunting/kaggle/UMUD Muscle
model: gpt-5.4
provider: openai
approval: never
sandbox: read-only
reasoning effort: none
reasoning summaries: none
session id: 01a06a6d-4d33-71b2-b294-5898608bcb5b
--------
user
# Panelist protocol

You are answering as one independent member of a multi-vendor panel. Other
models from other vendors are being asked the same thing, separately. They
cannot see your answer and you cannot see theirs. Your value here is that you
do NOT share their reasoning, so do not try to give the expected answer.

**These are mechanics, not opinions.** Nothing below tells you what to
conclude. If it ever seems to, that is a bug — ignore it and say so.

## What a useful answer looks like

- **Disagreement is the product.** If the framing is wrong, say the framing is
  wrong rather than answering the question as posed. A confident wrong answer
  costs more here than a hedged right one.
- **Abstaining is free and it is honest.** "The evidence given does not settle
  this" is a complete answer. It is not scored as a failure and it never
  reflects badly on you. Guessing to look decisive is the failure.
- **Say which claims are checkable.** Separate what you read in the supplied
  material from what you are supplying from training. Both are welcome;
  conflating them is not.
- **Name what would change your mind.** An answer with no falsifier cannot be
  acted on, because nobody can tell later whether it held.
- **Do not invent citations.** Every identifier you give is resolved against
  arXiv, Crossref and HTTP by code, and a fabricated one is worse than none. If
  you have no resolvable source, say the claim rests on training data.

## Missing context

You will sometimes be told explicitly which values were NOT given to you. Do
not fill those gaps by assumption. Name what you would need and abstain on the
parts that turn on it. Answering around a stated gap is the single most
expensive thing you can do here.

## Output

- Plain text. No preamble, no restating the question, no closing summary of
  what you just said.
- When a reply format is specified, follow it exactly — the reply is parsed by
  code, and a wrapper sentence or a markdown fence around JSON breaks it.
- Length should match the question. Long is not thorough.

## When you find an error in code, arithmetic, or a measurement

Finding it is half the job. **An error report that cannot be checked mechanically
is an opinion, and the person receiving it has to take your word for it.** So
whenever you assert that something is wrong — a wrong constant, a wrong formula,
a units confusion, an off-by-one, a bad statistical step, a claim the artifact
does not support — you must also return a **TEST PROTOCOL** in this shape:

- **CLAIM** — one sentence naming what is wrong, in the units it is wrong in.
- **TEST** — an executable check. Real code (any language, but say which), or an
  exact command, or a hand-computation with every input value written out. It
  must run against the artifact as supplied, not against a described version of
  it. If it needs a value you were not given, say so and stop, rather than
  inventing one.
- **EXPECTED IF I AM RIGHT** — the literal output the test prints when the error
  is present.
- **EXPECTED IF I AM WRONG** — the literal output when it is not. **If these two
  are the same, or you cannot state the second, you have not found an error.**
  Say that plainly instead.
- **SCOPE** — everywhere else the same value or formula could have propagated,
  named as specific files, functions or fields to grep. A wrong number is never
  one number; it is every number computed after it.
- **FIXED WHEN** — the condition under which the fix counts as verified. Not
  "the code was edited". A test that passes, or a value that now reconciles.

State the protocol even when the error looks obvious. The obvious ones are the
ones that get accepted without checking, and those are the ones that propagate.

If you are asked to re-review a fix, do not re-derive the error from scratch and
do not accept "fixed" on assertion. **Run the protocol against the new artifact,
report the actual output, and say RESOLVED or NOT RESOLVED against the FIXED
WHEN condition you were given.** If the fix is correct but incomplete — right at
the site, still wrong somewhere SCOPE named — that is NOT RESOLVED, and say
which site is still wrong.

## Vendor mechanics — openai family

You may also be loading a personal `AGENTS.md` from this machine. Where it
conflicts with the panelist protocol above, the protocol wins for this request.
In particular:

- **Write the artifact when asked.** If a request asks for a design, a fix, a
  patch or an implementation, produce it directly here. A standing rule to
  route implementation through another tool does not apply to panel work —
  written artifacts are the point, and a critique returned in place of the
  requested artifact is a non-answer.
- You are not operating on a repository unless a working directory was
  supplied. Do not propose to run commands or edit files; return the content.

CONTEXT FILES (2 attached), then the task.

=== BEGIN PLAN.md ===
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

=== END PLAN.md ===

=== BEGIN BRIEF.md ===
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
- Public notebooks/models: **none** (Code and Models tabs empty). Six-team cluster at 0.45134 is almost certainly stock DL_Track.
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

=== END BRIEF.md ===

Review the MEASUREMENT DESIGN of PLAN.md v2 for this Kaggle competition (UMUD score = tolerance-normalised MAE over PA/FL/MT, lower is better; public board scores 33% of 309 test images, so one submission's SE is about 0.053). The plan proposes per-target ISOLATION submissions (one measured target, the other two held at leaderboard-probed medians) against a measured constant floor of 1.03662 to attribute a pipeline's 0.41 excess error to PA, FL or MT before any retraining. Questions: (1) is the isolation design sound given the metric is additive over targets, and what would make each L4 falsifier uninformative (e.g. the ~178/309 fallback rows); (2) is the floor/ceiling framing right (floor 1.0366 constant, ceiling ~0.30 expert noise, cutoff 0.319); (3) name the failure modes that would let this ladder consume the 2026-10-12 feasibility gate without learning why the pipeline is worse than a constant. Method, not increments. Give a numbered list of concrete changes.
codex
1. Keep L4, but rewrite its purpose more narrowly: it is sound for public-score attribution, not for root-cause diagnosis. Checkable from the supplied metric: with the other two columns fixed at the same constants, UMUD is additive over PA, FL, and MT, so on the fixed public subset
   `S(full) - S(constant) = [S(PA only)-S(constant)] + [S(FL only)-S(constant)] + [S(MT only)-S(constant)]`
   up to rounding. If that identity does not hold across L1/L3/L4a/L4b/L4c, the issue is not “which target is bad”; it is CSV construction, fallback routing not held fixed, or a bookkeeping mismatch.

2. Replace each L4 falsifier `public > 1.0366` with an indifference-band rule. With 176/309 fallback rows, only about 133 rows differ from the constant, so an isolate only acts on about 43% of the test set and about 44 public images in expectation. That dilution means “near the floor” is not evidence that the target is fine; it is often evidence that the design had low power. Concrete change:
   `if |S(target only) - 1.03662| < pre-registered minimum detectable delta, classify as uninformative, not exculpatory`
   The current falsifier is too coarse.

3. Pre-compute a maximum possible movement for each isolate before submitting it. This is checkable from the CSVs alone: count rows where the isolate differs from the constant and derive the largest public-score change that target could produce under perfect predictions on those rows. If that upper bound is below your practical detection threshold, the submission is dead on arrival and should not spend quota.

4. Add one explicit “decomposition audit” step before interpreting leaderboard numbers. The audit is mechanical:
   - same `image_id` order and extensions as L1/L3
   - identical fallback mask across L3 and all three L4 files
   - only one prediction column changed per isolate
   - medians exactly the L1 values in the other two columns
   This is the main thing that can make the additive argument fail despite the metric itself being additive.

5. Reframe the fallback issue correctly. The fallback rows do not make the isolation logic invalid; they make it answer a different question: “What is the net contribution of this target under the current router?” not “Is the underlying estimator good?” If MT-only is bad, the fault could be the MT estimator, the scale, or the gate deciding which rows get MT. Concrete change: every L4 result should be interpreted jointly with the active-row mask and the route that produced it, not as a property of the target routine in isolation.

6. Stop calling `~0.30` a ceiling. From the supplied material, `1.03662` is a measured floor for one narrow model class: global constants at the probed public medians. `0.31898` is a dated public rank cutoff on 2026-09-04, not a stable target. `0.3032` is an expert mean on a different 35-image benchmark, so it is not a justified ceiling on the Kaggle test distribution. Concrete change: separate these three into
   - constant-class baseline: `1.03662`
   - current leaderboard target: `0.31898` as of 2026-09-04
   - external benchmark reference: `0.3032`, explicitly non-binding for test performance

7. Add a public/private warning to the design. The L4 decomposition can be exact on the public subset and still be wrong for the private subset, because it attributes error on about 102 public images only. Given your own SE estimate of about `0.053`, do not let a small public winner decide the private roadmap. Concrete change: require a dominance margin before promoting a rung, not just a rank ordering among L4a/L4b/L4c.

8. Insert a decision table that maps each possible L4 outcome to the next discriminating experiment, not to the next retrain. Right now the ladder risks learning only “target X hurts the public board.” Concrete examples:
   - MT isolate bad: distinguish aponeurosis-pair error from px/cm error before any model change
   - FL isolate bad: distinguish fascicle geometry from endpoint selection before porting DL_Track
   - PA isolate bad: distinguish angle convention / in-muscle QC from detector coverage before SAM2
   Without this table, L4 can consume submissions and still not answer why the pipeline loses to a constant.

9. Add one stopping rule tied to the 2026-10-12 gate: if the three isolate deltas are all small or ambiguous after dilution, do not proceed down L5/L6/L7 as written. That failure mode is the real risk here: the ladder keeps spending quota on increasingly specific fixes after an underpowered diagnostic. Concrete rule:
   `If L4 does not yield either (a) one dominant target contribution or (b) a failed additive identity, pause the ladder and redesign the measurement instrument.`

10. What would change my mind: if the three public isolate deltas sum to `1.44645 - 1.03662` within rounding, and one target accounts for most of that gap by a margin comfortably larger than your pre-registered detectability band, then the isolation design is sufficient for attribution on the public board. The supplied material does not yet show that; it only shows the algebra makes it possible.
tokens used
13,210
1. Keep L4, but rewrite its purpose more narrowly: it is sound for public-score attribution, not for root-cause diagnosis. Checkable from the supplied metric: with the other two columns fixed at the same constants, UMUD is additive over PA, FL, and MT, so on the fixed public subset
   `S(full) - S(constant) = [S(PA only)-S(constant)] + [S(FL only)-S(constant)] + [S(MT only)-S(constant)]`
   up to rounding. If that identity does not hold across L1/L3/L4a/L4b/L4c, the issue is not “which target is bad”; it is CSV construction, fallback routing not held fixed, or a bookkeeping mismatch.

2. Replace each L4 falsifier `public > 1.0366` with an indifference-band rule. With 176/309 fallback rows, only about 133 rows differ from the constant, so an isolate only acts on about 43% of the test set and about 44 public images in expectation. That dilution means “near the floor” is not evidence that the target is fine; it is often evidence that the design had low power. Concrete change:
   `if |S(target only) - 1.03662| < pre-registered minimum detectable delta, classify as uninformative, not exculpatory`
   The current falsifier is too coarse.

3. Pre-compute a maximum possible movement for each isolate before submitting it. This is checkable from the CSVs alone: count rows where the isolate differs from the constant and derive the largest public-score change that target could produce under perfect predictions on those rows. If that upper bound is below your practical detection threshold, the submission is dead on arrival and should not spend quota.

4. Add one explicit “decomposition audit” step before interpreting leaderboard numbers. The audit is mechanical:
   - same `image_id` order and extensions as L1/L3
   - identical fallback mask across L3 and all three L4 files
   - only one prediction column changed per isolate
   - medians exactly the L1 values in the other two columns
   This is the main thing that can make the additive argument fail despite the metric itself being additive.

5. Reframe the fallback issue correctly. The fallback rows do not make the isolation logic invalid; they make it answer a different question: “What is the net contribution of this target under the current router?” not “Is the underlying estimator good?” If MT-only is bad, the fault could be the MT estimator, the scale, or the gate deciding which rows get MT. Concrete change: every L4 result should be interpreted jointly with the active-row mask and the route that produced it, not as a property of the target routine in isolation.

6. Stop calling `~0.30` a ceiling. From the supplied material, `1.03662` is a measured floor for one narrow model class: global constants at the probed public medians. `0.31898` is a dated public rank cutoff on 2026-09-04, not a stable target. `0.3032` is an expert mean on a different 35-image benchmark, so it is not a justified ceiling on the Kaggle test distribution. Concrete change: separate these three into
   - constant-class baseline: `1.03662`
   - current leaderboard target: `0.31898` as of 2026-09-04
   - external benchmark reference: `0.3032`, explicitly non-binding for test performance

7. Add a public/private warning to the design. The L4 decomposition can be exact on the public subset and still be wrong for the private subset, because it attributes error on about 102 public images only. Given your own SE estimate of about `0.053`, do not let a small public winner decide the private roadmap. Concrete change: require a dominance margin before promoting a rung, not just a rank ordering among L4a/L4b/L4c.

8. Insert a decision table that maps each possible L4 outcome to the next discriminating experiment, not to the next retrain. Right now the ladder risks learning only “target X hurts the public board.” Concrete examples:
   - MT isolate bad: distinguish aponeurosis-pair error from px/cm error before any model change
   - FL isolate bad: distinguish fascicle geometry from endpoint selection before porting DL_Track
   - PA isolate bad: distinguish angle convention / in-muscle QC from detector coverage before SAM2
   Without this table, L4 can consume submissions and still not answer why the pipeline loses to a constant.

9. Add one stopping rule tied to the 2026-10-12 gate: if the three isolate deltas are all small or ambiguous after dilution, do not proceed down L5/L6/L7 as written. That failure mode is the real risk here: the ladder keeps spending quota on increasingly specific fixes after an underpowered diagnostic. Concrete rule:
   `If L4 does not yield either (a) one dominant target contribution or (b) a failed additive identity, pause the ladder and redesign the measurement instrument.`

10. What would change my mind: if the three public isolate deltas sum to `1.44645 - 1.03662` within rounding, and one target accounts for most of that gap by a margin comfortably larger than your pre-registered detectability band, then the isolation design is sufficient for attribution on the public board. The supplied material does not yet show that; it only shows the algebra makes it possible.

--- grok (xai) ok ---
1) Isolation is sound for *marginal public attribution* because the UMUD MAE term is additive across PA/FL/MT and the scorer is deterministic: L4x − L1 on the same public fold is an exact public delta for swapping one target, up to the 1e−6/1e−9 MedAE/RMSE terms (negligible at the 0.01 scale you care about). It is not sound as a *causal* diagnosis of why the joint pipeline loses, and it is not a private/full-test attribution (SE ≈ 0.053 applies to that generalisation, not to the public difference itself).

What makes each L4 falsifier (`public > 1.0366`) uninformative:
- Fallback composition: if the ~176/309 fallback rows in `sub_pipeline_a1.csv` are not byte-identical to the L1 probed medians, the delta mixes measurement error with fallback-constant mismatch. Checkable only by diffing those CSVs row-wise (not in the supplied text).
- Public overlap: only ~133 rows carry a non-median measurement; expected public overlap ≈ 133/309 × 102 ≈ 44 images. If the public draw is fallback-heavy, L4x ≈ L1 and the one-sided threshold is a coin flip around machine noise / MedAE dust, not a test of the measured target.
- Near-null outcomes: a score in a band of width ≪ the pre-registered effect (−0.10 to −0.25) neither confirms “MT is the fault” nor “MT is fine”; `> 1.0366` has no “too close to call” band.
- Coupling: scale / crop / pair-selection errors move FL and MT together. A clean MT isolation can pass while the joint failure mode that produced L3 = 1.44645 remains.
- Wrong CSV provenance: L4a reads `mt_mm` from the L3 artifact; if that file’s “measured” rows already include post-hoc clamps or mixed detectors, isolation attributes the artifact, not the named routine.

The PLAN inference that the 133 measured images average worse than the median is directionally implied by L3 vs L1 *if and only if* fallbacks equal the probed medians; that antecedent is asserted, not shown in the attached files.

2) Floor/ceiling framing is partly wrong.
- 1.03662 is a *constant-submission* public floor for this account, validated (A5). It is not the metric floor and not the relevant achievable baseline — BRIEF’s stock DL_Track cluster at 0.45134 is the feasibility reference the kill gate already uses.
- ~0.30 as “honest ceiling” is an OSF 35-image expert noise number (mean expert 0.3032) exported onto Kaggle public scores. A2 is already marked broken in PLAN; that export inherits the break. Treat 0.30 as a benchmark-domain noise floor, not a Kaggle public ceiling.
- 0.31898 is today’s public #3 snapshot (pays on *private* after FAIR). Fine as a moving public reference; not a verified private cutoff (A4 unverified).
- Comparing leader 0.29358 to expert floor 0.3032 to argue public overfit conflates two domains; that argument is not licensed by the supplied evidence once A2 is broken.

3) Failure modes that burn the 2026-10-12 gate without explaining “worse than constant”:
- Spending L4a–c on CSVs whose fallback rows ≠ L1, then reading every `> 1.0366` as “that target is dead,” and promoting L6/L7/L8 on noise or mixed attribution.
- Validating L6 (shape→scale) and L7 (FL port) on the 35-image OSF benchmark while A2 is broken, declaring local falsifiers passed, shipping to LB, still > 0.45134.
- Never measuring the fallback rate *on the public subset* (vs 176/309 global), so isolations with no public measured rows look like “measurement is fine.”
- Treating three single-target isolations as a substitute for one controlled ablation of “measured vs median on the same non-fallback mask,” then retraining the wrong stage (detector vs scale vs FL geometry).
- Consuming the ≤2 ladder slots/day on L4/L5 while the feasibility metric (“best *real pipeline* ≤ 0.45134”) never gets another joint submission after L3.
- Leaving A6 unresolved and later discovering the only ladder that “worked” locally was development-only external data — learning arrives as a compliance failure, not a measurement one.

Concrete changes:
1. Pre-register a mechanical CSV invariant before any L4 submit: on every fallback `image_id`, `(pa_deg, fl_mm, mt_mm)` equals the L1 triple; on measured rows, only the isolated column may differ. Fail the queue item if violated.
2. Replace binary falsifier `public > 1.0366` with a three-way rule on Δ = score(L4x) − 1.03662: Δ > +0.05 → target harmful on public; Δ < −0.05 → target helpful; else → uninformative / no promote (bounds = pre-registered minimum interpretable delta, not SE-of-one-score).
3. Log, from the L3 CSV and the public score alone you cannot recover the public IDs — so add a local proxy: report n_measured and n_fallback in the submitted file, and require n_measured ≥ a pre-set floor (e.g. 100) before treating L4 as powered; if the pipeline’s measured mask is tiny, isolation is parked.
4. Add one null control to the queue: resubmit L1 (or a shuffle of medians that is identical) is useless under a deterministic scorer — instead submit “measured MT only on a random half of measured rows, median elsewhere” only if you need a magnitude check; otherwise skip and spend the slot on a joint pipeline after isolations.
5. Re-anchor “ceiling / honest target” language: keep 1.03662 as constant floor; use 0.45134 as the 2026-10-12 feasibility bar; demote 0.30 to “OSF expert noise, not LB ceiling” until A2 is repaired or explicitly abandoned.
6. Insert a single pre-retrain ablation that PLAN currently skips: score the pipeline with *all three* measured on the non-fallback mask vs all-median on that same mask (one submission or a local reconstruction if you had public GT — you do not). Isolations attribute margins; this tests whether the joint measured triple is the 0.41 excess.
7. Hard-gate L6/L7 promotion on LB evidence from L4, not on OSF benchmark MAE, while A2 is broken; keep OSF as development-only until A6’s human gate closes.
8. Cap isolation spend: at most three L4 slots, then one joint pipeline submission aimed at the feasibility metric before any L8-class retrain — so the gate cannot be consumed by diagnostics alone.
9. Drop or re-derive the “MT is 44% of the constant error budget” claim with an explicit formula and the three constant-sweep scores; it is not derived in the attached PLAN/BRIEF text and should not drive L4a priority until it is.
10. State the fallback→median assumption as a numbered assumption (A11) with a verify-every and a CSV sha256 check; if broken, L3’s “133 images worse than guessing” and all L4 falsifiers are suspended until rebuilt.

Agreement is not confirmation: vendors' errors correlate. Read the dissent, and check anything checkable.
case: 20260904T031902Z_9218b7_panel
