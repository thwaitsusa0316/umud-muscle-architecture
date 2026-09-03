# Hybrid pipeline design

## The one rule

**The vision model never produces a number.** It only makes discrete choices between
candidates that deterministic code has already generated. Every millimetre and degree
comes from geometry.

This is not a stylistic preference, it is what the measurements say. Scored on the 35
expert-labelled images by the same harness that prices everything else:

| | PA MAE | FL MAE | MT MAE | UMUD |
|---|---|---|---|---|
| our geometry | 1.16° | 5.06 mm | 1.17 mm | **0.3349** |
| DL_Track | 1.45° | 3.74 mm | 1.32 mm | 0.3306 |
| expert mean | ~1.4° | ~5.1 mm | ~0.7 mm | 0.3032 |
| best constant | — | — | — | 0.7032 |
| **Gemini 3.1 Pro, unguided** | **7.20°** | **24.0 mm** | **3.17 mm** | **1.42** |

A frontier vision model asked to measure directly is *worse than submitting a single
constant for every image*. Asked instead to say which of three detected bright bands
is the deep aponeurosis, it agreed with our detector to within seven pixels
(y≈290/580 vs 283/584). Identification it can do; metrology it cannot.

## Why a hybrid at all

The pipeline scores 0.3349 on the benchmark and 1.4653 on the competition test set.
Two defects cause the gap, and neither is a measurement problem:

1. **Aponeurosis pair selection.** The network returns three or more bright
   horizontal bands — skin/superficial fascia, the superficial aponeurosis, the deep
   aponeurosis, sometimes a bone echo. Our rule takes the widest, best-separated pair
   and has no way to know which two actually bound the muscle. Getting this wrong
   corrupts all three targets at once: thickness spans the pair, pennation angle is
   measured against the deep member, fascicle length depends on both.
2. **Fascicle segmentation collapses on the test domain.** On one console screenshot
   the network put 3,029 fascicle pixels outside the muscle and 284 inside, while the
   belly was covered in plainly visible striations. DL_Track's weights were trained on
   other devices; the test set uses a Siemens Acuson Juniper, a Telemed ArtUS EXT-1H
   and a Philips Lumify.

Both are *recognition* failures. That is exactly the shape of problem a vision model
is good at and hand-written geometric rules are bad at — and exactly the shape that
does not require the model to be numerate.

## Stages

```
image
  │
  ├─ 1  SCALE                 deterministic CV + OCR          px per cm
  ├─ 2  APONEUROSIS CANDIDATES CNN                            N bright bands
  ├─ 3  BAND IDENTIFICATION   local VLM, discrete choice      which two bound the muscle
  ├─ 4  FASCICLE MASK         locally retrained U-Net         fascicle pixels
  ├─ 5  GEOMETRY              deterministic                   PA, FL, MT
  └─ 6  GROUP CONSENSUS       deterministic                   pooled over video runs
```

### 1. Scale — deterministic
`px_per_cm = sector_height / depth_cm`, the depth read by OCR from the console
annotation, cross-checked against the ruler tick comb which resolves the 0.5 vs 1 cm
tick ambiguity and contributes sub-pixel precision. Verified to +0.3 % and −2.6 % on
two unrelated consoles. Where no console text exists, fall back to the per-acquisition
group consensus, since frames of one take share a scale by construction.

*Open:* 176 of 309 test images still have no recoverable scale. Ground truth on the
benchmark takes only seven distinct values, and image shape nearly determines it, so
this should become a lookup over device configurations rather than a per-image
estimate.

### 2. Aponeurosis candidates — deterministic
Segment, keep connected structures spanning ≥15 % of the image width, fit each with a
quadratic. Emit *all* of them as candidates rather than choosing here. Cheap, and it
is the recall step: it must not miss the true pair, and it currently does not.

### 3. Band identification — local VLM, discrete choice
Render the image with each candidate band outlined and labelled `A`, `B`, `C`…, and
ask **Qwen3-VL 30B** (local, open weights, ~9 s/image measured):

> Which label marks the superficial aponeurosis of the muscle belly, and which marks
> the deep aponeurosis? Answer with two labels only.

Constraints that make this safe:
* the answer space is a handful of labels, so the model cannot invent a value;
* a refusal or unparseable answer falls back to the current geometric rule;
* an answer whose implied thickness is outside 5–60 mm is rejected outright;
* it is **validated on the benchmark**, where the correct pair is known from the
  expert thickness — if it does not beat the geometric rule there, it does not ship.

Local rather than Gemini: winning entries are re-run by the organizers from a public
GPL-3.0 repository under a FAIR check. Open weights with a fixed seed are
reproducible; a proprietary API is neither reproducible nor deterministic.

### 4. Fascicle mask — retrained locally
Train on the competition's own 2,761 fascicle image/mask pairs, which cover the test
devices in a way DL_Track's published weights do not. This is the local-GPU stage and
the place for a genuine tournament: U-Net/VGG16, SegFormer and UNet3+ ranked on
held-out Dice **and** on downstream PA/FL/MT error, because Dice alone does not say
whether the fascicles land inside the muscle.

Add a coverage metric to the ranking: *fraction of predicted fascicle pixels lying
between the two aponeuroses*. It needs no labels, runs on the real test images, and is
the statistic that exposed this failure in the first place.

### 5. Geometry — deterministic, conventions already derived
* thickness: mean vertical separation at three sites, measured **inner edge to inner
  edge** — the raters measure the muscle belly, and following both centrelines instead
  overstates thickness by one aponeurosis (8.5 %, ratio IQR 0.028);
* fascicle length: traced to the aponeurosis **centrelines**, the opposite convention,
  because a fascicle runs into the aponeurosis rather than stopping at its face;
* angle: median over fragments **inside the muscle only** — this was the bug that made
  the benchmark and the test set disagree;
* discard fascicles whose extrapolations cross, per Ritsche et al. 2024;
* fascicle length = mean of the trigonometric estimate and the traced estimate, whose
  biases are equal and opposite (−3.92 and +4.55 mm) and largely cancel (5.90 mm).

### 6. Group consensus — deterministic
27 runs of exactly five frames were identified in the test set. Within a run the scale
is identical and the targets barely move, so pool the per-frame estimates. Free
variance reduction, and it repairs frames where an earlier stage failed.

## Reproducibility: the VLM decisions are cached, not re-run

Winning entries are re-run by the organizers from the public repository. A vision
model on the inference path is a liability there: the tag `qwen3-vl:30b` moves, the
weights are a 19 GB download, ~20 GB of RAM is needed, and even at temperature 0 a
different quantisation or accelerator can change the answer. If the re-run does not
reproduce our numbers we lose the placing, not the argument.

So **stage 3 emits a committed artifact**: one row per image recording which band was
chosen as superficial and which as deep, plus the model digest and prompt that
produced it. The pipeline reads that file. `scripts/regenerate_band_choices.py`
rebuilds it from the images for anyone who wants to verify, but nothing on the
critical path needs a GPU or a model download, and the re-run is deterministic.

This is the general rule for any model we add: **non-deterministic components produce
artifacts, artifacts go in the repository, and the pipeline consumes artifacts.**

## Model inventory — measured, not assumed

| model | vision | on our ultrasound images | role |
|---|---|---|---|
| `qwen3-vl:30b` (19 GB) | yes | reads console text correctly, ~9-16 s/image | stage 3, primary |
| `gemma4:31b` (19 GB) | yes | to be gated | stage 3, second voice |
| `gemma4:latest` (8B) | declares vision | returns **empty** — unusable | — |
| `deepseek-r1:32b` | **no** — HTTP 400 on any image | rejects images | none |
| `llama4` (67 GB) | yes | will not co-reside with training in 64 GB | none |
| Gemini 3.1 Pro (via MAIBO) | yes | 1.42 on the benchmark measuring directly; located aponeuroses to 7 px | advisor / diagnosis only |

Two voices at stage 3 are worth having because the choice is discrete: agreement
raises confidence, disagreement falls back to geometry. Qwen and Gemma are different
vendors, which is the point -- two models from one family are one blind spot with
temperature. But a second voice must clear the benchmark gate **alone** before it is
allowed to vote; otherwise one good model becomes two mediocre ones.

Qwen3-VL misread "vl rechts" (vastus lateralis, right) as "likely a vein". Domain
knowledge is weak, which is tolerable only because stage 3 asks it to pick between
outlined candidates rather than to name anatomy.

## Candidates worth adding

**SAM2 (Meta, Apache-2.0) — the strongest single addition, and not a chat model.**
Two properties matter here:

* It segments from a *prompt* (a point or box), so the aponeurosis pair problem
  becomes "put a point on the muscle belly and take the boundary", rather than
  "classify three bands after the fact".
* It propagates masks through **video**. The test set contains 27 runs of exactly five
  consecutive frames. SAM2 can segment frame one and track the aponeuroses through the
  rest, which is both more stable than per-frame segmentation and exactly the
  structure the host pointed at in the overview.

Published comparisons find SAM2 generalises to unseen modalities better than MedSAM,
which was fine-tuned on a medical corpus that does not include this task; SAMUSA
adapts SAM2 to ultrasound specifically with boundary prompts. Apache-2.0 is one-way
compatible with our GPL-3.0.

**nnU-Net** remains the right trainable baseline for stage 4 alongside the
architectures already planned -- it is the standard against which medical segmentation
methods are judged, and it self-configures, which removes a tuning axis we would
otherwise have to search by hand.

**Not worth adding:** more general-purpose chat VLMs. The bottleneck is segmentation
quality and structure identification, not language. A second VLM only earns a place at
stage 3 as an independent vote.

## What must be true for this to be worth building

Each stage has a gate, checked on the benchmark or label-free on the test set:

| stage | gate |
|---|---|
| 3 band identification | beats the geometric rule at picking the pair on the 35 benchmark images; each voice must pass alone before it may vote |
| SAM2 (if adopted) | prompted aponeurosis masks beat the current CNN on benchmark UMUD, and its video propagation beats per-frame within the 27 five-frame runs |
| 4 fascicle model | raises in-muscle coverage on test images above DL_Track's, and does not regress benchmark UMUD |
| whole pipeline | predicted medians approach the leaderboard-probed PA 16.4°, FL 81.5 mm, MT 20.7 mm |

If stage 3 fails its gate the VLM is dropped and the geometric rule stays. Nothing here
depends on the vision model being good at anything except recognition.

## Explicitly rejected

* **VLM measures the parameters directly** — 1.42 on the benchmark, worse than a
  constant.
* **Anisotropy correction (aspect ≈ 1.35)** — fitted to leaderboard medians derived
  from broken segmentation; the control on native-resolution images rejects it
  decisively. Withdrawn.
* **Per-target calibration gains** — improved in-sample (0.4241) but leave-one-out was
  worse (0.4512). 35 images cannot support even three fitted scalars.
* **Manual annotation of the 309 test images** — the host permits it if declared, but
  Kaggle Foundational Rule 4.b prohibits it and the rules text is still unchanged. The
  ceiling is the expert noise floor either way, so the upside is small and the downside
  is disqualification.
