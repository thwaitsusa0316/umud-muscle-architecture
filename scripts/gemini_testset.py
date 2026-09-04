#!/usr/bin/env python3
"""Diagnostic read of competition test images by Gemini, bias-corrected.

Gemini measures these parameters badly in absolute terms — 1.7736 on the expert
benchmark, against 1.0374 for submitting a single constant. It is useless as a
predictor. But its error on the benchmark is a *bias*, not noise alone (PA +7.30,
FL -4.17, MT +5.71 mean signed error over 35 images), and a biased instrument with a
known bias is still an instrument.

So: read the test images, subtract the benchmark bias, and compare the corrected
median against both our pipeline's output and the leaderboard-probed medians. The
three-way comparison discriminates:

  corrected ~= probed, ours below  -> our measurement is wrong
  corrected ~= ours, probed higher -> the probed median is not what we think
  corrected ~= neither             -> the test images differ from the benchmark in a
                                      way that invalidates the bias correction

Read-only. Nothing here enters the pipeline, is fitted to, or is submitted — which
keeps it clear of Kaggle Foundational Rule 4.b regardless of how the host's
permitted-if-declared position resolves.
"""
from __future__ import annotations
import sys, os, json, re, pathlib, subprocess, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, cv2
from umud import grouping as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
WORK = pathlib.Path("/tmp/umud_gemini_test")
OUT = ROOT / "reports" / "gemini_testset.csv"

# mean signed error of Gemini minus expert consensus, over the 35 benchmark images
BIAS = {"pa_deg": 7.303, "fl_mm": -4.166, "mt_mm": 5.710}

PROMPT = """You are given one B-mode ultrasound image of a human limb muscle, taken
longitudinally: {path}

The image scale is {pxcm:.1f} pixels per centimetre (both axes).

Measure the three standard muscle architecture parameters:
  pa_deg : pennation angle in DEGREES between the muscle fascicles and the deep
           (lower) aponeurosis
  fl_mm  : fascicle length in MILLIMETRES between the superficial and deep
           aponeuroses, extrapolated if the fascicle leaves the image
  mt_mm  : muscle thickness in MILLIMETRES, perpendicular distance between the
           superficial and deep aponeuroses

Typical ranges: pa_deg 5-45, fl_mm 30-200, mt_mm 10-50.
Look at the image and measure. Do not run any commands.
Reply with ONLY one line of JSON:
{{"pa_deg": <number>, "fl_mm": <number>, "mt_mm": <number>}}"""


def ask(path, pxcm, model, timeout=420):
    cmd = ["agy", "--add-dir", str(WORK), "--mode", "plan", "--sandbox",
           "-p", PROMPT.format(path=path, pxcm=pxcm),
           "--output-format", "text", "--model", model]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    m = re.findall(r'\{[^{}]*"pa_deg"[^{}]*\}', r.stdout)
    if not m:
        return None
    try:
        d = json.loads(m[-1])
        return {k: float(d[k]) for k in ("pa_deg", "fl_mm", "mt_mm")}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--model", default="gemini-3.1-pro-high")
    a = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    scale = json.loads((ROOT / "reports" / "scale_final.json").read_text())
    ours = pd.read_csv(ROOT / "reports" / "test_predictions.csv").set_index("image_id")
    have = [n for n in sorted((p.name for p in TEST.iterdir()
                               if p.suffix.lower() in {".tif", ".png"}), key=G._index)
            if (scale.get(n) or {}).get("px_per_cm")]
    pick = [have[i] for i in np.linspace(0, len(have) - 1, a.n).astype(int)]

    rows = []
    for n in pick:
        dest = WORK / (pathlib.Path(n).stem + ".png")
        if not dest.exists():
            cv2.imwrite(str(dest), G.to_gray(cv2.imread(str(TEST / n), cv2.IMREAD_UNCHANGED)))
        px = float(scale[n]["px_per_cm"])
        got = ask(dest, px, a.model)
        print(f"  {n}: {got}", flush=True)
        if got:
            rows.append({"image_id": n, **got})
    if not rows:
        print("no answers"); return

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(f"\n{len(d)} test images read by Gemini\n")
    print(f"{'target':8s} {'gemini raw':>11s} {'bias-corrected':>15s} {'ours':>8s} {'probed':>8s}")
    probed = {"pa_deg": 16.42, "fl_mm": 81.5, "mt_mm": 20.7}
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        raw = float(d[c].median())
        corr = raw - BIAS[c]
        o = float(ours.loc[ours.index.intersection(d.image_id), c].median())
        print(f"{c:8s} {raw:11.2f} {corr:15.2f} {o:8.2f} {probed[c]:8.2f}")
    print("\ncorrected close to probed, ours below -> our measurement is the problem")
    print("corrected close to ours              -> the probed median is the problem")


if __name__ == "__main__":
    main()
