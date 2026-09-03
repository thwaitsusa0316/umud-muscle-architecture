#!/usr/bin/env python3
"""Score Gemini, unguided, on the 35 expert-labelled benchmark images.

The question this answers: could a frontier vision model just do the task, with us
only checking its work? It is scored by exactly the harness that prices our pipeline,
DL_Track and the seven human raters, so the answer is a number on the same scale.

Each image is handed over with its known pixel-per-cm scale and the competition's own
definitions, and nothing else -- no hints about which structures to use, no geometry.
Routed through MAIBO's antigravity engine (gemini-3.1-pro-high) since the gemini CLI
is disabled.
"""
from __future__ import annotations
import sys, os, json, re, pathlib, subprocess, shutil, argparse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from umud import validation as V

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = pathlib.Path("/tmp/umud_gemini")
OUT = ROOT / "reports" / "gemini_benchmark.csv"

PROMPT = """You are given one B-mode ultrasound image of a human limb muscle,
taken longitudinally: {path}

The image scale is {pxcm:.1f} pixels per centimetre (both axes).

Measure the three standard muscle architecture parameters:
  pa_deg : pennation angle, the angle in DEGREES between the muscle fascicles and
           the deep (lower) aponeurosis
  fl_mm  : fascicle length in MILLIMETRES, along a fascicle between the superficial
           and deep aponeuroses, extrapolated if the fascicle leaves the image
  mt_mm  : muscle thickness in MILLIMETRES, the perpendicular distance between the
           superficial and deep aponeuroses

Typical ranges: pa_deg 5-45, fl_mm 30-200, mt_mm 10-50.

Look at the image and measure. Do not run any commands.
Reply with ONLY one line of JSON and nothing else:
{{"pa_deg": <number>, "fl_mm": <number>, "mt_mm": <number>}}"""


def ask(path: pathlib.Path, pxcm: float, model: str, timeout: int = 420) -> dict | None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=35)
    ap.add_argument("--model", default="gemini-3.1-pro-high")
    a = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    t = V.load().head(a.limit)
    rows = []
    for i, r in t.iterrows():
        dest = WORK / f"{r.image_id}.png"
        if not dest.exists():
            import cv2
            img = cv2.imread(r.path, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(dest), img)
        got = ask(dest, float(r.px_per_cm), a.model)
        print(f"  {r.image_id}: {got}", flush=True)
        if got:
            rows.append({"image_id": r.image_id, **got})
    if not rows:
        print("no parseable answers"); return
    p = pd.DataFrame(rows)
    p.to_csv(OUT, index=False)
    truth = V.load()
    print(f"\nGemini answered {len(p)}/{len(t)} images")
    print(f"UMUD score = {V.score(p, truth):.4f}")
    print(V.report(p, truth).round(3).to_string(index=False))
    print("\nreference: ours 0.3349 | DL_Track 0.3306 | expert mean 0.3032 | "
          "best constant 0.7032")


if __name__ == "__main__":
    main()
