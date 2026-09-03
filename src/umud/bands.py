"""Stage 3: identify which two aponeurosis bands bound the muscle belly.

Stage 2 emits every wide bright band the network found. Choosing the right two is a
recognition problem, not a measurement one, and it is where the geometric rule fails:
the network typically returns skin/superficial fascia, the superficial aponeurosis and
the deep aponeurosis, and "widest, best separated pair" cannot tell the first from the
second. Getting it wrong corrupts all three targets at once.

The model is only ever shown labelled candidates and asked to pick two letters. It
never produces a number. Its answer is validated against a plausible-thickness range
and falls back to the geometric rule when absent, unparseable or implausible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import base64, json, re, urllib.request
import numpy as np
import cv2

OLLAMA = "http://localhost:11434/api/generate"
LETTERS = "ABCDEFGH"

PROMPT = """This is a longitudinal B-mode ultrasound image of a limb muscle.

Several bright horizontal structures have been outlined and labelled with letters.
Typically one of them is the skin or the fatty layer just under it, and two of them
are the aponeuroses that form the top and bottom boundary of the muscle belly. The
muscle belly is the region between those two, filled with fine diagonal striations
(the fascicles).

Which label marks the SUPERFICIAL (upper) aponeurosis of the muscle belly, and which
marks the DEEP (lower) aponeurosis?

Reply with ONLY one line of JSON, no other text:
{"superficial": "<letter>", "deep": "<letter>"}"""


@dataclass
class BandChoice:
    superficial: int | None      # index into the candidate list
    deep: int | None
    source: str                  # "vlm" | "agreement" | "geometry"
    raw: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def candidates(apo_mask: np.ndarray, min_frac: float = 0.15) -> list[dict]:
    """Wide bright structures, ordered top to bottom."""
    n, lbl, stats, _c = cv2.connectedComponentsWithStats(apo_mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] < min_frac * apo_mask.shape[1]:
            continue
        ys, xs = np.where(lbl == i)
        out.append({"y": float(ys.mean()), "ymin": int(ys.min()), "ymax": int(ys.max()),
                    "x0": int(xs.min()), "x1": int(xs.max()), "npix": int(len(ys))})
    return sorted(out, key=lambda d: d["y"])


def render(gray: np.ndarray, cands: list[dict]) -> np.ndarray:
    """Draw each candidate as an outlined box with a letter beside it."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(cands[:len(LETTERS)]):
        y0, y1 = max(c["ymin"] - 3, 0), min(c["ymax"] + 3, img.shape[0] - 1)
        cv2.rectangle(img, (c["x0"], y0), (c["x1"], y1), (0, 230, 255), 2)
        cv2.putText(img, LETTERS[i], (max(c["x0"] - 42, 4), int(c["y"]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 255), 3, cv2.LINE_AA)
    return img


def ask(img_bgr: np.ndarray, model: str, timeout: int = 600) -> tuple[dict | None, str]:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return None, ""
    body = {"model": model, "prompt": PROMPT,
            "images": [base64.b64encode(buf.tobytes()).decode()],
            "stream": False,
            # Qwen3-VL emits a long hidden reasoning block that ollama strips before
            # returning `response`. With a small budget the call reports
            # done_reason="length" and an EMPTY response - the model never reaches
            # the visible answer. Measured: ~1130 tokens consumed before the JSON
            # appears, so the budget has to clear that with room to spare.
            # `think: false` is not honoured on this route.
            # num_ctx: ollama otherwise loads this model at its full 262k context and
            # 44 GB of weights+cache, which does not leave room to work on a 64 GB
            # machine. One image and a short prompt need a tiny fraction of that.
            "options": {"temperature": 0, "num_predict": 2500, "seed": 0,
                        "num_ctx": 8192}}
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        txt = json.load(urllib.request.urlopen(req, timeout=timeout)).get("response", "")
    except Exception as e:
        return None, f"ERR {type(e).__name__}"
    m = re.findall(r'\{[^{}]*"superficial"[^{}]*\}', txt)
    if not m:
        return None, txt[:200]
    try:
        d = json.loads(m[-1])
        return {"superficial": str(d["superficial"]).strip().upper()[:1],
                "deep": str(d["deep"]).strip().upper()[:1]}, txt[:200]
    except Exception:
        return None, txt[:200]


def geometric_choice(cands: list[dict], px_per_cm: float) -> tuple[int, int] | None:
    """The existing rule: the pair whose separation is physiologically plausible."""
    best, pair = None, None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            sep_mm = abs(cands[j]["y"] - cands[i]["y"]) / (px_per_cm / 10.0)
            if not (5.0 <= sep_mm <= 60.0):
                continue
            span = min(cands[i]["x1"], cands[j]["x1"]) - max(cands[i]["x0"], cands[j]["x0"])
            if best is None or span > best:
                best, pair = span, (i, j)
    return pair


def choose(gray: np.ndarray, apo_mask: np.ndarray, px_per_cm: float,
           models: list[str] | None = None) -> tuple[BandChoice, list[dict]]:
    """Pick the muscle-bounding pair, with the geometric rule as the floor.

    With two models, agreement is required to override the geometric rule: a discrete
    choice is exactly the case where two independent voices are informative, and a
    single model being confidently wrong is the failure we are guarding against.
    """
    cands = candidates(apo_mask)
    geo = geometric_choice(cands, px_per_cm)
    fallback = BandChoice(*(geo if geo else (None, None)), source="geometry")
    if len(cands) < 2 or not models:
        return fallback, cands

    canvas = render(gray, cands)
    votes, raws = [], []
    for m in models:
        got, raw = ask(canvas, m)
        raws.append(f"{m}:{raw[:60]}")
        if not got:
            continue
        try:
            s, d = LETTERS.index(got["superficial"]), LETTERS.index(got["deep"])
        except ValueError:
            continue
        if s >= len(cands) or d >= len(cands) or s == d:
            continue
        if s > d:
            s, d = d, s
        sep_mm = abs(cands[d]["y"] - cands[s]["y"]) / (px_per_cm / 10.0)
        if not (5.0 <= sep_mm <= 60.0):      # implausible thickness: reject outright
            continue
        votes.append((s, d))
    if not votes:
        return fallback, cands
    if len(votes) > 1 and len(set(votes)) > 1:
        return fallback, cands              # disagreement -> trust geometry
    src = "agreement" if len(votes) > 1 else "vlm"
    return BandChoice(votes[0][0], votes[0][1], src, " | ".join(raws)), cands
