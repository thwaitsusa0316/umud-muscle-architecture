"""Recover pixels-per-centimetre from the B-mode sector height and the printed depth.

Ultrasound consoles annotate the imaging depth on screen ("4.5 cm", "Tiefe 4.0 cm")
and the B-mode sector spans exactly that range from the skin line down, so

    px_per_cm = sector_height_px / depth_cm

This is a far better primary estimator than the ruler tick comb in `ruler.py`. The
sector is a large, high-contrast, high-texture rectangle that is easy to segment on
every console, and the depth string is short, cleanly rendered console text rather
than a low-SNR comb of 1-2 px marks whose pitch, position and major/minor pattern
differ per manufacturer. Verified on two unrelated consoles:

    IMG_00001  607 px / 4.5 cm = 134.9   vs 135.2 from the tick comb  (0.2 %)
    IMG_00252  603 px / 4.0 cm = 150.7   vs 150.x from the ruler labels

`ruler.py` is retained as an independent cross-check: where both fire and agree the
scale is trustworthy, and where they disagree the image needs review.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
import numpy as np
import cv2

_CM = re.compile(r"(\d{1,2})\s*[.,]?\s*(\d)?\s*c\s*m", re.I)


@dataclass
class DepthScale:
    px_per_cm: float | None
    depth_cm: float | None
    sector: tuple[int, int, int, int] | None   # x, y, w, h
    text: str
    source: str                                 # "ocr+sector" | "sector-only" | "none"

    def as_dict(self) -> dict:
        return asdict(self)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """Start and end (exclusive) of the longest contiguous True run."""
    best = cur = None
    best_len = 0
    for i, f in enumerate(flags):
        if f:
            cur = i if cur is None else cur
            if i - cur + 1 > best_len:
                best_len, best = i - cur + 1, (cur, i + 1)
        else:
            cur = None
    return best


def _extend(profile: np.ndarray, run: tuple[int, int] | None, frac: float) -> tuple[int, int] | None:
    """Grow a run outwards while occupancy stays above `frac` of the peak.

    The far field of a B-mode sector is dim but still part of the image, so a
    single high threshold clips the sector short and shrinks the measured depth.
    A low threshold alone would instead bleed into console text, hence: locate the
    bright core first, then extend through the dim tail.
    """
    if run is None:
        return None
    lo, hi = run
    cut = frac * float(profile.max())
    while lo > 0 and profile[lo - 1] > cut:
        lo -= 1
    while hi < len(profile) and profile[hi] > cut:
        hi += 1
    return lo, hi


def find_sector(gray: np.ndarray, floor: int = 12) -> tuple[int, int, int, int] | None:
    """Bounding box of the B-mode image region, by row/column occupancy.

    The sector of a linear-array scan is a solid rectangle of non-black pixels,
    so the fraction of lit pixels per column is high across its width and low
    outside it, where only sparse console text lives. Taking the longest run over
    that profile is robust to the deep part of the sector being dim -- which is
    what defeated a texture- or component-based mask, since low-echo tissue at
    depth breaks the sector into pieces and the bounding box collapses to the
    bright superficial band.
    """
    nz = (gray > floor)
    if not nz.any():
        return None
    col = nz.mean(axis=0)
    row = nz.mean(axis=1)
    cs = _extend(col, _longest_run(col > 0.5 * float(col.max())), 0.15)
    rs = _extend(row, _longest_run(row > 0.5 * float(row.max())), 0.15)
    if cs is None or rs is None:
        return None
    x, w = cs[0], cs[1] - cs[0]
    y, h = rs[0], rs[1] - rs[0]
    if w < 40 or h < 40:
        return None
    return int(x), int(y), int(w), int(h)


def _ocr(region: np.ndarray) -> str:
    """OCR a console region. Console text is light-on-dark; tesseract wants the
    opposite, and these glyphs are small, so invert and upscale first."""
    import pytesseract
    if region.size == 0:
        return ""
    r = cv2.resize(region, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = 255 - r
    r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    try:
        return pytesseract.image_to_string(
            r, config="--psm 11 -c tessedit_char_whitelist=0123456789.,cmTiefeDepth ")
    except Exception:
        return ""


def read_depth_cm(gray: np.ndarray, sector: tuple[int, int, int, int] | None) -> tuple[float | None, str]:
    """Find a '<n>[.<n>] cm' depth annotation in the console furniture.

    Only the border ring is read: blanking the sector stops tissue speckle from
    generating spurious glyphs, and the depth label is always drawn outside it.
    """
    g = gray.copy()
    if sector:
        x, y, w, h = sector
        g[y:y + h, x:x + w] = 0
    text = _ocr(g)
    cands: list[float] = []
    for whole, frac in _CM.findall(text):
        val = float(whole) + (float(frac) / 10 if frac else 0.0)
        if 1.0 <= val <= 15.0:            # plausible musculoskeletal imaging depth
            cands.append(val)
        elif 15.0 < val <= 150.0 and 1.0 <= val / 10 <= 15.0:
            # consoles render the decimal point as a 1 px dot that OCR routinely
            # drops ("4.5 cm" -> "45cm"); recover it when the tenfold reading is
            # implausible as a depth but a tenth of it is not
            cands.append(val / 10)
    if not cands:
        return None, text
    return max(cands), text               # the depth is the largest labelled value


def estimate(img: np.ndarray) -> DepthScale:
    gray = to_gray(img)
    sector = find_sector(gray)
    depth, text = read_depth_cm(gray, sector)
    if sector and depth:
        return DepthScale(px_per_cm=sector[3] / depth, depth_cm=depth,
                          sector=sector, text=text.strip(), source="ocr+sector")
    return DepthScale(px_per_cm=None, depth_cm=depth, sector=sector,
                      text=text.strip(), source="sector-only" if sector else "none")
