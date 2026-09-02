"""Recover physical scale (pixels per centimetre) from ultrasound ruler ticks.

Why this matters
----------------
Two of the three targets (FL, MT) are in millimetres, so every prediction needs a
px->mm factor. The delivered images have been rescaled from their acquisition
resolution (training images sit at a handful of canonical sizes such as
800x1200 while their masks retain native sizes like 556x996), and an *anisotropic*
rescale changes measured angles as well as lengths. Measuring the tick pitch
separately along x and y lets us both recover the scale and detect any residual
anisotropy, instead of trusting a nominal aspect ratio.

Ultrasound consoles draw a ruler as a row or column of short, evenly spaced
bright marks just outside the B-mode sector. The pitch is a single dominant
spatial frequency, so a windowed autocorrelation of an edge-band profile
recovers it robustly without needing to segment individual ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import cv2


@dataclass
class ScaleEstimate:
    """Tick pitch along one axis, in pixels."""
    axis: str          # "x" (ruler along the bottom/top) or "y" (left/right)
    pitch_px: float    # spacing between adjacent ticks
    confidence: float  # 0..1, autocorrelation peak prominence
    band: tuple[int, int]  # the strip of the image the profile came from
    duty: float        # sparsity of the strip; rulers are thin combs
    n_ticks: int       # how many pitches span the detected ruler

    def as_dict(self) -> dict:
        return asdict(self)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation of a mean-removed signal, normalised to r[0]=1."""
    x = x - x.mean()
    if not np.any(x):
        return np.zeros_like(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    r = np.fft.irfft(f * np.conj(f))[:n]
    # unbiased: divide by the number of overlapping samples at each lag
    r = r / np.maximum(np.arange(n, 0, -1), 1)
    return r / r[0] if r[0] != 0 else r


def _parabolic_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak location by fitting a parabola through i-1, i, i+1."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2 * b + c
    return float(i) if denom == 0 else float(i) + 0.5 * (a - c) / denom


def estimate_pitch(profile: np.ndarray, min_lag: int = 8,
                   max_lag: int | None = None) -> tuple[float, float]:
    """Dominant period of a 1-D profile, via autocorrelation.

    Returns (pitch_px, confidence). Confidence rewards a tall primary peak that
    is corroborated by a peak near twice the lag -- a genuine ruler is periodic
    over many cycles, whereas tissue texture gives a single spurious bump.
    """
    p = profile.astype(np.float64)
    # high-pass: remove slow illumination drift so only the tick comb survives
    p = p - cv2.GaussianBlur(p.reshape(-1, 1), (1, 31), 0).ravel()
    r = _autocorr(p)
    hi = max_lag or len(p) // 4
    hi = min(hi, len(r) - 2)
    if hi <= min_lag:
        return 0.0, 0.0
    seg = r[min_lag:hi]
    i = int(np.argmax(seg)) + min_lag
    peak = float(r[i])
    if peak <= 0:
        return 0.0, 0.0
    pitch = _parabolic_peak(r, i)
    # corroboration from the first harmonic
    j = int(round(2 * pitch))
    harm = float(r[j]) if j < len(r) else 0.0
    conf = max(0.0, peak) * (0.6 + 0.4 * max(0.0, min(1.0, harm / peak)))
    return pitch, float(min(1.0, conf))


def duty_cycle(profile: np.ndarray) -> float:
    """Fraction of the profile sitting in its upper half-range.

    A ruler is a sparse comb of thin marks (duty ~0.01-0.02); a console text
    block is also periodic but dense (duty ~0.13-0.15) and otherwise outscores
    the real ruler on autocorrelation alone. This is the discriminator.
    """
    p = np.asarray(profile, dtype=np.float64)
    span = p.max() - p.min()
    if span <= 0:
        return 1.0
    return float((p > p.min() + 0.5 * span).mean())


def _band_profile(gray: np.ndarray, axis: str, lo: int, hi: int) -> np.ndarray:
    """Collapse a border strip into a 1-D profile along the ruler direction."""
    if axis == "x":                      # ruler runs left-right; strip is rows lo:hi
        band = gray[lo:hi, :]
        return band.max(axis=0).astype(np.float64)
    band = gray[:, lo:hi]                # ruler runs top-bottom; strip is cols lo:hi
    return band.max(axis=1).astype(np.float64)


def scan_axis(gray: np.ndarray, axis: str, strip: int = 14,
              search_frac: float = 0.18, max_duty: float = 0.06) -> ScaleEstimate | None:
    """Slide a thin strip along both borders of one axis and keep the most periodic.

    We do not know a priori which edge carries the ruler (Siemens draws it on the
    right, other consoles along the bottom), nor exactly where the console UI ends,
    so we score every candidate strip and take the best.
    """
    h, w = gray.shape[:2]
    extent = h if axis == "x" else w      # thickness direction
    reach = max(strip * 2, int(extent * search_frac))
    candidates: list[tuple[float, ScaleEstimate]] = []
    for lo in list(range(0, reach, strip // 2)) + list(range(extent - reach, extent - strip, strip // 2)):
        hi = lo + strip
        if lo < 0 or hi > extent:
            continue
        prof = _band_profile(gray, axis, lo, hi)
        d = duty_cycle(prof)
        if d > max_duty:                  # dense band: console text, not a ruler
            continue
        pitch, conf = estimate_pitch(prof)
        if pitch >= 8 and conf > 0:
            span = (w if axis == "x" else h)
            est = ScaleEstimate(axis=axis, pitch_px=pitch, confidence=conf,
                                band=(lo, hi), duty=d,
                                n_ticks=int(span // max(pitch, 1)))
            candidates.append((conf, est))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def estimate_scale(img: np.ndarray) -> dict:
    """Estimate tick pitch along x and y for one image.

    The host confirmed that unlabelled ticks can be assumed 1 cm apart, so a
    detected pitch converts directly to px/cm. Consoles that label their ruler in
    2/5/10 mm steps need the label read too -- handled separately by the OCR path.
    """
    gray = to_gray(img)
    out: dict = {"shape": gray.shape}
    for axis in ("x", "y"):
        est = scan_axis(gray, axis)
        out[axis] = est.as_dict() if est else None
    px_x = out["x"]["pitch_px"] if out["x"] else None
    px_y = out["y"]["pitch_px"] if out["y"] else None
    out["anisotropy"] = (px_x / px_y) if (px_x and px_y) else None
    return out
