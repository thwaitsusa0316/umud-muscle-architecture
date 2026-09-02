"""Locate console ruler ticks as image components and fit a comb to them.

`scale.py` estimates the ruler pitch from the autocorrelation of a border-strip
profile. That is cheap and works where the ruler is the only periodic thing near
the edge, but it cannot tell a ruler from a colour bar or a block of console text
except through a sparsity heuristic, and it throws away the evidence that makes a
ruler unmistakable: the ticks are individual marks of near-identical size, sitting
on a common line, at integer multiples of one pitch.

This module uses that evidence directly. It thresholds the border region, keeps
connected components shaped like tick marks, clusters them onto a common line, and
fits `position = phase + pitch * k` allowing for ticks that were missed. The fit
residual then gives an honest confidence, and gaps in `k` are tolerated rather than
corrupting the estimate the way a missed tick corrupts a nearest-neighbour spacing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import cv2


@dataclass
class Ruler:
    axis: str              # "y": ticks stacked vertically (ruler on a side edge)
                           # "x": ticks in a row (ruler along top/bottom edge)
    pitch_px: float        # spacing between adjacent ticks
    n_ticks: int           # ticks actually detected
    span_ticks: int        # pitches spanned, including gaps
    residual_px: float     # RMS of the comb fit
    line_coord: float      # where the tick line sits on the perpendicular axis
    tick_len_px: float     # median tick length; separates major from minor ticks
    positions: list[float]
    confidence: float

    def as_dict(self) -> dict:
        return asdict(self)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _tick_components(region: np.ndarray, axis: str) -> list[tuple[float, float, float]]:
    """Return (along, across, length) for every component shaped like a tick.

    `along` is the coordinate in the direction the ruler runs, `across` the
    perpendicular one. A tick is a short bar: a few pixels thick along the ruler
    and somewhat longer across it, and solidly filled.
    """
    if region.size == 0:
        return []
    hi = float(region.max())
    if hi < 20:
        return []
    thresh = max(25.0, 0.45 * hi)
    binary = (region > thresh).astype(np.uint8)
    n, _lbl, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if area < 3 or w * h == 0 or area / (w * h) < 0.35:
            continue
        if axis == "y":          # horizontal dash: thin in y, longer in x
            thick, length = h, w
            along, across = cent[i][1], cent[i][0]
        else:                    # vertical dash: thin in x, longer in y
            thick, length = w, h
            along, across = cent[i][0], cent[i][1]
        if not (1 <= thick <= 8 and 2 <= length <= 45):
            continue
        out.append((float(along), float(across), float(length)))
    return out


def _comb_strength(pos: np.ndarray, pitch: float) -> float:
    """How well `pos` lies on a lattice of spacing `pitch`, in [0, 1].

    This is the Rayleigh statistic of the phases: 1 when every position is an
    exact multiple of the pitch (whatever the offset), ~1/sqrt(n) for scattered
    positions. Unlike a nearest-neighbour spacing it is unaffected by ticks that
    were missed, which is the normal case -- consoles draw alternating major and
    minor ticks and the faint ones fall below any threshold.
    """
    if pitch <= 0:
        return 0.0
    ph = 2 * np.pi * (np.asarray(pos) / pitch)
    return float(abs(np.exp(1j * ph).mean()))


def _fit_comb(pos: np.ndarray, min_pitch: float = 6.0,
              strength: float = 0.90) -> tuple[float, float, float]:
    """Recover the fundamental tick pitch from 1-D positions with gaps.

    Scans candidate pitches from coarse to fine and takes the *largest* that still
    explains every position. Taking the largest matters: any integer fraction of
    the true pitch also fits perfectly, so a search that simply maximises fit
    collapses towards zero. Conversely a multiple of the pitch does not fit -- with
    ticks at indices 0,2,3,5,7,9 the doubled period scores only 0.33 -- so the
    largest well-fitting period is the fundamental.
    """
    pos = np.sort(np.asarray(pos, dtype=np.float64))
    if len(pos) < 4:
        return 0.0, np.inf, 0.0
    span = float(pos[-1] - pos[0])
    if span < min_pitch:
        return 0.0, np.inf, 0.0
    grid = np.linspace(min_pitch, span, 4000)
    best_p = 0.0
    for p in grid[::-1]:                       # coarse pass, large pitch first
        if _comb_strength(pos, p) >= strength:
            best_p = float(p)
            break
    if best_p == 0.0:
        return 0.0, np.inf, 0.0
    # refine locally, then snap to integer indices and solve exactly
    fine = np.linspace(best_p * 0.96, best_p * 1.04, 400)
    best_p = float(max(fine, key=lambda p: _comb_strength(pos, p)))
    k = np.round((pos - pos[0]) / best_p)
    if len(np.unique(k)) < 3:
        return 0.0, np.inf, 0.0
    A = np.vstack([k, np.ones_like(k)]).T
    sol, *_ = np.linalg.lstsq(A, pos, rcond=None)
    pitch, phase = float(sol[0]), float(sol[1])
    if pitch <= min_pitch:
        return 0.0, np.inf, 0.0
    resid = float(np.sqrt(np.mean((pos - (phase + pitch * k)) ** 2)))
    return pitch, resid, float(k.max() - k.min())


def find_ruler(gray: np.ndarray, axis: str, edge_frac: float = 0.22,
               line_tol: float = 6.0, min_ticks: int = 4) -> Ruler | None:
    """Best ruler along one axis, searching both edges of the perpendicular one."""
    h, w = gray.shape[:2]
    extent = w if axis == "y" else h     # ruler sits near one end of this axis
    reach = max(30, int(extent * edge_frac))
    best: Ruler | None = None
    for lo, hi in ((0, reach), (extent - reach, extent)):
        region = gray[:, lo:hi] if axis == "y" else gray[lo:hi, :]
        comps = _tick_components(region, axis)
        if len(comps) < min_ticks:
            continue
        along = np.array([c[0] for c in comps])
        across = np.array([c[1] for c in comps]) + lo
        length = np.array([c[2] for c in comps])
        # ticks of one ruler share a common perpendicular coordinate; take the
        # densest such line rather than assuming where the console drew it
        thresh = max(25.0, 0.45 * float(region.max()))
        for centre in np.unique(np.round(across)):
            sel = np.abs(across - centre) <= line_tol
            if sel.sum() < min_ticks:
                continue
            lens = length[sel]
            # ticks are drawn identically; glyph fragments are not
            if lens.mean() <= 0 or lens.std() / lens.mean() > 0.30:
                continue
            # the ruler line is otherwise empty, whereas a column through a text
            # block is full of ink and would otherwise pass the comb test on the
            # periodicity of the text baselines alone
            c0 = int(centre - line_tol) - lo, int(centre + line_tol) - lo
            strip = (region[:, max(c0[0], 0):c0[1]] if axis == "y"
                     else region[max(c0[0], 0):c0[1], :])
            if strip.size == 0:
                continue
            ink = (strip > thresh)
            # consoles rule a solid border line right where the ticks stand; it is
            # not clutter, so drop any fully-drawn line before judging sparsity
            keep = ink.mean(axis=0 if axis == "y" else 1) <= 0.5
            ink = ink[:, keep] if axis == "y" else ink[keep, :]
            if ink.size == 0 or float(ink.mean()) > 0.03:
                continue
            pitch, resid, span = _fit_comb(along[sel])
            if pitch <= 2 or not np.isfinite(resid):
                continue
            # a real ruler fits its comb to within a pixel or two
            conf = float(np.clip(1.0 - resid / max(pitch * 0.08, 1.0), 0.0, 1.0))
            conf *= float(np.clip(sel.sum() / 6.0, 0.0, 1.0))
            if best is None or conf > best.confidence:
                best = Ruler(axis=axis, pitch_px=pitch, n_ticks=int(sel.sum()),
                             span_ticks=int(span), residual_px=resid,
                             line_coord=float(np.median(across[sel])),
                             tick_len_px=float(np.median(length[sel])),
                             positions=sorted(float(v) for v in along[sel]),
                             confidence=conf)
    return best


def estimate(img: np.ndarray) -> dict:
    """Ruler pitch along both axes for one image."""
    gray = to_gray(img)
    out: dict = {"shape": tuple(gray.shape)}
    for axis in ("x", "y"):
        r = find_ruler(gray, axis)
        out[axis] = r.as_dict() if r else None
    return out
