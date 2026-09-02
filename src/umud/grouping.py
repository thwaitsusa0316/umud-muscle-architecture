"""Group test images by acquisition, so scale is solved per group not per image.

The overview states the test set includes stretches of consecutive video frames
("5 images in a row"). Frames from one acquisition share a console layout, a depth
setting and therefore an identical pixel-to-millimetre scale, and their true PA / FL
/ MT barely move between adjacent frames. Both facts are worth a lot:

  * scale detection only has to succeed on *some* frame of a group, and a robust
    consensus over the group then covers the frames where the ruler was missed;
  * predictions can be pooled across a group, which cuts variance without needing a
    better model.

Two groupings are computed. `chrome` clusters on the static console furniture in
the border ring -- text, ruler, logos -- which is constant for one device and depth
setting and independent of the moving anatomy. `runs` splits those clusters into
maximal stretches of consecutive image ids whose full frames actually resemble each
other, which is the tighter notion needed for temporal pooling.
"""
from __future__ import annotations

import re
import numpy as np
import cv2


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def chrome_signature(img: np.ndarray, size: int = 48, border: float = 0.16) -> np.ndarray:
    """Fingerprint the static console furniture around the B-mode sector.

    Keeps only the border ring, so moving tissue in the middle of the frame cannot
    change the signature, and binarises it so gain differences do not either.
    """
    g = to_gray(img).astype(np.float32)
    h, w = g.shape
    bh, bw = max(1, int(h * border)), max(1, int(w * border))
    ring = g.copy()
    ring[bh:h - bh, bw:w - bw] = 0                      # blank the sector
    ring = cv2.resize(ring, (size, size), interpolation=cv2.INTER_AREA)
    hi = ring.max()
    if hi <= 0:
        return np.zeros(size * size, dtype=np.float32)
    v = (ring > 0.35 * hi).astype(np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def frame_signature(img: np.ndarray, size: int = 64) -> np.ndarray:
    """Whole-frame fingerprint, for deciding whether two frames are adjacent."""
    g = to_gray(img).astype(np.float32)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    g = g - g.mean()
    n = np.linalg.norm(g)
    return (g / n).ravel() if n > 0 else g.ravel()


def _index(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def cluster_chrome(sigs: dict[str, np.ndarray], thresh: float = 0.90) -> dict[str, int]:
    """Greedy single-pass clustering on cosine similarity of chrome signatures."""
    names = sorted(sigs, key=_index)
    centres: list[np.ndarray] = []
    counts: list[int] = []
    label: dict[str, int] = {}
    for n in names:
        v = sigs[n]
        best, best_s = -1, thresh
        for i, c in enumerate(centres):
            s = float(np.dot(v, c) / (np.linalg.norm(c) or 1.0))
            if s > best_s:
                best, best_s = i, s
        if best < 0:
            centres.append(v.copy())
            counts.append(1)
            label[n] = len(centres) - 1
        else:
            centres[best] = (centres[best] * counts[best] + v) / (counts[best] + 1)
            counts[best] += 1
            label[n] = best
    return label


def find_runs(frames: dict[str, np.ndarray], chrome: dict[str, int],
              thresh: float = 0.72) -> dict[str, int]:
    """Split into maximal runs of consecutive ids that look like the same take."""
    names = sorted(frames, key=_index)
    run: dict[str, int] = {}
    cur = 0
    run[names[0]] = cur
    for prev, cur_name in zip(names, names[1:]):
        contiguous = _index(cur_name) == _index(prev) + 1
        same_chrome = chrome.get(cur_name) == chrome.get(prev)
        sim = float(np.dot(frames[cur_name], frames[prev]))
        if not (contiguous and same_chrome and sim >= thresh):
            cur += 1
        run[cur_name] = cur
    return run


def consensus(values: dict[str, float], groups: dict[str, int],
              rel_tol: float = 0.05) -> dict[str, float]:
    """Propagate a robust per-group value to every member of the group.

    Uses the median of the confident members, then discards members that disagree
    with it by more than `rel_tol` before taking the final median, so one bad
    detection in a group cannot drag the whole group off.
    """
    by_group: dict[int, list[float]] = {}
    for name, g in groups.items():
        if name in values and np.isfinite(values[name]):
            by_group.setdefault(g, []).append(values[name])
    out: dict[str, float] = {}
    resolved: dict[int, float] = {}
    for g, vals in by_group.items():
        med = float(np.median(vals))
        keep = [v for v in vals if abs(v - med) <= rel_tol * max(med, 1e-9)]
        resolved[g] = float(np.median(keep)) if keep else med
    for name, g in groups.items():
        if g in resolved:
            out[name] = resolved[g]
    return out
