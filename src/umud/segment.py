"""Inference with the DL_Track_US pretrained aponeurosis / fascicle networks.

Weights: VGG16-U-Net, Apache-2.0, from https://osf.io/7mjsc (DL_Track_US v0.3.0).
Apache-2.0 is one-way compatible with the GPL-3.0 the winning entry must carry.
Used here as the reference baseline our own models have to beat, and to give the
geometry stage something to run on before we have trained anything.

Masks come back at the original image resolution: the networks take 512x512, which
for most of these images is an anisotropic resize, and any angle measured in that
frame is wrong.
"""
from __future__ import annotations

import os
import pathlib
import numpy as np
import cv2

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = pathlib.Path(__file__).resolve().parents[2]
APO = ROOT / "models" / "model-apo-VGG16_DiceBCE_512_300.h5"
FASC = ROOT / "models" / "model-fasc-VGG16_DiceBCE_200_512.h5"
SIZE = 512

_cache: dict[str, object] = {}


def _load(path: pathlib.Path):
    key = str(path)
    if key not in _cache:
        import tensorflow as tf
        _cache[key] = tf.keras.models.load_model(str(path), compile=False)
    return _cache[key]


def _prep(gray: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return (cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            .astype(np.float32) / 255.0)


def predict(gray: np.ndarray, thr_apo: float = 0.5, thr_fasc: float = 0.5
            ) -> tuple[np.ndarray, np.ndarray]:
    """Binary (aponeurosis, fascicle) masks at the input image's resolution."""
    x = _prep(gray)[None, ...]
    h, w = gray.shape[:2]
    out = []
    for path, thr in ((APO, thr_apo), (FASC, thr_fasc)):
        p = np.asarray(_load(path).predict(x, verbose=0))[0, ..., 0]
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append((p > thr).astype(np.uint8))
    return out[0], out[1]


def predict_batch(grays: list[np.ndarray], thr_apo: float = 0.5, thr_fasc: float = 0.5
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batched inference; the per-call overhead on these networks dominates otherwise."""
    x = np.stack([_prep(g) for g in grays])
    preds = {}
    for path in (APO, FASC):
        preds[path] = np.asarray(_load(path).predict(x, verbose=0))[..., 0]
    out = []
    for i, g in enumerate(grays):
        h, w = g.shape[:2]
        a = (cv2.resize(preds[APO][i], (w, h), interpolation=cv2.INTER_LINEAR) > thr_apo).astype(np.uint8)
        f = (cv2.resize(preds[FASC][i], (w, h), interpolation=cv2.INTER_LINEAR) > thr_fasc).astype(np.uint8)
        out.append((a, f))
    return out
