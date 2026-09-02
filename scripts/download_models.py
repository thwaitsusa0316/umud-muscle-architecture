#!/usr/bin/env python3
"""Fetch the DL_Track_US pretrained segmentation weights.

The weights are 310 MB each, over GitHub's per-file limit, so they are not in the
repository. They are published by the challenge host under Apache-2.0 at
https://osf.io/7mjsc (DL_Track_US v0.3.0) -- Apache-2.0 is one-way compatible with
the GPL-3.0 this project carries.
"""
from __future__ import annotations
import pathlib, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FILES = {
    "model-apo-VGG16_DiceBCE_512_300.h5": "https://osf.io/download/925xp/",
    "model-fasc-VGG16_DiceBCE_200_512.h5": "https://osf.io/download/pfb6c/",
}


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = MODELS / name
        if dest.exists():
            print(f"have {name}")
            continue
        print(f"downloading {name} ...")
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
        print(f"  -> {dest} ({dest.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
