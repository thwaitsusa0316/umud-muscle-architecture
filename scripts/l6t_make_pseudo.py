#!/usr/bin/env python3
"""L6t-b step 1: pseudo-label the 309 test frames with the shipped pw6 fascicle U-Net.

Runs ONLY after human gate 3 (A18, PLAN v32) is answered "permitted": it reads the
competition's own test images and writes a training-style image/mask pair per frame
under --out (imgs/<stem>.png, masks/<stem>.png, 0/255). No external labels, no hand
labelling: every mask is the model's own prediction (Foundational Rule 4.b is about
human labelling of the test records, which this is not).

Mask thickness is matched to the host labels: the host fascicle masks are 1-2 px
lines (~0.15 % positive at native size) that train_fascicle.py dilates twice before
the 512 resize. A raw sigmoid > thr mask is 3-6 px wide at 512, so it is
skeletonised to 1 px at 512 first and then upsampled (nearest) to the native grid;
the trainer's dilate then gives the same ~2 px line at 512 that the host labels get.

Frames whose skeleton covers < --min-frac or > --max-frac of the 512 grid are
skipped (empty or blown-out predictions teach nothing useful); the list is written
to --summary so the count is auditable (pre-registered: >= 250 of 309 kept, else
the pseudo set is too thin and L6t-b is not launched).
"""
from __future__ import annotations
import sys, os, json, pathlib, argparse
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, cv2
from skimage.morphology import skeletonize

TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
EXT = {".tif", ".tiff", ".png"}
SIZE = 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "fasc_unet_pw6_best.pt"))
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--thr", type=float, default=0.9, help="shipped production threshold")
    ap.add_argument("--out", default=str(ROOT / "data" / "pseudo" / "l6t"))
    ap.add_argument("--summary", default=str(ROOT / "reports" / "l6t" / "pseudo_summary.json"))
    ap.add_argument("--min-frac", type=float, default=0.0005)
    ap.add_argument("--max-frac", type=float, default=0.02)
    ap.add_argument("--min-keep", type=int, default=250)
    ap.add_argument("--test-dir", default=str(TEST))
    a = ap.parse_args()

    from eval_new_fascicle import load_model
    import torch
    model, dev = load_model(a.arch, a.ckpt)
    test_dir = pathlib.Path(a.test_dir)
    names = sorted(p.name for p in test_dir.iterdir() if p.suffix.lower() in EXT)
    if not names:
        print(f"ERROR: no test images under {test_dir}", file=sys.stderr)
        return 2
    out = pathlib.Path(a.out)
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    # A rerun must not train on stale pairs from an earlier pass (a frame kept then may be
    # skipped now): clear both output subdirs before writing anything.
    n_stale = 0
    for sub in ("imgs", "masks"):
        for old in (out / sub).glob("*.png"):
            old.unlink(); n_stale += 1
    if n_stale:
        print(f"cleared {n_stale} stale files under {out}")
    rows, kept = [], 0
    for n in names:
        img = cv2.imread(str(test_dir / n), cv2.IMREAD_UNCHANGED)
        if img is None:
            rows.append({"image": n, "kept": False, "reason": "unreadable"}); continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = gray.shape[:2]
        x = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = np.repeat((x.astype(np.float32) / 255.0)[None], 3, 0)[None]
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
        binm = p > a.thr
        skel = skeletonize(binm).astype(np.uint8)
        frac = float(skel.mean())
        keep = a.min_frac <= frac <= a.max_frac
        rows.append({"image": n, "kept": bool(keep), "frac512": round(frac, 6),
                     "raw_frac512": round(float(binm.mean()), 6)})
        if not keep:
            continue
        native = cv2.resize(skel * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        stem = pathlib.Path(n).stem
        cv2.imwrite(str(out / "imgs" / f"{stem}.png"), gray)
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), native)
        kept += 1
    n_img = len(list((out / "imgs").glob("*.png"))); n_msk = len(list((out / "masks").glob("*.png")))
    if n_img != kept or n_msk != kept:
        print(f"ERROR: on-disk pair count imgs={n_img} masks={n_msk} != n_kept={kept}", file=sys.stderr)
        return 2
    summ = {"ckpt": a.ckpt, "arch": a.arch, "thr": a.thr, "n_test": len(names), "n_kept": kept,
            "min_keep": a.min_keep, "min_frac": a.min_frac, "max_frac": a.max_frac,
            "out": str(out), "n_pairs_on_disk": n_img, "rows": rows}
    sp = pathlib.Path(a.summary); sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summ, indent=1))
    print(f"pseudo-labelled {kept}/{len(names)} test frames -> {out} (summary {sp})")
    if kept < a.min_keep:
        print(f"ERROR: kept {kept} < min_keep {a.min_keep}: pseudo set too thin, do not launch L6t-b",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
