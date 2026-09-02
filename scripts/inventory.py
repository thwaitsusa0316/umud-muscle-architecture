#!/usr/bin/env python3
"""Inventory the UMUD dataset: shapes, dtypes, img/mask pairing, test-set grouping."""
import pathlib, collections, json
import numpy as np, cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DIRS = {
    "fasc_img": RAW / "fasc_imgs_v1" / "fasc_images_new_model_v1",
    "fasc_mask": RAW / "fasc_masks_v1" / "fasc_masks_new_model_v1",
    "apo_img": RAW / "apo_imgs_v1" / "apo_images_new_model_v1",
    "apo_mask": RAW / "apo_masks_v1" / "apo_masks_new_model_v1",
    "test": RAW / "test_images_v2" / "test_set_v2",
}
IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def listing(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def probe(p):
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    return dict(shape=a.shape, dtype=str(a.dtype), vmin=int(a.min()), vmax=int(a.max()))


def main():
    report = {}
    for key, d in DIRS.items():
        files = listing(d)
        skipped = [p.name for p in d.iterdir() if p.suffix.lower() not in IMG_EXT]
        shapes = collections.Counter()
        dtypes = collections.Counter()
        bad = []
        for p in files:
            info = probe(p)
            if info is None:
                bad.append(p.name); continue
            shapes[info["shape"]] += 1
            dtypes[info["dtype"]] += 1
        report[key] = dict(n=len(files), skipped=skipped, unreadable=bad,
                           shapes=shapes.most_common(12), dtypes=dtypes.most_common())
        print(f"\n=== {key}  ({len(files)} images) dir={d.name}")
        if skipped: print("  non-image files:", skipped)
        if bad: print("  UNREADABLE:", bad[:5], "...total", len(bad))
        print("  dtypes:", dtypes.most_common())
        print("  top shapes:")
        for s, c in shapes.most_common(10):
            print(f"    {s}  x{c}")
        if len(shapes) > 10: print(f"    ... {len(shapes)} distinct shapes total")

    # pairing check between imgs and masks
    for a, b in [("fasc_img", "fasc_mask"), ("apo_img", "apo_mask")]:
        A = {p.stem for p in listing(DIRS[a])}
        B = {p.stem for p in listing(DIRS[b])}
        print(f"\n=== pairing {a} <-> {b}")
        print(f"  img only: {len(A-B)} {sorted(A-B)[:5]}")
        print(f"  mask only: {len(B-A)} {sorted(B-A)[:5]}")
        print(f"  paired: {len(A&B)}")
        mism = []
        for stem in sorted(A & B):
            pa = next(p for p in listing(DIRS[a]) if p.stem == stem)
            pb = next(p for p in listing(DIRS[b]) if p.stem == stem)
            ia, ib = probe(pa), probe(pb)
            if ia and ib and ia["shape"][:2] != ib["shape"][:2]:
                mism.append((stem, ia["shape"], ib["shape"]))
            if len(mism) > 40: break
        print(f"  shape mismatches (first 40 scanned): {len(mism)}")
        for m in mism[:8]: print("   ", m)

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "inventory.json").write_text(
        json.dumps({k: {kk: (str(vv) if kk == "shapes" else vv) for kk, vv in v.items()}
                    for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
