#!/usr/bin/env python3
"""Does the retrained fascicle model actually put fascicles inside the muscle?

Val Dice says the model reproduces held-out training labels. It does not say the
predictions land in the muscle belly on the competition's own images, which is the
failure that cost us the test set. This measures that directly:

    in-muscle coverage = fascicle pixels between the aponeuroses / all fascicle pixels

Baseline to beat, measured with DL_Track's published fascicle weights on the same
images and the same aponeurosis pair: median 0.326, mean 0.510.
"""
from __future__ import annotations
import sys, os, json, pathlib, argparse, warnings

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cv2, torch
import segmentation_models_pytorch as smp
from umud import segment as S, bands as B, grouping as G

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
SIZE = 512


def load_model(arch="unet", ckpt=None):
    m = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN,
         "segformer": smp.Segformer}[arch](   # same table as train_fascicle.build
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    p = ckpt or (ROOT / "checkpoints" / f"fasc_{arch}_best.pt")
    m.load_state_dict(torch.load(p, map_location="cpu"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return m.to(dev).eval(), dev


def predict_ours(model, dev, gray, thr=0.5):
    h, w = gray.shape[:2]
    x = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x = np.repeat((x.astype(np.float32) / 255.0)[None], 3, 0)[None]
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return (cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) > thr).astype(np.uint8)


def coverage(fasc, cands, pair):
    if not pair or pair[0] is None:
        return None
    lo, hi = sorted((cands[pair[0]]["y"], cands[pair[1]]["y"]))
    tot = int((fasc > 0).sum())
    if tot == 0:
        return None
    return float((fasc[int(lo):int(hi)] > 0).sum()) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--arch", default="unet")
    a = ap.parse_args()

    model, dev = load_model(a.arch)
    scale = json.loads((ROOT / "reports" / "scale_final.json").read_text())
    names = sorted((p.name for p in TEST.iterdir()
                    if p.suffix.lower() in {".tif", ".png"}), key=G._index)
    rows = []
    for i in range(0, len(names), 24):
        if len(rows) >= a.limit:
            break
        chunk = names[i:i + 24]
        grays = [G.to_gray(cv2.imread(str(TEST / n), cv2.IMREAD_UNCHANGED)) for n in chunk]
        masks = S.predict_batch(grays, thr_apo=0.35, thr_fasc=0.10)   # apo + DL_Track fasc
        for n, g, (am, fm_dl) in zip(chunk, grays, masks):
            if len(rows) >= a.limit:
                break
            px = (scale.get(n) or {}).get("px_per_cm")
            if not px:
                continue
            cands = B.candidates(am)
            pair = B.geometric_choice(cands, float(px))
            if not pair:
                continue
            fm_ours = predict_ours(model, dev, g)
            rows.append({"image_id": n,
                         "cov_dltrack": coverage(fm_dl, cands, pair),
                         "cov_ours": coverage(fm_ours, cands, pair),
                         "px_dltrack": int((fm_dl > 0).sum()),
                         "px_ours": int((fm_ours > 0).sum())})
            print(f"  {n} dl={rows[-1]['cov_dltrack']} ours={rows[-1]['cov_ours']}", flush=True)

    d = pd.DataFrame(rows).dropna(subset=["cov_dltrack", "cov_ours"])
    d.to_csv(ROOT / "reports" / "coverage_new_model.csv", index=False)
    print(f"\ntest images compared: {len(d)}   (higher coverage is better, max 1.0)")
    print(f"  DL_Track weights   median {d.cov_dltrack.median():.3f}   mean {d.cov_dltrack.mean():.3f}")
    print(f"  our retrained model median {d.cov_ours.median():.3f}   mean {d.cov_ours.mean():.3f}")
    better = int((d.cov_ours > d.cov_dltrack).sum())
    print(f"  ours better on {better}/{len(d)} images")
    print(f"\n  fascicle pixels found: DL_Track median {int(d.px_dltrack.median()):,}, "
          f"ours median {int(d.px_ours.median()):,}")
    verdict = "PASS" if d.cov_ours.median() > d.cov_dltrack.median() + 0.05 else "FAIL"
    print(f"\nGATE: {verdict} (needs median coverage up by >0.05 over 0.326 baseline)")


if __name__ == "__main__":
    main()
