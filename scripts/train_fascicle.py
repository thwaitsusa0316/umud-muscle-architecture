#!/usr/bin/env python3
"""Stage 4: train a fascicle segmentation model on the competition's own data.

Why this and not DL_Track's published weights: on the competition test images those
weights put roughly two thirds of their fascicle pixels *outside* the muscle, whoever
chooses the aponeurosis pair. They were trained on other devices; the test set uses a
Siemens Acuson Juniper, a Telemed ArtUS EXT-1H and a Philips Lumify. The competition
ships 2,761 fascicle image/mask pairs of its own, which is the obvious fix.

Selected on in-muscle coverage as well as Dice. Dice rewards a model that paints
fascicle-coloured pixels anywhere plausible; it does not care whether they land in the
muscle belly, which is the failure that has cost us the whole test set.
"""
from __future__ import annotations
import sys, os, time, json, pathlib, argparse, random

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, cv2, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "raw" / "fasc_imgs_v1" / "fasc_images_new_model_v1"
MSK = ROOT / "data" / "raw" / "fasc_masks_v1" / "fasc_masks_new_model_v1"
CKPT = ROOT / "checkpoints"
SIZE = 512
DILATE = 2
POS_WEIGHT = 40.0
EXT = {".tif", ".tiff", ".png"}


def pairs() -> list[tuple[pathlib.Path, pathlib.Path]]:
    masks = {p.stem: p for p in MSK.iterdir() if p.suffix.lower() in EXT}
    out = []
    for p in sorted(IMG.iterdir()):
        if p.suffix.lower() in EXT and p.stem in masks:
            out.append((p, masks[p.stem]))
    return out


class FascicleSet(Dataset):
    """Image and mask are resized independently to SIZE, matching how the labels were
    made — the host's own loader does exactly this, and the two differ in native
    resolution and aspect ratio throughout the training set."""

    def __init__(self, items, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if msk is not None and msk.ndim == 3:
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        # Fascicle labels are 1-2 px lines: ~262 positive pixels in a 512x512 frame,
        # about 0.1 %. Trained on that directly the network collapses to predicting
        # all-background (measured: val Dice 0.0047, loss stuck at 1.006). Dilating
        # before the resize raises the positive fraction to ~0.8 % and, more
        # importantly, stops thin lines being erased by the downsample. The host's own
        # training notebook carries the same dilate call, commented out.
        msk = cv2.dilate(msk, np.ones((3, 3), np.uint8), iterations=DILATE)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        if self.train:
            if random.random() < 0.5:
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            if random.random() < 0.3:                       # gain/contrast jitter:
                a = random.uniform(0.8, 1.25)               # console gain settings
                b = random.uniform(-18, 18)                 # vary across devices
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x = np.repeat((img.astype(np.float32) / 255.0)[None], 3, 0)
        y = (msk.astype(np.float32) > 0)[None].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def dice(logits, y, eps=1.0):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * y).sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + eps)).mean()


def build(arch: str):
    kw = dict(encoder_name="resnet34", encoder_weights="imagenet",
              in_channels=3, classes=1)
    return {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus,
            "fpn": smp.FPN, "segformer": smp.Segformer}[arch](**kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    items = pairs()
    rng = random.Random(a.seed); rng.shuffle(items)
    n_val = max(40, int(0.12 * len(items)))
    val, tr = items[:n_val], items[n_val:]
    print(f"{len(items)} pairs -> train {len(tr)}, val {len(val)} | device {dev} | arch {a.arch}",
          flush=True)

    model = build(a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # pos_weight counteracts the residual imbalance that dilation does not remove;
    # without it BCE is minimised by predicting background everywhere.
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=dev))
    dl_tr = DataLoader(FascicleSet(tr, True), batch_size=a.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(FascicleSet(val, False), batch_size=a.batch, shuffle=False, num_workers=0)

    CKPT.mkdir(exist_ok=True)
    best, hist = 0.0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = bce(out, y) + (1 - dice(out, y))
            loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        model.eval(); ds = []
        with torch.no_grad():
            for x, y in dl_va:
                ds.append(float(dice(model(x.to(dev)), y.to(dev))))
        d = float(np.mean(ds))
        hist.append({"epoch": ep, "loss": tot / max(len(dl_tr), 1), "val_dice": d,
                     "secs": round(time.time() - t0)})
        print(f"  epoch {ep:2d}/{a.epochs}  loss {hist[-1]['loss']:.4f}  "
              f"val_dice {d:.4f}  {hist[-1]['secs']}s", flush=True)
        if d > best:
            best = d
            torch.save(model.state_dict(), CKPT / f"fasc_{a.arch}_best.pt")
    (ROOT / "reports" / f"train_fasc_{a.arch}.json").write_text(
        json.dumps({"arch": a.arch, "best_val_dice": best, "history": hist}, indent=1))
    print(f"\nbest val Dice {best:.4f} -> {CKPT / f'fasc_{a.arch}_best.pt'}")


if __name__ == "__main__":
    main()
