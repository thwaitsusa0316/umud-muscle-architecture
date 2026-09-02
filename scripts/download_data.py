#!/usr/bin/env python3
"""Download the UMUD competition data using the Kaggle OAuth token in ~/.kaggle/credentials.json."""
import json, os, sys, pathlib, urllib.request, shutil

COMP = "umud-challenge-muscle-architecture-in-ultrasound-data"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def token() -> str:
    cred = json.loads((pathlib.Path.home() / ".kaggle" / "credentials.json").read_text())
    return cred["access_token"]


def fetch(url: str, dest: pathlib.Path) -> pathlib.Path:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token()})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e9:6.2f} / {total/1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def main() -> None:
    zip_path = DATA / f"{COMP}.zip"
    if not zip_path.exists():
        print("Downloading competition archive (~6 GB)...")
        fetch(f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}", zip_path)
    print(f"Archive: {zip_path} ({zip_path.stat().st_size/1e9:.2f} GB)")
    raw = DATA / "raw"
    if not raw.exists():
        print("Extracting...")
        shutil.unpack_archive(zip_path, raw)
    print("Done ->", raw)


if __name__ == "__main__":
    main()
