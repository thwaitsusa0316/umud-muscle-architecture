"""Submission helpers for the UMUD challenge.

Two format traps, both confirmed empirically:
  * the host's sample_submission.csv is semicolon-delimited with a BOM, but the
    grader wants plain comma-delimited UTF-8;
  * image_id must carry the true per-file extension (251 .tif, 58 .png).
"""
from __future__ import annotations
import json, pathlib, time, urllib.request, urllib.parse, mimetypes, uuid
import pandas as pd

COMP = "umud-challenge-muscle-architecture-in-ultrasound-data"
ROOT = pathlib.Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "data" / "raw" / "test_images_v2" / "test_set_v2"
API = "https://www.kaggle.com/api/v1"


def token() -> str:
    cred = json.loads((pathlib.Path.home() / ".kaggle" / "credentials.json").read_text())
    return cred["access_token"]


def _req(url, data=None, headers=None, method=None):
    h = {"Authorization": "Bearer " + token()}
    h.update(headers or {})
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read().decode())


def test_ids() -> list[str]:
    """The 309 test image_ids, with correct per-file extensions."""
    return sorted(p.name for p in TEST_DIR.iterdir()
                  if p.suffix.lower() in {".tif", ".tiff", ".png"})


def write_submission(df: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    """Validate and write a submission in the exact format the host expects."""
    need = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    assert list(df.columns) == need, f"columns must be {need}, got {list(df.columns)}"
    ids = test_ids()
    assert len(df) == len(ids) == 309, f"expected 309 rows, got {len(df)}"
    missing = set(ids) - set(df.image_id)
    extra = set(df.image_id) - set(ids)
    assert not missing and not extra, f"id mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
    assert df[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all(), "NaNs in predictions"
    path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: the host's sample_submission.csv is semicolon-delimited with a BOM, but
    # Kaggle's grader parses plain comma-delimited UTF-8 ("ID column image_id not found"
    # otherwise). Always write commas, no BOM.
    df.to_csv(path, sep=",", index=False, encoding="utf-8", float_format="%.4f")
    return path


def submit(path: pathlib.Path, message: str) -> None:
    """Upload a submission file to Kaggle via the official client."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    api.competition_submit(file_name=str(path), message=message, competition=COMP)


def latest_submissions(n: int = 10) -> pd.DataFrame:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    subs = api.competition_submissions(COMP)
    rows = [{"date": str(getattr(s, "date", "")), "desc": getattr(s, "description", ""),
             "status": str(getattr(s, "status", "")),
             "public": getattr(s, "public_score", None),
             "private": getattr(s, "private_score", None)} for s in subs]
    return pd.DataFrame(rows).head(n)
