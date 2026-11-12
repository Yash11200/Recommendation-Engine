"""
download.py
-----------
Downloads the official MovieLens 25M dataset from GroupLens
and extracts it into data/raw/ml-25m/

Run once:
    python -m src.data.download
"""

import os
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

# Project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "ml-25m.zip"
EXTRACT_DIR = RAW_DIR / "ml-25m"

# Official source
URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_file(url: str, dest: Path):
  """Download a large file with a progress bar."""
  dest.parent.mkdir(parents=True, exist_ok=True)

  if dest.exists():
    print(f"[OK] {dest.name} already exists – skipping download.")
    return

  print(f"Downloading {url} ...")
  response = requests.get(url, stream=True)
  response.raise_for_status()

  total = int(response.headers.get("content-length", 0))
  with open(dest, "wb") as f, tqdm(
      total=total, unit="B", unit_scale=True, desc=dest.name
  ) as bar:
    for chunk in response.iter_content(chunk_size=8192):
      if chunk:
        f.write(chunk)
        bar.update(len(chunk))

  print(f"[OK] Downloaded → {dest}")


def extract_zip(zip_path: Path, extract_to: Path):
  """Extract the zip if the target folder does not exist yet."""
  if extract_to.exists() and any(extract_to.iterdir()):
    print(f"[OK] {extract_to} already extracted – skipping.")
    return

  print(f"Extracting {zip_path.name} ...")
  with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_to.parent)
  print(f"[OK] Extracted → {extract_to}")


def main():
  print("=" * 60)
  print("MovieLens 25M – Download & Extract")
  print("=" * 60)

  download_file(URL, ZIP_PATH)
  extract_zip(ZIP_PATH, EXTRACT_DIR)

  # Quick sanity check
  ratings = EXTRACT_DIR / "ratings.csv"
  movies = EXTRACT_DIR / "movies.csv"
  if ratings.exists() and movies.exists():
    print("\nFiles ready:")
    print(f"  • {ratings}")
    print(f"  • {movies}")
    print("\nYou can now run: python -m src.data.preprocess")
  else:
    print("ERROR: Expected files not found after extraction.")


if __name__ == "__main__":
  main()
