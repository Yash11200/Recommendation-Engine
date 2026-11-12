"""
preprocess.py
-------------
Loads MovieLens 25M, performs a clean time-based split
(train / validation / test), and writes Parquet files.

Important: We NEVER split randomly. We respect the global timeline
so that the model is trained only on past interactions.

Usage:
    python -m src.data.preprocess
    python -m src.data.preprocess --sample 2000000   # for faster iteration
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "ml-25m"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_raw():
  """Load ratings and movies from the extracted MovieLens folder."""
  ratings_path = RAW_DIR / "ratings.csv"
  movies_path = RAW_DIR / "movies.csv"

  if not ratings_path.exists():
    raise FileNotFoundError(
      f"{ratings_path} not found. Run: python -m src.data.download first"
    )

  print("Loading ratings.csv (this can take ~30-60s for full 25M)...")
  ratings = pd.read_csv(
    ratings_path,
    dtype={
      "userId": "int32",
      "movieId": "int32",
      "rating": "float32",
      "timestamp": "int64",
    },
  )

  print("Loading movies.csv...")
  movies = pd.read_csv(movies_path)

  print(f"  Ratings : {len(ratings):,} rows")
  print(f"  Movies  : {len(movies):,} rows")
  print(f"  Users   : {ratings['userId'].nunique():,}")
  print(f"  Items   : {ratings['movieId'].nunique():,}")

  return ratings, movies


def time_based_split(ratings: pd.DataFrame, val_ratio=0.1, test_ratio=0.1):
  """
  Global temporal split.
  - Sort everything by timestamp
  - Last `test_ratio` fraction → test
  - Previous `val_ratio` fraction → validation
  - Everything before that → train

  This prevents any future leakage.
  """
  print("\nPerforming global time-based split...")
  ratings = ratings.sort_values("timestamp").reset_index(drop=True)

  n = len(ratings)
  test_size = int(n * test_ratio)
  val_size = int(n * val_ratio)

  test = ratings.iloc[-test_size:].copy()
  val = ratings.iloc[-(test_size + val_size) : -test_size].copy()
  train = ratings.iloc[: -(test_size + val_size)].copy()

  print(f"  Train : {len(train):,}  ({train['timestamp'].min()} → {train['timestamp'].max()})")
  print(f"  Val   : {len(val):,}  ({val['timestamp'].min()} → {val['timestamp'].max()})")
  print(f"  Test  : {len(test):,}  ({test['timestamp'].min()} → {test['timestamp'].max()})")

  return train, val, test


def add_basic_features(ratings: pd.DataFrame, movies: pd.DataFrame):
  """
  Join movie metadata and create a few simple features
  that will be useful later for ranking / cold-start.
  """
  # Keep only needed movie columns
  movies = movies[["movieId", "title", "genres"]].copy()

  # Simple genre list (pipe-separated → list)
  movies["genres_list"] = movies["genres"].str.split("|")

  # Merge
  df = ratings.merge(movies, on="movieId", how="left")

  # Item popularity (will be recalculated properly with Spark later)
  # For now just a placeholder column – real popularity computed offline
  return df


def save_splits(train, val, test, movies, sample_size=None):
  """Write Parquet files. Optionally subsample for fast development."""
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

  if sample_size is not None and sample_size < len(train):
    print(f"\n[Dev mode] Sampling {sample_size:,} training interactions...")
    # Keep the most recent sample so temporal order is preserved
    train = train.tail(sample_size).copy()

  print("\nSaving Parquet files...")
  train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
  val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
  test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
  movies.to_parquet(PROCESSED_DIR / "movies.parquet", index=False)

  print(f"[OK] Files written to {PROCESSED_DIR}")
  print(f"     train.parquet  → {len(train):,} rows")
  print(f"     val.parquet    → {len(val):,} rows")
  print(f"     test.parquet   → {len(test):,} rows")
  print(f"     movies.parquet → {len(movies):,} rows")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--sample",
    type=int,
    default=None,
    help="Optional: keep only the last N training interactions (fast iteration)",
  )
  args = parser.parse_args()

  print("=" * 60)
  print("MovieLens 25M – Time-based Preprocessing")
  print("=" * 60)

  ratings, movies = load_raw()
  train, val, test = time_based_split(ratings)

  # Add title & genres
  train = add_basic_features(train, movies)
  val = add_basic_features(val, movies)
  test = add_basic_features(test, movies)

  save_splits(train, val, test, movies, sample_size=args.sample)

  print("\nDay 1 data pipeline finished successfully.")
  print("Next: python -m src.data.spark_features")


if __name__ == "__main__":
  main()
