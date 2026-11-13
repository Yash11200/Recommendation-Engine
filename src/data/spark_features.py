"""
spark_features.py (Fast Pandas Alternative)
-------------------------------------------
Builds offline item, user, and genre features without requiring a JVM.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def main():
  print("=" * 60)
  print("Feature Engineering (Fast Pandas Mode)")
  print("=" * 60)

  train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
  movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")

  print(f"  Train rows : {len(train):,}")
  print(f"  Movies     : {len(movies):,}")

  # 1. Item popularity features
  print("\nComputing item popularity...")
  item_stats = (
    train.groupby("movieId", as_index=False)
    .agg(
      rating_count=("rating", "count"),
      avg_rating=("rating", "mean"),
      std_rating=("rating", "std"),
    )
  )
  item_stats["std_rating"] = item_stats["std_rating"].fillna(0.0)
  item_stats["log_rating_count"] = np.log1p(item_stats["rating_count"])

  # 2. User activity features
  print("Computing user activity...")
  user_stats = (
    train.groupby("userId", as_index=False)
    .agg(
      user_rating_count=("rating", "count"),
      user_avg_rating=("rating", "mean"),
      first_rating_ts=("timestamp", "min"),
      last_rating_ts=("timestamp", "max"),
    )
  )
  user_stats["user_log_count"] = np.log1p(user_stats["user_rating_count"])
  user_stats["user_history_span_sec"] = (
    user_stats["last_rating_ts"] - user_stats["first_rating_ts"]
  )

  # 3. Genre popularity features
  print("Computing genre features...")
  if "genres_list" not in movies.columns:
    movies["genres_list"] = movies["genres"].fillna("(no genres listed)").str.split("|")

  movies_exploded = movies[["movieId", "genres_list"]].explode("genres_list").rename(
    columns={"genres_list": "genre"}
  )
  genre_stats = (
    train[["movieId", "rating"]]
    .merge(movies_exploded, on="movieId", how="inner")
    .groupby("genre", as_index=False)
    .agg(
      genre_rating_count=("rating", "count"),
      genre_avg_rating=("rating", "mean"),
    )
  )

  # 4. Write feature tables
  print("Writing feature tables...")
  item_stats.to_parquet(PROCESSED_DIR / "item_features.parquet", index=False)
  user_stats.to_parquet(PROCESSED_DIR / "user_features.parquet", index=False)
  genre_stats.to_parquet(PROCESSED_DIR / "genre_features.parquet", index=False)

  train_enriched = train.merge(item_stats, on="movieId", how="left")
  train_enriched.to_parquet(PROCESSED_DIR / "train_enriched.parquet", index=False)

  print("\n[OK] Feature tables saved:")
  print("  • item_features.parquet")
  print("  • user_features.parquet")
  print("  • genre_features.parquet")
  print("  • train_enriched.parquet")

  print("\nTop 10 most popular movies (by rating count):")
  print(item_stats.sort_values("rating_count", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
  main()
