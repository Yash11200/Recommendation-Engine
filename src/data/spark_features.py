"""
spark_features.py
-----------------
Uses PySpark (local mode) to build useful offline features:

1. Item popularity (number of ratings + average rating)
2. User activity (number of ratings given by each user)
3. Genre-level statistics
4. Simple user history length

These features will later feed the ranking model and cold-start logic.

Run after preprocess.py:
    python -m src.data.spark_features
"""

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def create_spark():
  """Lightweight local Spark session – good enough for this project size."""
  spark = (
    SparkSession.builder
    .appName("RecSys-FeatureEngineering")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.ui.showConsoleProgress", "true")
    .getOrCreate()
  )
  spark.sparkContext.setLogLevel("WARN")
  return spark


def main():
  print("=" * 60)
  print("Spark Feature Engineering")
  print("=" * 60)

  spark = create_spark()

  # ------------------------------------------------------------------
  # Load processed data
  # ------------------------------------------------------------------
  train_path = str(PROCESSED_DIR / "train.parquet")
  movies_path = str(PROCESSED_DIR / "movies.parquet")

  print("Loading train.parquet into Spark...")
  train = spark.read.parquet(train_path)
  movies = spark.read.parquet(movies_path)

  print(f"  Train rows : {train.count():,}")
  print(f"  Movies     : {movies.count():,}")

  # ------------------------------------------------------------------
  # 1. Item popularity features
  # ------------------------------------------------------------------
  print("\nComputing item popularity...")
  item_stats = (
    train.groupBy("movieId")
    .agg(
      F.count("*").alias("rating_count"),
      F.avg("rating").alias("avg_rating"),
      F.stddev("rating").alias("std_rating"),
    )
    .withColumn("log_rating_count", F.log1p("rating_count"))
  )

  # ------------------------------------------------------------------
  # 2. User activity features
  # ------------------------------------------------------------------
  print("Computing user activity...")
  user_stats = (
    train.groupBy("userId")
    .agg(
      F.count("*").alias("user_rating_count"),
      F.avg("rating").alias("user_avg_rating"),
      F.min("timestamp").alias("first_rating_ts"),
      F.max("timestamp").alias("last_rating_ts"),
    )
    .withColumn("user_log_count", F.log1p("user_rating_count"))
  )

  # ------------------------------------------------------------------
  # 3. Genre explosion + genre popularity (simple)
  # ------------------------------------------------------------------
  print("Computing genre features...")
  # movies already has genres_list from preprocess (array of strings)
  movies_exploded = (
    movies
    .withColumn("genre", F.explode_outer("genres_list"))
    .select("movieId", "genre")
  )

  genre_stats = (
    train.join(movies_exploded, on="movieId", how="inner")
    .groupBy("genre")
    .agg(
      F.count("*").alias("genre_rating_count"),
      F.avg("rating").alias("genre_avg_rating"),
    )
  )

  # ------------------------------------------------------------------
  # 4. Join everything back and write
  # ------------------------------------------------------------------
  print("Writing feature tables...")

  item_stats.write.mode("overwrite").parquet(str(PROCESSED_DIR / "item_features.parquet"))
  user_stats.write.mode("overwrite").parquet(str(PROCESSED_DIR / "user_features.parquet"))
  genre_stats.write.mode("overwrite").parquet(str(PROCESSED_DIR / "genre_features.parquet"))

  # Also create a convenient train + item features view
  train_enriched = train.join(item_stats, on="movieId", how="left")
  train_enriched.write.mode("overwrite").parquet(
    str(PROCESSED_DIR / "train_enriched.parquet")
  )

  print("\n[OK] Feature tables saved:")
  print(f"  • item_features.parquet")
  print(f"  • user_features.parquet")
  print(f"  • genre_features.parquet")
  print(f"  • train_enriched.parquet")

  # Quick preview
  print("\nTop 10 most popular movies (by rating count):")
  item_stats.orderBy(F.desc("rating_count")).show(10, truncate=False)

  spark.stop()
  print("\nSpark feature engineering finished.")


if __name__ == "__main__":
  main()
