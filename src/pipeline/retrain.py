"""
retrain.py
----------
Simple end-to-end retrain pipeline.
This can be called by Airflow or run manually.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_step(args: list[str], description: str):
  print(f"\n>>> {description}")
  cmd = [sys.executable, "-m"] + args
  print(f"Running: {' '.join(cmd)}")

  start_time = time.time()
  result = subprocess.run(cmd, cwd=ROOT)
  elapsed = time.time() - start_time

  if result.returncode != 0:
    print(f"ERROR in step: {description} (failed after {elapsed:.1f}s)")
    sys.exit(1)

  print(f"✓ {description} completed in {elapsed:.1f}s")


def main():
  print("=" * 60)
  print("Full Retrain Pipeline")
  print("=" * 60)

  pipeline_start = time.time()

  # 1. Preprocess (optional – only if new data arrived)
  # run_step(["src.data.preprocess", "--sample", "2000000"], "Preprocess data")

  # 2. Spark features
  run_step(["src.data.spark_features"], "Spark feature engineering")

  # 3. Train Two-Tower
  run_step(["src.models.train_retrieval"], "Train Two-Tower retrieval model")

  # 4. Build FAISS
  run_step(["src.retrieval.faiss_index"], "Build FAISS index")

  # 5. Train Ranking
  run_step(["src.models.train_ranking"], "Train Ranking model")

  total_elapsed = time.time() - pipeline_start
  print("\n" + "=" * 60)
  print(f"Full pipeline completed successfully in {total_elapsed:.1f}s!")
  print("=" * 60)


if __name__ == "__main__":
  main()
