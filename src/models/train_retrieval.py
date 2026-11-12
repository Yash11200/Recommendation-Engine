"""
train_retrieval.py
------------------
Trains the Two-Tower retrieval model on the processed MovieLens data.
Saves the trained model and the item embeddings for FAISS later.
"""

import os

# Must be set BEFORE importing tensorflow or tensorflow_recommenders
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Optional: hides the oneDNN info logs

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

from src.models.two_tower import UserModel, ItemModel, TwoTowerModel

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
MODEL_DIR = ROOT / "saved_models" / "two_tower"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(sample_size: int = None):
    """Load train data and prepare TensorFlow datasets."""
    print("Loading train.parquet...")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    # Optional: further reduce size for faster experiments
    if sample_size is not None and len(train_df) > sample_size:
        print(f"Sampling {sample_size:,} rows for faster training...")
        train_df = train_df.sample(n=sample_size, random_state=42)

    # Convert IDs to string (required by StringLookup)
    train_df["user_id"] = train_df["userId"].astype(str)
    train_df["movie_id"] = train_df["movieId"].astype(str)

    # Unique vocabularies
    unique_user_ids = np.unique(train_df["user_id"])
    unique_movie_ids = np.unique(train_df["movie_id"])

    print(f"Users  : {len(unique_user_ids):,}")
    print(f"Movies : {len(unique_movie_ids):,}")
    print(f"Interactions : {len(train_df):,}")

    # TensorFlow dataset
    ratings = tf.data.Dataset.from_tensor_slices({
        "user_id": train_df["user_id"].values,
        "movie_id": train_df["movie_id"].values,
    })

    # Shuffle and batch
    ratings = ratings.shuffle(100_000).batch(4096).cache()

    return ratings, unique_user_ids, unique_movie_ids


def train(
    embedding_dim: int = 64,
    epochs: int = 5,
    learning_rate: float = 0.1,
    sample_size: int = None,
):
    print("=" * 60)
    print("Training Two-Tower Retrieval Model")
    print("=" * 60)

    ratings, unique_user_ids, unique_movie_ids = load_data(sample_size)

    # Build towers
    user_model = UserModel(unique_user_ids, embedding_dim)
    item_model = ItemModel(unique_movie_ids, embedding_dim)

    # Metrics and task
    metrics = tfrs.metrics.FactorizedTopK(
        candidates=tf.data.Dataset.from_tensor_slices(unique_movie_ids)
        .batch(1024)
        .map(item_model)
    )

    task = tfrs.tasks.Retrieval(metrics=metrics)

    model = TwoTowerModel(user_model, item_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))

    # Train
    print("\nStarting training...")
    model.fit(ratings, epochs=epochs)

    # -------------------------------------------------
    # Save the full model
    # -------------------------------------------------
    print("\nSaving model...")
    model.save_weights(str(MODEL_DIR / "weights"))
    print(f"Model weights saved to {MODEL_DIR}")

    # -------------------------------------------------
    # Extract and save item embeddings (needed for FAISS)
    # -------------------------------------------------
    print("Extracting item embeddings...")
    item_ids = tf.constant(unique_movie_ids)
    item_embeddings = item_model(item_ids).numpy()

    np.save(EMBEDDINGS_DIR / "item_embeddings.npy", item_embeddings)
    np.save(EMBEDDINGS_DIR / "item_ids.npy", unique_movie_ids)

    print(f"Item embeddings shape : {item_embeddings.shape}")
    print(f"Saved to {EMBEDDINGS_DIR}")

    print("\nDay 2 training finished successfully!")


if __name__ == "__main__":
    # For Day 2 we start with a smaller sample so training finishes quickly.
    # Later you can remove sample_size or increase it.
    train(
        embedding_dim=64,
        epochs=3,               # increase to 5–8 later
        learning_rate=0.1,
        sample_size=1_500_000,  # good balance for Day 2
    )