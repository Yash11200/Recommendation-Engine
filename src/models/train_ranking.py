"""
train_ranking.py
----------------
Trains a simple neural ranking model.
We use user embedding + item embedding as input and predict relevance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from src.models.ranking import RankingModel

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
MODEL_DIR = ROOT / "saved_models" / "ranking"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def create_training_data(sample_size=300_000, neg_ratio=1):
  """
  Create positive and negative pairs from training data.
  Actual user-movie interactions are positive (1.0), and random pairs are negative (0.0).
  """
  print("Loading data...")
  train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
  train_df["userId"] = train_df["userId"].astype(str)
  train_df["movieId"] = train_df["movieId"].astype(str)

  # Load embeddings
  item_embeddings = np.load(EMBEDDINGS_DIR / "item_embeddings.npy")
  item_ids = np.load(EMBEDDINGS_DIR / "item_ids.npy", allow_pickle=True)
  item_id_to_idx = {str(mid): idx for idx, mid in enumerate(item_ids)}

  # Filter valid movies and sample interactions
  train_df = train_df[train_df["movieId"].isin(item_id_to_idx)]
  if len(train_df) > sample_size:
    train_df = train_df.sample(n=sample_size, random_state=42)

  # We need user embeddings. For Day 5 we create a simple table.
  # In a full system we would load the trained user tower.
  print("Preparing training pairs...")

  user_ids = train_df["userId"].unique()
  user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}

  # Simple user embedding table (random for demo – replace with real later)
  rng = np.random.default_rng(42)
  user_emb_table = rng.normal(0, 0.1, size=(len(user_ids), 64)).astype("float32")

  # Vectorized lookup for positive pairs
  pos_user_indices = train_df["userId"].map(user_id_to_idx).values
  pos_item_indices = train_df["movieId"].map(item_id_to_idx).values

  # Generate random negative pairs
  num_pos = len(pos_user_indices)
  neg_user_indices = np.repeat(pos_user_indices, neg_ratio)
  neg_item_indices = rng.integers(0, len(item_ids), size=num_pos * neg_ratio)

  # Combine positives and negatives
  all_user_indices = np.concatenate([pos_user_indices, neg_user_indices])
  all_item_indices = np.concatenate([pos_item_indices, neg_item_indices])

  X_user = user_emb_table[all_user_indices]
  X_item = item_embeddings[all_item_indices]
  y = np.concatenate([
    np.ones(num_pos, dtype="float32"),
    np.zeros(num_pos * neg_ratio, dtype="float32")
  ])

  print(f"Training samples: {len(y):,} (Positives: {num_pos:,}, Negatives: {num_pos * neg_ratio:,})")
  return X_user, X_item, y


def train(epochs=5, batch_size=1024):
  print("=" * 60)
  print("Training Ranking Model")
  print("=" * 60)

  X_user, X_item, y = create_training_data()

  model = RankingModel(embedding_dim=64, hidden_units=[128, 64])

  model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
  )

  # Shuffle in NumPy before creating dataset so positives and negatives are mixed
  indices = np.arange(len(y))
  np.random.default_rng(42).shuffle(indices)

  dataset = tf.data.Dataset.from_tensor_slices((
    {"user_embedding": X_user[indices], "item_embedding": X_item[indices]},
    y[indices]
  )).shuffle(10000).batch(batch_size)

  model.fit(dataset, epochs=epochs)

  # Save weights using Keras 3 required extension
  weights_path = MODEL_DIR / "model.weights.h5"
  model.save_weights(str(weights_path))
  print(f"\nRanking model saved to {weights_path}")
  print("Day 5 ranking training finished.")


if __name__ == "__main__":
  train(epochs=3)
