"""
recommender.py
--------------
Core recommendation logic:
- Load Two-Tower models + FAISS
- Handle normal users and cold-start users
- Return ranked movie recommendations
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
from typing import List, Dict

from src.models.two_tower import UserModel, ItemModel

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
INDEX_DIR = ROOT / "data" / "faiss_index"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "saved_models" / "two_tower"


class RecommenderService:
  def __init__(self):
    print("Loading models and index...")

    # Load FAISS
    self.index = faiss.read_index(str(INDEX_DIR / "item_index.faiss"))
    self.item_ids = np.load(INDEX_DIR / "item_ids.npy", allow_pickle=True)

    # Load item embeddings
    self.item_embeddings = np.load(EMBEDDINGS_DIR / "item_embeddings.npy").astype("float32")

    # Load movie metadata
    self.movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")
    self.movies["movieId"] = self.movies["movieId"].astype(str)
    self.movie_dict = self.movies.set_index("movieId").to_dict("index")

    # Load training data for user vocabulary
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    train_df["userId"] = train_df["userId"].astype(str)
    train_df["movieId"] = train_df["movieId"].astype(str)
    self.unique_user_ids = np.unique(train_df["userId"])
    self.unique_movie_ids = self.item_ids

    # Popularity fallback (for cold-start)
    item_features_path = PROCESSED_DIR / "item_features.parquet"
    if item_features_path.exists():
      item_features = pd.read_parquet(item_features_path)
      item_features["movieId"] = item_features["movieId"].astype(str)
      self.popular_movies = (
        item_features
        .sort_values("rating_count", ascending=False)
        .head(50)["movieId"]
        .tolist()
      )
    else:
      print("item_features.parquet not found; computing popularity from train.parquet...")
      self.popular_movies = (
        train_df["movieId"]
        .value_counts()
        .head(50)
        .index
        .tolist()
      )

    # Rebuild user and item towers
    self.user_model = UserModel(self.unique_user_ids, embedding_dim=64)
    self.item_model = ItemModel(self.unique_movie_ids, embedding_dim=64)

    # Load trained weights if saved
    user_weights_path = MODEL_DIR / "user_model.weights.h5"
    if user_weights_path.exists():
      self.user_model.load_weights(str(user_weights_path))
      print("Loaded trained user model weights.")

    print("Models and FAISS index loaded successfully.")

  def is_known_user(self, user_id: str) -> bool:
    return str(user_id) in self.unique_user_ids

  def get_user_embedding(self, user_id: str) -> np.ndarray:
    user_tensor = tf.constant([user_id])
    emb = self.user_model(user_tensor).numpy()
    return emb

  def retrieve(self, user_embedding: np.ndarray, top_k: int = 100) -> List[str]:
    """Retrieve top-k candidates using FAISS."""
    user_embedding = user_embedding.astype("float32")
    faiss.normalize_L2(user_embedding)

    distances, indices = self.index.search(user_embedding, top_k)
    candidate_ids = self.item_ids[indices[0]].tolist()
    return candidate_ids

  def format_recommendations(self, movie_ids: List[str], limit: int = 10) -> List[Dict]:
    """Convert movie IDs to nice response objects."""
    results = []
    for mid in movie_ids[:limit]:
      info = self.movie_dict.get(str(mid), {})
      results.append({
        "movie_id": str(mid),
        "title": info.get("title", "Unknown"),
        "genres": info.get("genres", "")
      })
    return results

  def recommend(self, user_id: str, top_k: int = 10) -> List[Dict]:
    """
    Main recommendation method.
    Handles both known users and cold-start users.
    """
    user_id = str(user_id)

    # ---------- Cold-start handling ----------
    if not self.is_known_user(user_id):
      print(f"Cold-start user detected: {user_id}")
      return self.format_recommendations(self.popular_movies, top_k)

    # ---------- Normal user ----------
    user_emb = self.get_user_embedding(user_id)
    candidates = self.retrieve(user_emb, top_k=100)

    return self.format_recommendations(candidates, top_k)
