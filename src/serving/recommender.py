"""
recommender.py
--------------
Core recommendation logic with optional ranking.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import faiss
from typing import List, Dict

from src.models.two_tower import UserModel, ItemModel
from src.models.ranking import RankingModel

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
INDEX_DIR = ROOT / "data" / "faiss_index"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "saved_models"


class RecommenderService:
  def __init__(self):
    print("Loading models and index...")

    self.index = faiss.read_index(str(INDEX_DIR / "item_index.faiss"))
    self.item_ids = np.load(INDEX_DIR / "item_ids.npy", allow_pickle=True)
    self.item_id_to_idx = {str(mid): i for i, mid in enumerate(self.item_ids)}
    self.item_embeddings = np.load(EMBEDDINGS_DIR / "item_embeddings.npy").astype("float32")

    self.movies = pd.read_parquet(PROCESSED_DIR / "movies.parquet")
    self.movies["movieId"] = self.movies["movieId"].astype(str)
    self.movie_dict = self.movies.set_index("movieId").to_dict("index")

    # Popularity for cold-start
    item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")
    item_features["movieId"] = item_features["movieId"].astype(str)
    self.popular_movies = (
      item_features.sort_values("rating_count", ascending=False)
      .head(50)["movieId"].tolist()
    )

    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    self.unique_user_ids = np.unique(train_df["userId"].astype(str))

    self.user_model = UserModel(self.unique_user_ids, embedding_dim=64)
    self.item_model = ItemModel(self.item_ids, embedding_dim=64)

    # Ranking model
    self.ranking_model = RankingModel(embedding_dim=64)
    ranking_weights = MODEL_DIR / "ranking" / "model.weights.h5"

    if ranking_weights.exists():
      try:
        # Build model weights before loading in Keras 3
        dummy_input = {
          "user_embedding": tf.zeros((1, 64), dtype=tf.float32),
          "item_embedding": tf.zeros((1, 64), dtype=tf.float32)
        }
        self.ranking_model(dummy_input)
        self.ranking_model.load_weights(str(ranking_weights))
        self.has_ranking = True
        print("Ranking model loaded.")
      except Exception as e:
        self.has_ranking = False
        print(f"Failed to load ranking weights ({e}) – using retrieval only.")
    else:
      self.has_ranking = False
      print("Ranking model not trained yet – using retrieval only.")

    print("RecommenderService ready.")

  def is_known_user(self, user_id: str) -> bool:
    return str(user_id) in self.unique_user_ids

  def get_user_embedding(self, user_id: str) -> np.ndarray:
    return self.user_model(tf.constant([str(user_id)])).numpy()

  def retrieve(self, user_embedding: np.ndarray, top_k: int = 100) -> List[str]:
    user_embedding = user_embedding.astype("float32")
    faiss.normalize_L2(user_embedding)
    distances, indices = self.index.search(user_embedding, top_k)
    return self.item_ids[indices[0]].tolist()

  def rank_candidates(self, user_emb: np.ndarray, candidate_ids: List[str]) -> List[str]:
    """Re-rank candidates using the ranking model."""
    if not self.has_ranking:
      return candidate_ids

    item_embs = []
    valid_ids = []

    for mid in candidate_ids:
      idx = self.item_id_to_idx.get(str(mid))
      if idx is not None:
        item_embs.append(self.item_embeddings[idx])
        valid_ids.append(mid)

    if not item_embs:
      return candidate_ids

    item_embs = np.array(item_embs, dtype="float32")
    user_emb_repeated = np.repeat(user_emb, len(item_embs), axis=0)

    scores = self.ranking_model({
      "user_embedding": user_emb_repeated,
      "item_embedding": item_embs
    }).numpy().flatten()

    ranked_indices = np.argsort(-scores)
    return [valid_ids[i] for i in ranked_indices]

  def format_recommendations(self, movie_ids: List[str], limit: int = 10) -> List[Dict]:
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
    user_id = str(user_id)

    if not self.is_known_user(user_id):
      return self.format_recommendations(self.popular_movies, top_k)

    user_emb = self.get_user_embedding(user_id)
    candidates = self.retrieve(user_emb, top_k=100)
    ranked = self.rank_candidates(user_emb, candidates)

    return self.format_recommendations(ranked, top_k)
