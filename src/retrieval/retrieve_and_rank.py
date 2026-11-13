"""
retrieve_and_rank.py
--------------------
End-to-end retrieval + ranking for a given user.
1. Get user embedding from the Two-Tower model
2. Search FAISS for top-K candidates
3. Re-rank the candidates with the Ranking model
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
import faiss
import pandas as pd

from src.models.two_tower import UserModel, ItemModel
from src.models.ranking import RankingModel

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
INDEX_DIR = ROOT / "data" / "faiss_index"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "saved_models" / "two_tower"


def load_faiss():
    index = faiss.read_index(str(INDEX_DIR / "item_index.faiss"))
    item_ids = np.load(INDEX_DIR / "item_ids.npy", allow_pickle=True)
    return index, item_ids


def get_user_embedding(user_id: str, user_model, unique_user_ids):
    """Get embedding for a single user."""
    # If user is completely new (cold-start), we will handle later
    user_tensor = tf.constant([user_id])
    return user_model(user_tensor).numpy()


def retrieve_candidates(user_embedding, index, item_ids, top_k=100):
    """Retrieve top-k candidates using FAISS."""
    # Normalize query
    user_embedding = user_embedding.astype("float32")
    faiss.normalize_L2(user_embedding)

    distances, indices = index.search(user_embedding, top_k)

    candidate_ids = item_ids[indices[0]]
    scores = distances[0]

    return candidate_ids, scores


def main():
    print("=" * 60)
    print("Retrieval + Ranking Demo")
    print("=" * 60)

    # Load FAISS
    index, item_ids = load_faiss()
    print(f"FAISS index loaded. Vectors: {index.ntotal}")

    # For demo we just pick a random existing user from training data
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    sample_user = str(train_df["userId"].sample(1).iloc[0])
    print(f"\nDemo user_id: {sample_user}")

    # Load item embeddings (we already have them)
    item_embeddings = np.load(EMBEDDINGS_DIR / "item_embeddings.npy")

    # Note: In a full production system we would also load the trained user_model
    # and ranking model. For Day 3 we demonstrate the retrieval part cleanly.

    # ---- Retrieval ----
    # For this demo we use a random item embedding as a proxy query
    # (Tomorrow we will properly load the user tower)
    query_embedding = item_embeddings[np.random.randint(0, len(item_embeddings))].reshape(1, -1)

    candidate_ids, retrieval_scores = retrieve_candidates(query_embedding, index, item_ids, top_k=20)

    print("\nTop 10 candidates from FAISS retrieval:")
    for i, (mid, score) in enumerate(zip(candidate_ids[:10], retrieval_scores[:10])):
        print(f"{i+1:2d}. Movie ID: {mid:<10}  Score: {score:.4f}")

    print("\nDay 3 retrieval part working successfully!")


if __name__ == "__main__":
    main()