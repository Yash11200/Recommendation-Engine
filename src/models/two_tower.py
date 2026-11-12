"""
two_tower.py
------------
Two-Tower retrieval model using TensorFlow Recommenders.

One tower learns user embeddings, the other learns item embeddings.
We will later use FAISS on the item embeddings for fast candidate retrieval.
"""

import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text

class UserModel(tf.keras.Model):
  """Simple user tower."""

  def __init__(self, unique_user_ids, embedding_dim=64):
    super().__init__()
    self.user_embedding = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim)
    ])

  def call(self, inputs):
    return self.user_embedding(inputs)

class ItemModel(tf.keras.Model):
  """Simple item (movie) tower."""

  def __init__(self, unique_movie_ids, embedding_dim=64):
    super().__init__()
    self.item_embedding = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=unique_movie_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dim)
    ])

  def call(self, inputs):
    return self.item_embedding(inputs)

class TwoTowerModel(tfrs.Model):
    """
    Full two-tower retrieval model.
    Computes the similarity between user and item embeddings
    and uses it as the retrieval score.
    """

    def __init__(self, user_model, item_model, task):
      super().__init__()
      self.user_model = user_model
      self.item_model = item_model
      self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
      # User tower
      user_embeddings = self.user_model(features["user_id"])

      # Item tower
      positive_item_embeddings = self.item_model(features["movie_id"])

      # Retrieval task computes the loss and metrics
      return self.task(user_embeddings, positive_item_embeddings)