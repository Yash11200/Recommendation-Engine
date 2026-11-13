"""
ranking.py
----------
Simple but effective neural ranking model.
Takes user features + item features and outputs a relevance score.
We keep it clean and easy to understand.
"""

import tensorflow as tf


class RankingModel(tf.keras.Model):
    """
    A basic deep ranking model.
    Input: concatenated user embedding + item embedding (+ optional features)
    Output: a single score (higher = more relevant)
    """

    def __init__(self, embedding_dim=64, hidden_units=[128, 64]):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Deep layers
        layers = []
        for units in hidden_units:
            layers.append(tf.keras.layers.Dense(units, activation="relu"))
            layers.append(tf.keras.layers.Dropout(0.2))

        layers.append(tf.keras.layers.Dense(1))  # final score

        self.deep = tf.keras.Sequential(layers)

    def call(self, inputs, training=False):
        """
        inputs: dictionary with
            - user_embedding
            - item_embedding
        """
        user_emb = inputs["user_embedding"]
        item_emb = inputs["item_embedding"]

        # Simple concatenation
        x = tf.concat([user_emb, item_emb], axis=1)

        score = self.deep(x, training=training)
        return score