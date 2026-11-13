"""
redis_cache.py
--------------
Simple Redis helper for caching recommendation results.
"""

import json
import redis
from typing import List, Optional

class RecommendationCache:
  def __init__(self, host="localhost", port=6379, db=0, ttl=3600):
    """
    ttl = time-to-live in seconds (default 1 hour)
    """
    self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    self.ttl = ttl

  def _key(self, user_id: str, top_k: int) -> str:
    return f"rec:{user_id}:{top_k}"

  def get(self, user_id: str, top_k: int = 10) -> Optional[List[dict]]:
    """Return cached recommendations or None."""
    data = self.client.get(self._key(user_id, top_k))
    if data:
      return json.loads(data)
    return None

  def set(self, user_id: str, recommendations: List[dict], top_k: int = 10):
    """Cache recommendations."""
    self.client.setex(
      self._key(user_id, top_k),
      self.ttl,
      json.dumps(recommendations)
    )

  def ping(self) -> bool:
    """Check if Redis is available."""
    try:
      return self.client.ping()
    except Exception:
      return False
