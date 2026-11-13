"""
api.py
------
FastAPI application that serves recommendations.
Includes Redis caching and cold-start support.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from src.serving.recommender import RecommenderService
from src.serving.redis_cache import RecommendationCache

app = FastAPI(
  title="RecoMind API",
  description="Large-Scale Personalized Recommendation Engine",
  version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

recommender = RecommenderService()
cache = RecommendationCache()


class RecommendationResponse(BaseModel):
  user_id: str
  recommendations: List[dict]
  cached: bool
  cold_start: bool


@app.get("/")
def root():
  return {
    "message": "RecoMind Recommendation API is running",
    "docs": "/docs"
  }


@app.get("/health")
def health():
  redis_ok = cache.ping()
  return {
    "status": "ok",
    "redis": "connected" if redis_ok else "not available (caching disabled)"
  }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
  user_id: str,
  top_k: int = Query(10, ge=1, le=50)
):
  """
  Get top-k recommendations for a user.
  Uses Redis cache when available.
  """
  cached_result = cache.get(user_id, top_k)
  if cached_result is not None:
    return {
      "user_id": user_id,
      "recommendations": cached_result,
      "cached": True,
      "cold_start": not recommender.is_known_user(user_id)
    }

  try:
    recs = recommender.recommend(user_id, top_k=top_k)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

  try:
    cache.set(user_id, recs, top_k)
  except Exception:
    pass

  return {
    "user_id": user_id,
    "recommendations": recs,
    "cached": False,
    "cold_start": not recommender.is_known_user(user_id)
  }


if __name__ == "__main__":
  uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
