# Large-Scale Personalized Recommendation Engine

**Internship Project – AIML-03 (Project #1 of 2)**

A two-stage recommender system inspired by Netflix / Amazon:

1. **Retrieval stage** – Two-tower model (TensorFlow Recommenders) + FAISS for fast candidate generation
2. **Ranking stage** – Deep ranking model (DCN / Wide & Deep style) that re-scores candidates

### Required Tech Stack
- TensorFlow Recommenders
- FAISS
- Redis
- Apache Spark (PySpark)
- Airflow
- Triton Inference Server (or TF Serving alternative for demo)

### Dataset
MovieLens 25M (GroupLens) – time-based split (no random leakage)

---

## Project Structure

```
recommendation-engine/
├── data/
│   ├── raw/              # Original MovieLens zip + extracted files
│   ├── processed/        # Parquet files (train/val/test + features)
│   └── embeddings/       # Saved user/item embeddings
├── src/
│   ├── data/             # download, preprocess, spark features
│   ├── models/           # two-tower + ranking models
│   ├── retrieval/        # FAISS index
│   ├── serving/          # FastAPI + Redis
│   └── utils/
├── frontend/             # Attractive React + Tailwind UI
├── airflow/dags/         # Retraining schedule
├── triton/               # Model serving configs
├── requirements.txt
└── README.md
```

---

## Quick Start (Day 1)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download MovieLens 25M (~250 MB)
python -m src.data.download

# 4. Time-based split + basic features
python -m src.data.preprocess

# Optional – use a smaller sample for fast development
python -m src.data.preprocess --sample 2000000

# 5. Spark feature engineering (popularity, user activity, genres)
python -m src.data.spark_features
```

---

## Day-by-Day Progress

| Day | Status | Focus |
|-----|--------|-------|
| 1   | ✅     | Data pipeline + project skeleton + UI shell |
| 2   |        | Two-tower retrieval model |
| 3   |        | FAISS index + Ranking model |
| 4   |        | FastAPI + Redis + cold-start |
| 5   |        | Full pipeline + Airflow + Triton light |
| 6   |        | Attractive UI polish + demo |
| 7   |        | Metrics, docs, presentation, reflection |

---

## Deliverables Checklist
- [ ] Working demo (UI + backend)
- [ ] Clean GitHub repository with README & setup
- [ ] Architecture diagram + API docs
- [ ] Test results (Recall@K, NDCG@K, etc.)
- [ ] 10-minute presentation
- [ ] Individual reflection

---

## Notes
- Keep code simple and well-commented
- Prefer clarity over cleverness
- Always use time-based splits
- Cache predictions in Redis
- Handle cold-start users with popularity + content fallbacks
