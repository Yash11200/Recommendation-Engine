
Individual Reflection: Large-Scale Personalized Recommendation Engine (AIML-03)


1. Project Overview & Goals

During this project, I designed and built RecoMind, an end-to-end, two-stage personalized recommendation engine modeled after production systems at Netflix and Amazon. Working with the MovieLens 25M dataset, my primary objective was to bridge the gap between offline machine learning modeling and low-latency online serving—combining PySpark, TensorFlow Recommenders, FAISS, FastAPI, Redis, and a React frontend.

2. Key Technical Learnings

• Why Two-Stage Architectures Matter: Running a heavy neural ranking model across tens of thousands of catalog items per request is computationally impractical. Separating candidate retrieval (Two-Tower + FAISS) from fine-grained scoring (Neural Ranking Model) allowed me to narrow down the item space in milliseconds while still applying rich interaction features to the top candidates.
• Preventing Data Leakage with Time-Based Splits: Early on, I realized that random train/test splits artificially inflate offline metrics because the model learns from future user behavior to predict past ratings. Enforcing a strict chronological split—and computing Spark aggregation features strictly on historical windows—made my evaluation metrics (Recall@K and NDCG@K) trustworthy.
• Production Resilience & Caching: Building the serving layer taught me that an ML model is only useful if the surrounding infrastructure handles edge cases. Adding a Redis caching layer cut repeat-query latency down to ~4 ms, while implementing a Popularity Fallback ensured unseen cold-start users still receive immediate, sensible recommendations without breaking the API.

3. Challenges Faced & How I Solved Them

• Scaling Feature Engineering on 25M Rows: Processing 25 million ratings in standard Pandas caused memory bottlenecks during local development. I solved this by adding a --sample flag for rapid prototyping and migrating heavy aggregations (user activity, item popularity, and genre stats) to Apache Spark (PySpark) backed by Parquet storage.
• Aligning Embedding Dimensions & FAISS Indexing: Ensuring consistent ID mappings between TensorFlow Recommenders' string lookup vocabularies, the saved NumPy embeddings, and FAISS integer indices required careful data validation. Writing clean, modular scripts under src/retrieval/ and verifying top-K outputs manually prevented silent ID mismatches.

4. What I Would Do Differently Next Time

• Earlier End-to-End Integration: I spent significant time tuning offline model hyperparameters before connecting the FastAPI backend and React UI. In future projects, I would build a thin end-to-end pipeline with simple baselines first, then iterate on model accuracy once serving contracts are locked in.
• Richer Session Context: Currently, user embeddings rely on historical batch training. Incorporating short-term session history (such as the last 5 movies clicked in the UI) would make recommendations adapt dynamically in real time.

5. Conclusion

This project strengthened my understanding of full-stack machine learning engineering—from distributed data processing and vector similarity search to REST API design and UI integration. It reinforced my belief that clean data pipelines, leak-free evaluation, and low-latency serving infrastructure are just as critical to a recommender system's success as the neural network architecture itself.