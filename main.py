import numpy as np
import time
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Create 85 fake support tickets
docs = [f"Customer login issue number {i}" for i in range(85)]

# Create embeddings once
doc_embeddings = embed_model.encode(docs, normalize_embeddings=True)

class Request(BaseModel):
    query: str
    k: int = 7
    rerank: bool = True
    rerankK: int = 4

@app.post("/search")
def search(req: Request):
    start = time.time()

    # Stage 1: Vector Search
    q_emb = embed_model.encode([req.query], normalize_embeddings=True)
    scores = cosine_similarity(q_emb, doc_embeddings)[0]

    top_idx = np.argsort(scores)[::-1][:req.k]

    candidates = [
        {
            "id": int(i),
            "score": float(scores[i]),
            "content": docs[i]
        }
        for i in top_idx
    ]

    # Stage 2: Re-ranking
    if req.rerank:
        pairs = [[req.query, c["content"]] for c in candidates]
        rerank_scores = rerank_model.predict(pairs)

        min_s = min(rerank_scores)
        max_s = max(rerank_scores)
        norm_scores = [(s - min_s) / (max_s - min_s + 1e-8) for s in rerank_scores]

        for i in range(len(candidates)):
            candidates[i]["score"] = float(norm_scores[i])

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": req.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(docs)
        }
    }




