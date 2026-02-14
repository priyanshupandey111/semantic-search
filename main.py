import numpy as np
import time
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Only embedding model (lightweight)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 85 fake support tickets
docs = [f"Customer login issue number {i}" for i in range(85)]
doc_embeddings = model.encode(docs, normalize_embeddings=True)

class Request(BaseModel):
    query: str
    k: int = 7
    rerank: bool = True
    rerankK: int = 4

@app.post("/search")
def search(req: Request):
    start = time.time()

    query_emb = model.encode([req.query], normalize_embeddings=True)
    scores = cosine_similarity(query_emb, doc_embeddings)[0]

    top_idx = np.argsort(scores)[::-1][:req.rerankK]

    results = [
        {
            "id": int(i),
            "score": float(scores[i]),
            "content": docs[i],
            "metadata": {"source": "support_ticket"}
        }
        for i in top_idx
    ]

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": False,
        "metrics": {
            "latency": latency,
            "totalDocs": len(docs)
        }
    }





