import numpy as np
from typing import List, Dict

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_top_k(query_emb: List[float], corpus: List[Dict], k: int = 5) -> List[Dict]:
    scored = [
        {**item, "score": cosine_similarity(query_emb, item["embedding"])}
        for item in corpus
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]
