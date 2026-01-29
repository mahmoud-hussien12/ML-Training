from typing import List
from src.retrieval.retriever import Retriever
import numpy as np

def try_retrieval_configs(query_embedding: np.ndarray, retrievers: List[Retriever]):
    results = {}
    for retriever in retrievers:
        key = f"k={retriever.top_k},threshold={retriever.score_threshold}"
        retrieved = retriever.retrieve(query_embedding)
        results[key] = {"num_chunks": len(retrieved), "chunks": retrieved}
    return results
        