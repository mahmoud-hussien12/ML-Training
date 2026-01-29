from src.vector_store.faiss_store import FaissVectorStore
import numpy as np
from typing import List, Dict

class Retriever:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        top_k: int = 5,
        score_threshold: float | None = None,
        source_filter: str | None = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.source_filter = source_filter
    
    def retrieve(self, query_embedding: np.ndarray) -> List[Dict]:
        results = self.vector_store.search(query_embedding, self.top_k)
        filtered = []
        for result in results:
            if self.score_threshold is not None and result[0] < self.score_threshold:
                continue
            if self.source_filter is not None and result[1]["source"] != self.source_filter:
                continue
            filtered.append(result)
        return filtered