import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
import os
import pickle

class FaissVectorStore:
    def __init__(self,
        embedding_dim: int = 2048,
        index_path: str = "data/index/faiss_index/index.faiss",
        metadata_path: str = "data/index/faiss_index/metadata.pkl"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim

        self.index = faiss.IndexFlatIP(embedding_dim)

        self.metadata: List[Dict[str, Any]] = []
    
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):

        assert embeddings.shape[0] == len(metadata)
        assert embeddings.shape[1] == self.embedding_dim

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            results.append((float(score), self.metadata[index]))
        return results
    
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Index or metadata file not found")
        
        self.index = faiss.read_index(self.index_path)

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
    

