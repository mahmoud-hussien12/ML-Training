from src.embeddings.embedding_model import EmbeddingModel
from fastembed import TextEmbedding
import os

from typing import List
import numpy as np

class TextEmbedder(EmbeddingModel):
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        self.model = model
        self.client = TextEmbedding(model)
        self.batch_size = batch_size
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        embeddings = list(self.client.embed(documents))
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        embeddings = list(self.client.embed([query]))
        return np.array(embeddings, dtype=np.float32)