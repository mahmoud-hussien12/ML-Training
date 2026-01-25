from src.embeddings.embedding_model import EmbeddingModel
from openai import OpenAI
import os

from typing import List
import numpy as np

class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 32):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            data = response.data if response.data else []
            embeddings.extend([item.embedding for item in data])
        return np.array(embeddings, dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=query)

        return np.array(response.data[0].embedding, dtype=np.float32)