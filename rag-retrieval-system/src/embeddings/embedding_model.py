from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingModel(ABC):
    
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents.

        Args:
            documents: List of documents

        Returns:
            np.ndarray of shape (n_documents, embedding_dim)
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.

        Args:
            query: Query string

        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        pass
        