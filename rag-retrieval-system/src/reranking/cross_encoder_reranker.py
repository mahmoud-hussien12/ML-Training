from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-TinyBERT-L2-v2',
        top_k: int = 3,
    ):
        self.model_name = model_name
        self.model = CrossEncoder(self.model_name)
        self.top_k = top_k

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict],
    ) -> List[Dict]:
        """
        Rerank retrieved chunks using cross-encoder scoring.
        """
        if not retrieved_chunks:
            return []

        pairs = [
            (query, r["metadata"]["text"])
            for r in retrieved_chunks
        ]

        scores = self.model.predict(pairs)

        # Attach scores
        for r, score in zip(retrieved_chunks, scores):
            r["rerank_score"] = float(score)

        # Sort by rerank score
        reranked = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[: self.top_k]