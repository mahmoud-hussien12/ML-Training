from typing import Dict, List

from src.embeddings.text_embedding import TextEmbedder
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.retrieval.retriever import Retriever
from src.rag.generator import Generator
from src.rag.prompt_builder import build_prompt
from src.reranking.cross_encoder_reranker import CrossEncoderReranker

class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
    ):
        self.retriever = retriever
        self.generator = Generator(model_name="llama3.2:1b")
        self.embedder = OpenAIEmbedding(model="llama3.2:1b")
        self.reranker = CrossEncoderReranker(top_k=3)
    
    def run(self, query: str) -> Dict:
        query_embedding = self.embedder.embed_query(query)
        retrieved_chunks = self.retriever.retrieve(
            query_embedding=query_embedding
        )

        reranked_chunks = self.reranker.rerank(
            query=query,
            retrieved_chunks=retrieved_chunks
        )

        prompt = build_prompt(
            query=query,
            retrieved_chunks=reranked_chunks,
        )
        answer = self.generator.generate(prompt)
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "reranked_chunks": reranked_chunks,
            "prompt": prompt
        }
        