from typing import Dict, List

from src.embeddings.text_embedding import TextEmbedder
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.retrieval.retriever import Retriever
from src.rag.generator import Generator
from src.rag.prompt_builder import build_prompt


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
    ):
        self.retriever = retriever
        self.generator = Generator(model_name="llama3.2:1b")
        self.embedder = OpenAIEmbedding(model="llama3.2:1b")
    
    def run(self, query: str) -> Dict:
        query_embedding = self.embedder.embed_query(query)
        retrieved_chunks = self.retriever.retrieve(query_embedding)
        prompt = build_prompt(query, retrieved_chunks)
        answer = self.generator.generate(prompt)
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt
        }
        