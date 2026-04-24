import sys
import os
sys.path.append(os.getcwd())

from src.vector_store.faiss_store import FaissVectorStore
from src.ingestion.ingest import ingest_documents
from src.retrieval.retriever import Retriever
from src.retrieval.tuning import try_retrieval_configs
from src.rag.rag_pipeline import RAGPipeline

def main():
    DATA_PATH = "data/raw/"
    INDEX_PATH = "data/index/faiss_index"

    vector_store, embedder = ingest_documents(
        data_dir=DATA_PATH,
        vector_store=FaissVectorStore(),
    )

    vector_store.load()

    query = "What is the capital of France?"
    print("Query:", query)
    print("########## raw search ####################")
    query_embedding = embedder.embed_query(query)
    results = vector_store.search(query_embedding, top_k=5)
    for result in results:
        print(result)
    
    print("########## try retriever configs ##########")
    retrievers = [
        Retriever(vector_store, top_k=3),
        Retriever(vector_store, top_k=5),
        Retriever(vector_store, top_k=8),
        Retriever(vector_store, top_k=5, score_threshold=0.75),
        Retriever(vector_store, top_k=5, score_threshold=0.75, source_filter="capitals.txt"),
    ]

    results = try_retrieval_configs(query_embedding, retrievers)

    for config, output in results.items():
        print(config, "→", output["num_chunks"])
    print(output["chunks"])


    print("########## RAG Pipeline ##########")
    retriever = Retriever(vector_store, top_k=5, score_threshold=0.75, source_filter="capitals.txt")
    rag_pipeline = RAGPipeline(retriever)
    result = rag_pipeline.run(query)
    print(result)
    
    
if __name__ == "__main__":
    main()