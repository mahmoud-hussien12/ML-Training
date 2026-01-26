import sys
import os
sys.path.append(os.getcwd())

from src.vector_store.faiss_store import FaissVectorStore
from src.ingestion.ingest import ingest_documents

def main():
    DATA_PATH = "data/raw/"
    INDEX_PATH = "data/index/faiss_index"

    vector_store, embedder = ingest_documents(
        data_dir=DATA_PATH,
        vector_store=FaissVectorStore(),
    )

    vector_store.load()

    query = "What is the capital of France?"
    query_embedding = embedder.embed_query(query)
    results = vector_store.search(query_embedding, top_k=5)
    for result in results:
        print(result)



if __name__ == "__main__":
    main()