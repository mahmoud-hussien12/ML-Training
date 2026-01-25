import sys
import os
sys.path.append(os.getcwd())

from src.ingestion.load_documents import load_documents
from src.chunking.text_splitter import split_documents
from src.embeddings.embedder import embed_texts
from src.embeddings.text_embedding import TextEmbedder
from src.vector_store.faiss_store import FaissVectorStore

def build_pipeline(data_path: str, index_path: str):
    """
    Build FAISS index from raw documents.
    """
    print("Loading documents...")
    documents = load_documents(data_path)

    print(f"Loaded {len(documents)} documents")

    print("Splitting documents into chunks...")
    chunks = split_documents(
        documents,
        chunk_size=500,
        overlap=100,
    )

    print(f"Generated {len(chunks)} chunks")

    text_embedding = TextEmbedder()
    print("Embedding chunks...")
    embeddings = text_embedding.embed_documents(chunks)

    print("Building FAISS index...")
    store = FaissVectorStore(embedding_dim=embeddings.shape[1])
    store.add(embeddings, [{"text": chunk, "id": i} for i, chunk in enumerate(chunks)])

    print("Saving index to disk...")
    store.save()

    return embeddings

def main():
    DATA_PATH = "data/raw/sample_docs.txt"
    INDEX_PATH = "data/index/faiss_index"

    # Step 1: Build index (run once or when data changes)
    embeddings = build_pipeline(
        data_path=DATA_PATH,
        index_path=INDEX_PATH,
    )

    store = FaissVectorStore(embedding_dim=embeddings.shape[1])
    store.load()

    results = store.search(embeddings[1], top_k=5)

    print(results[0])



if __name__ == "__main__":
    main()