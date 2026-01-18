from ingestion.load_documents import load_documents
from chunking.text_splitter import split_documents
from embeddings.embedder import embed_texts
from vector_store.faiss_store import (
    build_faiss_index,
    save_index,
    load_index,
    search_index,
)


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

    print("Embedding chunks...")
    embeddings = embed_texts(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index to disk...")
    save_index(index, chunks, index_path)

    print("Pipeline completed successfully!")


def query_pipeline(query: str, index_path: str, top_k: int = 5):
    """
    Load FAISS index and retrieve relevant chunks for a query.
    """
    print("Loading FAISS index...")
    index, chunks = load_index(index_path)

    print(f"Embedding query: '{query}'")
    query_embedding = embed_texts([query])

    print("Retrieving relevant chunks...")
    results = search_index(
        index=index,
        query_embedding=query_embedding,
        chunks=chunks,
        top_k=top_k,
    )

    print("\nTop retrieved chunks:")
    for i, chunk in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(chunk)


if __name__ == "__main__":
    DATA_PATH = "data/raw/sample_docs.txt"
    INDEX_PATH = "data/index/faiss_index"

    # Step 1: Build index (run once or when data changes)
    build_pipeline(
        data_path=DATA_PATH,
        index_path=INDEX_PATH,
    )

    # Step 2: Query
    user_query = "What is customer churn?"
    query_pipeline(
        query=user_query,
        index_path=INDEX_PATH,
        top_k=5,
    )
