from src.ingestion.document_loader import load_text_files
from src.chunking.chunker import chunk_documents
from src.embeddings.text_embedding import TextEmbedder
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vector_store.faiss_store import FaissVectorStore


def ingest_documents(
    data_dir: str,
    vector_store: FaissVectorStore,
):
    documents = load_text_files(data_dir)

    # Chunk documents sentence by sentence to get correct captial (the max sentence length is 70)
    chunked_docs = chunk_documents(
        documents,
        chunk_method="delimiter",
        delimiter="\n"
    )

    metadatas = []
    texts = []

    for chunk in chunked_docs:
        texts.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        })

    embedder = OpenAIEmbedding(model="llama3.2:1b")
    embeddings = embedder.embed_documents(texts)

    vector_store.add(embeddings, metadatas)
    vector_store.save()

    return vector_store, embedder

