from typing import List, Dict

def chunk_text_by_size(text: str, chunk_size: int = 500, overlap: int = 50 ) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunk_text_by_delimiter(text: str, delimiter: str = "\n") -> List[str]:
    return text.split(delimiter)


def chunk_documents(
    documents: List[Dict],
    chunk_method: str = "size",
    delimiter: str = "\n",
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Chunk documents and attach metadata.
    """
    chunked_docs = []
    for doc in documents:
        if chunk_method == "size":
            chunks = chunk_text_by_size(doc["text"], chunk_size, overlap)
        elif chunk_method == "delimiter":
            chunks = chunk_text_by_delimiter(doc["text"], delimiter)
        else:
            raise ValueError("Invalid chunk method")

        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "text": chunk,
                "source": doc["source"],
                "chunk_id": i,
            })

    return chunked_docs
