from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100 ) -> List[str]:
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