import faiss
import numpy as np
from typing import List, Tuple
import os
import pickle

def build_faiss_index(embeddings: List[List[float]]):
    vectors = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, chunks: List[str], index_path: str):    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(index_path: str) -> Tuple[faiss.Index, List[str]]:
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def search_index(index: faiss.Index, query_embedding: List[List[float]], chunks: List[str], top_k: int = 5) -> List[str]:
    query_vectors = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_vectors, top_k) 
    return [chunks[i] for i in indices[0]]

