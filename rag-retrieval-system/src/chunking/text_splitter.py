def split_documents(docs: list[str], chunk_size=500, overlap=100):
    return [doc[:chunk_size] for doc in docs]
