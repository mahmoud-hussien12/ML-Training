from typing import Dict, List

def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    conext_blocks = []
    for r in retrieved_chunks:
        text = r[1]["text"]
        source = r[1]["source"]
        conext_blocks.append(f"Source: {source}\nText: {text}")
    context = "\n\n".join(conext_blocks)
    prompt = f"""
You are an assistant answering questions using ONLY the context below.
If the answer is not contained in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
""".strip()
    return prompt
