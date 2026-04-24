from typing import List, Dict

def compute_keyword_recall(
    retrieved_chunks: List[Dict],
    keywords: List[str],
) -> float:
    """
    Checks if retrieved chunks contain expected keywords.
    """
    retrieved_text = " ".join(
        [r["metadata"]["text"].lower() for r in retrieved_chunks]
    )

    hits = sum(1 for kw in keywords if kw.lower() in retrieved_text)

    return hits / len(keywords)