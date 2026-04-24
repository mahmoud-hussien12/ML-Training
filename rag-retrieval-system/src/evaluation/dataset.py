from typing import List, Dict

def get_eval_dataset() -> List[Dict]:
    """
    Each sample contains:
    - query
    - expected_answer (optional)
    - relevant_keywords (for retrieval eval)
    """
    return [
        {
            "query": "What is customer churn?",
            "expected_answer": "Customer churn refers to customers leaving a service.",
            "keywords": ["churn", "customer", "leave"],
        },
        {
            "query": "What is FAISS?",
            "expected_answer": "FAISS is a library for similarity search.",
            "keywords": ["FAISS", "similarity search"],
        },
    ]