from .retrieval_eval import compute_keyword_recall


def evaluate_retrieval(dataset, rag_pipeline):
    scores = []

    for sample in dataset:
        result = rag_pipeline.run(sample["query"])

        recall = compute_keyword_recall(
            result["reranked_chunks"],
            sample["keywords"],
        )

        scores.append(recall)

        print(f"\nQuery: {sample['query']}")
        print(f"Recall: {recall:.2f}")

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Retrieval Recall: {avg_score:.2f}")

    return avg_score