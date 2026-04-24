def simple_answer_match(
    predicted: str,
    expected: str,
) -> float:
    """
    Basic string overlap score.
    """
    predicted = predicted.lower()
    expected = expected.lower()

    common = sum(1 for word in expected.split() if word in predicted)

    return common / len(expected.split())