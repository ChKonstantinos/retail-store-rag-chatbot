from src.retriever import get_retrieval_results


def test_query(query: str, expected_title_keyword: str, k: int = 3):
    result = get_retrieval_results(query=query, k=k)

    retrieved_titles = [
        source["title"] for source in result["sources"]
    ]

    print("=" * 80)
    print(f"Query: {query}")
    print("Retrieved titles:")
    for title in retrieved_titles:
        print(f"- {title}")

    success = any(
        expected_title_keyword.lower() in title.lower()
        for title in retrieved_titles
        if title
    )

    if success:
        print("Status: PASS")
    else:
        print("Status: FAIL")

    return success


if __name__ == "__main__":
    tests = [
        {
            "query": "What should I do if a customer returns a product without a receipt?",
            "expected": "Returns"
        },
        {
            "query": "What should the cashier do when closing the register?",
            "expected": "Cash Register"
        },
        {
            "query": "How should damaged items be handled during receiving?",
            "expected": "Inventory"
        },
    ]

    passed = 0

    for test in tests:
        if test_query(
            query=test["query"],
            expected_title_keyword=test["expected"],
            k=3
        ):
            passed += 1

    print("=" * 80)
    print(f"Passed {passed}/{len(tests)} retrieval tests")