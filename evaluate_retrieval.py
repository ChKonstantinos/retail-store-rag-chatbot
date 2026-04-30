import pandas as pd

from src.retriever import get_retrieval_results
from src.config import OUTPUTS_DIR


EVALUATION_QUERIES = [
    {
        "query": "What should I do if a customer returns a product without a receipt?",
        "expected_document": "Returns and Refunds Policy",
        "business_area": "Customer Service"
    },
    {
        "query": "Can a customer get a cash refund without proof of purchase?",
        "expected_document": "Returns and Refunds Policy",
        "business_area": "Customer Service"
    },
    {
        "query": "What should the cashier do at the end of the shift?",
        "expected_document": "Cash Register Standard Operating Procedure",
        "business_area": "Cash Control"
    },
    {
        "query": "When is supervisor approval needed for price changes?",
        "expected_document": "Cash Register Standard Operating Procedure",
        "business_area": "Cash Control"
    },
    {
        "query": "During stock receiving, what should employees do with damaged or missing items?",
        "expected_document": "Inventory Receiving and Replenishment SOP",
        "business_area": "Inventory"
    },
    {
        "query": "What should employees check when goods arrive?",
        "expected_document": "Inventory Receiving and Replenishment SOP",
        "business_area": "Inventory"
    },
]


def evaluate_query(query: str, expected_document: str, business_area: str, k: int = 3):
    retrieval = get_retrieval_results(query=query, k=k)

    retrieved_titles = [
        source["title"] for source in retrieval["sources"]
    ]

    top_1 = retrieved_titles[0] if retrieved_titles else None

    hit_at_k = expected_document in retrieved_titles
    hit_at_1 = expected_document == top_1

    return {
        "query": query,
        "business_area": business_area,
        "expected_document": expected_document,
        "top_1_document": top_1,
        "retrieved_documents": " | ".join(retrieved_titles),
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
    }


def main():
    print("=" * 80)
    print("Evaluating Retail Store RAG Retrieval")
    print("=" * 80)

    results = []

    for item in EVALUATION_QUERIES:
        result = evaluate_query(
            query=item["query"],
            expected_document=item["expected_document"],
            business_area=item["business_area"],
            k=3
        )
        results.append(result)

    df = pd.DataFrame(results)

    hit_at_1 = df["hit_at_1"].mean()
    hit_at_k = df["hit_at_k"].mean()

    print("\nEvaluation Results:")
    print(df[[
        "business_area",
        "expected_document",
        "top_1_document",
        "hit_at_1",
        "hit_at_k"
    ]])

    print("\nMetrics:")
    print(f"Hit@1: {hit_at_1:.2f}")
    print(f"Hit@3: {hit_at_k:.2f}")

    output_path = OUTPUTS_DIR / "evaluation" / "retrieval_evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nSaved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()