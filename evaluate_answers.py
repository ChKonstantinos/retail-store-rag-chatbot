import pandas as pd

from src.rag_pipeline import generate_answer
from src.config import OUTPUTS_DIR


EVALUATION_QUESTIONS = [
    {
        "query": "What should I do if a customer returns a product without a receipt?",
        "expected_source": "Returns and Refunds Policy",
        "business_area": "Customer Service"
    },
    {
        "query": "What should the cashier do at the end of the shift?",
        "expected_source": "Cash Register Standard Operating Procedure",
        "business_area": "Cash Control"
    },
    {
        "query": "During stock receiving, what should employees do with damaged or missing items?",
        "expected_source": "Inventory Receiving and Replenishment SOP",
        "business_area": "Inventory"
    },
    {
        "query": "Can employees approve salary increases for team members?",
        "expected_source": None,
        "business_area": "Out of Scope"
    }
]


def evaluate_answer(query: str, expected_source: str, business_area: str):
    result = generate_answer(query=query, k=3)

    answer = result["answer"]
    sources = result["sources"]

    retrieved_titles = [
        source["title"] for source in sources
    ]

    source_found = (
        expected_source in retrieved_titles
        if expected_source is not None
        else True
    )

    return {
        "query": query,
        "business_area": business_area,
        "expected_source": expected_source,
        "retrieved_sources": " | ".join([str(title) for title in retrieved_titles]),
        "source_found": source_found,
        "answer": answer
    }


def main():
    results = []

    for item in EVALUATION_QUESTIONS:
        results.append(
            evaluate_answer(
                query=item["query"],
                expected_source=item["expected_source"],
                business_area=item["business_area"]
            )
        )

    df = pd.DataFrame(results)

    output_path = OUTPUTS_DIR / "evaluation" / "answer_evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("=" * 80)
    print("Answer Evaluation Completed")
    print("=" * 80)
    print(df[["business_area", "expected_source", "source_found"]])
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()