from src.rag_pipeline import generate_answer


def run_chatbot():
    """
    Command-line chatbot interface for the Retail Store RAG Assistant.
    """

    print("=" * 70)
    print("Retail Store Virtual Manager Chatbot")
    print("Ask a store operations question.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    while True:
        user_query = input("\nYour question: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("\nChatbot session ended.")
            break

        if not user_query:
            print("Please enter a valid question.")
            continue

        result = generate_answer(query=user_query, k=3)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(
                f"- {source.get('title')} "
                f"({source.get('file_name')}, chunk {source.get('chunk_id')})"
            )


if __name__ == "__main__":
    run_chatbot()