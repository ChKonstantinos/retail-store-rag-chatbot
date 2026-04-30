from src.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.document_loader import load_documents
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


def main():
    print("=" * 70)
    print("Building Retail Store RAG Vector Index")
    print("=" * 70)

    print("\nLoading documents...")
    documents = load_documents(RAW_DATA_DIR)
    print(f"Loaded documents: {len(documents)}")

    print("\nSplitting documents into chunks...")
    chunks = split_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print(f"Created chunks: {len(chunks)}")

    print("\nCreating Chroma vector store...")
    create_vector_store(chunks)
    print("Vector store created successfully.")

    print("\nIndex build completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()