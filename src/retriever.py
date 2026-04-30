from typing import List, Dict, Any

from langchain_core.documents import Document

from src.vector_store import load_vector_store


def retrieve_relevant_documents(
    query: str,
    k: int = 3
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a user query.
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query=query,
        k=k
    )

    return results


def format_retrieved_documents(
    documents: List[Document]
) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    """

    formatted_context = []

    for i, doc in enumerate(documents, start=1):
        title = doc.metadata.get("title", "Unknown Document")
        file_name = doc.metadata.get("file_name", "Unknown File")
        chunk_id = doc.metadata.get("chunk_id", "Unknown Chunk")

        formatted_context.append(
            f"""
[Source {i}]
Document Title: {title}
File Name: {file_name}
Chunk ID: {chunk_id}

Content:
{doc.page_content}
"""
        )

    return "\n".join(formatted_context)


def get_retrieval_results(
    query: str,
    k: int = 3
) -> Dict[str, Any]:
    """
    Full retrieval step:
    query -> relevant documents -> formatted context.
    Removes duplicate chunks.
    """

    documents = retrieve_relevant_documents(query=query, k=k)

    unique_documents = []
    seen_ids = set()

    for doc in documents:
        chunk_source_id = doc.metadata.get("chunk_source_id")

        if chunk_source_id not in seen_ids:
            unique_documents.append(doc)
            seen_ids.add(chunk_source_id)

    context = format_retrieved_documents(unique_documents)

    sources = [
        {
            "title": doc.metadata.get("title"),
            "file_name": doc.metadata.get("file_name"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
        }
        for doc in unique_documents
    ]

    return {
        "query": query,
        "context": context,
        "sources": sources,
        "documents": unique_documents
    }