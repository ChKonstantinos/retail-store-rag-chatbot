from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import VECTOR_DB_DIR, COLLECTION_NAME
from src.embeddings import get_embedding_model


def convert_chunks_to_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert our custom chunk dictionaries into LangChain Document objects.
    """

    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
        )

    return documents


def create_vector_store(chunks: List[Dict[str, Any]]) -> Chroma:
    """
    Create and persist Chroma vector store from text chunks.
    """

    embedding_model = get_embedding_model()
    documents = convert_chunks_to_documents(chunks)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR)
    )

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load existing Chroma vector store from disk.
    """

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(VECTOR_DB_DIR)
    )

    return vector_store