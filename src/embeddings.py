from typing import List

from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_PROVIDER,
    OPENAI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
)


def get_embedding_model():
    """
    Return embedding model based on selected provider.

    Supported providers:
    - openai: OpenAI embeddings API
    - local: sentence-transformers via HuggingFaceEmbeddings
    """

    if EMBEDDING_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please add it to your .env file."
            )

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )

    if EMBEDDING_PROVIDER == "local":
        return HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}. "
        "Use 'openai' or 'local'."
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts.
    """

    embedding_model = get_embedding_model()
    return embedding_model.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """
    Embed a single user query.
    """

    embedding_model = get_embedding_model()
    return embedding_model.embed_query(text)