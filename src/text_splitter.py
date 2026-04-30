from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 120
) -> List[Dict[str, Any]]:
    """
    Split loaded documents into smaller text chunks while preserving metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        split_texts = splitter.split_text(text)

        for i, chunk_text in enumerate(split_texts):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["chunk_source_id"] = f"{metadata['file_name']}_chunk_{i}"

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks