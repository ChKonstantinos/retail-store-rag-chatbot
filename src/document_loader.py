from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader


def extract_title_from_text(text: str, fallback_title: str) -> str:
    """
    Extract document title from the first lines of the text.
    If no title is found, use the filename as fallback.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:5]:
        if line.lower().startswith("document title:"):
            return line.replace("Document Title:", "").strip()

    return fallback_title


def load_txt_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a .txt document and return text with metadata.
    """
    text = file_path.read_text(encoding="utf-8")

    return {
        "text": text,
        "metadata": {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "txt",
            "title": extract_title_from_text(text, file_path.stem),
        },
    }


def load_pdf_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a .pdf document and return text with metadata.
    """
    reader = PdfReader(str(file_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(page_text)

    text = "\n".join(pages)

    return {
        "text": text,
        "metadata": {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "pdf",
            "title": extract_title_from_text(text, file_path.stem),
            "total_pages": len(reader.pages),
        },
    }


def load_documents(raw_data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all supported documents from raw data folder.
    Supported formats: .txt, .pdf
    """
    documents = []

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    supported_files = list(raw_data_dir.glob("*.txt")) + list(raw_data_dir.glob("*.pdf"))

    if not supported_files:
        raise FileNotFoundError(f"No .txt or .pdf files found in: {raw_data_dir}")

    for file_path in supported_files:
        try:
            if file_path.suffix.lower() == ".txt":
                document = load_txt_file(file_path)
            elif file_path.suffix.lower() == ".pdf":
                document = load_pdf_file(file_path)
            else:
                continue

            documents.append(document)

        except Exception as error:
            print(f"Error loading {file_path.name}: {error}")

    return documents