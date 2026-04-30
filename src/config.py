from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db" / "chroma"

OUTPUTS_DIR = BASE_DIR / "outputs"
RETRIEVED_CONTEXTS_DIR = OUTPUTS_DIR / "retrieved_contexts"
SAMPLE_ANSWERS_DIR = OUTPUTS_DIR / "sample_answers"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_PROVIDER = "local"  # options: "openai", "local"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_PROVIDER = "fallback"  # options: "openai", "fallback"

OPENAI_CHAT_MODEL = "gpt-4o-mini"
#CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

COLLECTION_NAME = "retail_store_sops"