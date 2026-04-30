import sys
from pathlib import Path
from src.vector_store import load_vector_store
import subprocess

import streamlit as st

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import VECTOR_DB_DIR, RAW_DATA_DIR


def ensure_vector_db():
    """
    Build vector DB on startup if it does not exist or is incomplete.
    """

    chroma_db_file = Path(VECTOR_DB_DIR) / "chroma.sqlite3"

    raw_files = list(Path(RAW_DATA_DIR).glob("*.txt")) + list(Path(RAW_DATA_DIR).glob("*.pdf"))

    if not raw_files:
        st.error("No documents found in data/raw/. Please check that raw SOP files are uploaded to GitHub.")
        st.stop()

    if not chroma_db_file.exists():
        with st.spinner("Vector DB not found. Building index..."):
            result = subprocess.run(
                [sys.executable, "build_index.py"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True
            )

        if result.returncode != 0:
            st.error("Failed to build vector index.")
            st.code(result.stderr)
            st.stop()
        else:
            st.success("Vector index built successfully.")


ensure_vector_db()

from src.rag_pipeline import generate_answer


st.set_page_config(
    page_title="Retail Store Virtual Manager",
    page_icon="🛒",
    layout="wide"
)


st.title("🛒 Retail Store Virtual Manager")
st.subheader("RAG-powered assistant for store operations and SOPs")

st.markdown(
    """
This chatbot helps store employees answer operational questions based on internal company documents.

**Current mode:** Retrieval-based / RAG assistant  
**Use cases:** Returns, cash register procedures, inventory receiving, damaged goods handling
"""
)

with st.sidebar:
    st.header("Settings")
    k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=5,
        value=3
    )

    st.markdown("---")
    st.markdown("### Example questions")
    st.markdown(
        """
- What should I do if a customer returns a product without a receipt?
- What should the cashier do at the end of the shift?
- During stock receiving, what should employees do with damaged or missing items?
- Can employees approve salary increases for team members?
"""
    )


query = st.text_input(
    "Ask a store operations question:",
    placeholder="Example: What should I do if a customer returns a product without a receipt?"
)

if st.button("Ask Virtual Store Manager"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching company documents..."):
            result = generate_answer(query=query, k=k)

        st.markdown("## Answer")
        st.write(result["answer"])

        st.markdown("## Sources")
        for source in result["sources"]:
            st.markdown(
                f"""
**{source.get("title")}**  
File: `{source.get("file_name")}`  
Chunk: `{source.get("chunk_id")}`
"""
            )

        with st.expander("Retrieved Context"):
            st.text(result["retrieved_context"])

        with st.expander("Generated Prompt"):
            st.text(result["prompt"])