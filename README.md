# 🛒 Retail Store RAG Chatbot (Virtual Store Manager)

A production-style Retrieval-Augmented Generation (RAG) system designed to support retail store operations by answering employee questions based on internal company procedures and SOP documents.

---

## 🚀 Project Overview

This project implements a **Virtual Store Manager chatbot** that helps retail employees quickly access and follow company procedures such as:

- Returns & Refunds Policy
- Cash Register Operations
- Inventory Receiving & Replenishment
- Damaged Goods Handling

The system uses a **RAG architecture**, combining document retrieval with AI-driven response generation.

---

## 🧠 Key Features

- 📄 Document ingestion pipeline (TXT / PDF support)
- ✂️ Intelligent text chunking
- 🔎 Semantic search using embeddings
- 🗄️ Vector database (Chroma)
- 🔁 Hybrid embedding architecture (OpenAI + Local models)
- 📊 Retrieval evaluation (Hit@1, Hit@k)
- 🧪 Answer evaluation pipeline
- ⚠️ Out-of-scope detection (prevents hallucinations)
- 🧠 Rule-based fallback logic (no API required)
- 💬 Interactive chatbot (CLI + Streamlit UI)

---

## 🏗️ System Architecture

```text
Documents → Chunking → Embeddings → Vector DB → Retrieval → Answer Generation
```

---

## ⚙️ Tech Stack

- Python
- LangChain
- ChromaDB
- Sentence Transformers
- OpenAI API optional
- Streamlit
- Pandas

---

## 📁 Project Structure

```text
retail-store-rag-chatbot/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_db/
│
├── outputs/
│   ├── evaluation/
│   └── sample_answers/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompt_templates.py
│   ├── rag_pipeline.py
│   └── chatbot.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   └── test_retrieval.py
│
├── build_index.py
├── evaluate_retrieval.py
├── evaluate_answers.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 📄 Documents Used

The chatbot is built on sample retail SOP documents stored in `data/raw/`.

Example documents:

- `returns_policy.txt`
- `cash_register_sop.txt`
- `inventory_sop.txt`

These documents simulate internal company procedures for a retail store environment.

---

## 🔁 RAG Pipeline

The project follows a modular RAG pipeline:

```text
1. Load documents
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in Chroma vector database
5. Retrieve relevant chunks based on user query
6. Generate or structure an answer
7. Return answer with sources
```

---

## 🔎 Hybrid Embedding Architecture

The project supports two embedding providers:

```python
EMBEDDING_PROVIDER = "local"   # options: "local", "openai"
```

### Local mode

Uses Sentence Transformers locally:

```python
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### OpenAI mode

Can optionally use OpenAI embeddings:

```python
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

This allows development without API cost while keeping the project ready for production API-based usage.

---

## 🧠 LLM / Answer Generation Modes

The project supports:

```python
LLM_PROVIDER = "fallback"   # options: "fallback", "openai"
```

### Fallback mode

Works without an API key.

It uses:

- retrieved sources
- rule-based business logic
- out-of-scope detection
- source-grounded responses

### OpenAI mode

Can be enabled later for full natural-language answer generation.

---

## ⚠️ Out-of-Scope Detection

The system includes a guardrail for unsupported questions.

Example unsupported query:

```text
Can employees approve salary increases for team members?
```

Expected behavior:

```text
I could not find this procedure in the available company documents.
This question appears to be outside the current Retail Store SOP knowledge base.
Recommended action: escalate to HR or the store manager.
```

This prevents the chatbot from inventing unsupported policies.

---

## 📊 Evaluation

### Retrieval Performance

The retrieval system is evaluated using business-specific queries.

Current results:

```text
Hit@1: 100%
Hit@3: 100%
```

Evaluation areas:

- Customer Service
- Cash Control
- Inventory Operations

### What the metrics mean

```text
Hit@1 = the correct document was retrieved as the first result
Hit@3 = the correct document was found within the top 3 results
```

---

## 🧪 Answer Quality Evaluation

The project includes answer evaluation logic to check:

- whether the correct source was retrieved
- whether out-of-scope questions are handled safely
- whether responses remain grounded in company documents

Output is saved to:

```text
outputs/evaluation/answer_evaluation.csv
```

---

## 💬 Example Questions

```text
What should I do if a customer returns a product without a receipt?
What should the cashier do at the end of the shift?
During stock receiving, what should employees do with damaged or missing items?
Can employees approve salary increases for team members?
```

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build vector index

```bash
python build_index.py
```

### 3. Run retrieval evaluation

```bash
python evaluate_retrieval.py
```

### 4. Run answer evaluation

```bash
python evaluate_answers.py
```

### 5. Run CLI chatbot

```bash
python -m src.chatbot
```

### 6. Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

## 🔑 Optional OpenAI Setup

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Then switch provider in `src/config.py`:

```python
EMBEDDING_PROVIDER = "openai"
LLM_PROVIDER = "openai"
```

---

## 🧠 Business Value

This project demonstrates how AI can support retail operations by:

- reducing employee training time
- standardizing store procedures
- minimizing operational mistakes
- supporting frontline employees
- improving decision consistency
- giving fast access to SOPs and internal policies

---

## 🏢 Example Business Use Cases

### 1. New employee onboarding

Employees can ask questions instead of searching through manuals.

### 2. Store procedure support

Cashiers and store staff can quickly check how to handle operational cases.

### 3. Customer service consistency

Employees can follow the same approved policy for returns, refunds, and damaged items.

### 4. Inventory control support

Store teams can check receiving, replenishment, and damaged goods procedures.

### 5. Manager decision support

Managers can use the chatbot as a quick reference tool during daily operations.

---

## 🚀 Future Improvements

Potential next steps:

- Add more SOP documents
- Add PDF manuals from real business processes
- Add local LLM support with Ollama
- Add OpenAI answer generation
- Add reranking for improved retrieval accuracy
- Add user feedback buttons
- Add admin document upload
- Add authentication
- Deploy the app on Streamlit Cloud or Hugging Face Spaces

---

## 👤 Author

Konstantinos Chasiotis  
FMCG & Logistics Professional transitioning into AI & Data roles.

---