📄 Enterprise GenAI RAG (FastAPI + FAISS + BM25 + Hybrid) + MLflow Monitoring

Enterprise-ready Retrieval Augmented Generation (RAG) system built with:

✅ FastAPI backend for upload + Q&A
✅ PDF ingestion + chunking
✅ FAISS semantic retrieval (Sentence Transformers)
✅ BM25 keyword retrieval
✅ Hybrid retrieval (FAISS + BM25 combined)
✅ Flan-T5 LLM answer generation
✅ MLflow monitoring + evaluation + A/B testing
✅ Streamlit UI for chat

🚀 Features
✅ Document Ingestion

Upload any PDF document

Extract and clean text (fix broken words, remove extra spaces)

Chunk the text with overlap for better retrieval quality

✅ Retrieval Options

You can query using 3 retriever modes:

Retriever	Type	Best For
faiss	Semantic	Meaning-based search
bm25	Keyword	Exact matching search
hybrid	Combined	Best overall performance
✅ Answer Generation

Uses google/flan-t5-base

Generates short + clean answers

Falls back safely if context missing

✅ MLflow Monitoring & Evaluation

Logs metrics like:

recall_at_5

faithfulness

hallucination_rate

latency_sec

Supports:

RAG evaluation pipeline

A/B evaluation for multiple retrievers

📂 Project Structure
enterprise-genai-rag/
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── __init__.py
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── chunker.py
│   │   └── __init__.py
│   │
│   ├── embeddings/
│   │   ├── embedder.py
│   │   ├── index_manager.py
│   │   └── __init__.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── router.py
│   │   └── __init__.py
│   │
│   ├── llm/
│   │   ├── generator.py
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   └── text.py
│   │
│   └── __init__.py
│
├── mlops/
│   ├── evaluation/
│   │   ├── run_eval.py
│   │   ├── run_ab_eval.py
│   │   ├── build_bm25.py
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   ├── mlflow/
│   │   ├── monitoring.py
│   │   ├── utils.py
│   │   └── __init__.py
│   │
│   ├── mlflow_logger.py
│   └── __init__.py
│
├── ui/
│   └── app.py
│
├── data/
│   ├── documents/
│   ├── bm25/
│   └── eval/questions.json
│
├── vector_store/
│   └── (faiss index saved here)
│
├── mlflow.db
├── requirements.txt
└── README.md

⚙️ Setup Instructions
✅ 1) Clone Repository
git clone https://github.com/Subhajitdas99/enterprise-genai-rag.git
cd enterprise-genai-rag

✅ 2) Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

✅ 3) Install Dependencies
pip install -r requirements.txt

▶️ Run FastAPI Server

Start the API:

uvicorn src.api.main:app --reload


Now open Swagger Docs:

✅ http://127.0.0.1:8000/docs

📤 Upload PDF
Endpoint

POST /upload

Upload a PDF file through Swagger UI or Streamlit UI.

✅ Response:

{
  "status": "indexed",
  "file": "yourfile.pdf"
}

❓ Ask Questions (RAG)
Endpoint

GET /ask

Example:

http://127.0.0.1:8000/ask?query=What%20is%20the%20registration%20date%3F&retriever_type=faiss

Query Params
Param	Description
query	User question
retriever_type	faiss / bm25 / hybrid

✅ Response example:

{
  "answer": "...",
  "sources": [
    {
      "page": 1,
      "text": "...",
      "score": 1.62,
      "source": "faiss"
    }
  ],
  "retriever": "faiss"
}

🖥️ Run Streamlit UI
streamlit run ui/app.py

Open:
✅ http://localhost:8501
📊 MLflow Monitoring
✅ Start MLflow UI

mlflow ui --workers 1

Open:
✅ http://127.0.0.1:5000
✅ Build BM25 Index (Required for BM25 / Hybrid)

Run:

python -m mlops.evaluation.build_bm25

✅ Output:

✅ BM25 index built successfully

This creates:

data/bm25/docs.pkl
data/bm25/bm25.pkl

✅ RAG Evaluation (Single Retriever)

Runs evaluation using your dataset:

python -m mlops.evaluation.run_eval


Metrics logged to MLflow:
✅ recall_at_5
✅ faithfulness
✅ latency_sec
✅ hallucination_rate

✅ A/B Evaluation (FAISS vs BM25 vs Hybrid)

Run:

python -m mlops.evaluation.run_ab_eval


This will create multiple MLflow runs:
✅ retriever=faiss
✅ retriever=bm25
✅ retriever=hybrid

✅ RAG Evaluation (Single Retriever)

Runs evaluation using your dataset:

python -m mlops.evaluation.run_eval


Metrics logged to MLflow:
✅ recall_at_5
✅ faithfulness
✅ latency_sec
✅ hallucination_rate

📌 Notes / Troubleshooting
✅ HuggingFace timeouts

If model loads slow, run once manually:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
✅ Page file / memory error (Windows)

If you see:
paging file too small
✅ Increase Virtual Memory in Windows settings.

✅ Future Improvements (Next Steps)

⭐ Add reranker (CrossEncoder)
⭐ Store embeddings in a DB (Chroma / Qdrant)
⭐ Add conversation memory (chat history RAG)
⭐ Add Docker support
⭐ Add CI/CD pipeline + GitHub Actions
⭐ Add proper eval dataset + leaderboard

👨‍💻Author
Subhajit Das
https://github.com/Subhajitdas99/enterprise-genai-rag
