from fastapi import FastAPI, UploadFile
import shutil
import os
import time

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.embeddings.embedder import build_index
from src.retrieval.router import retrieve
from src.llm.generator import generate_answer

from mlops.mlflow_logger import setup_mlflow, log_rag_request


app = FastAPI(title="Enterprise GenAI RAG")

DATA_DIR = "data/documents"
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Setup mlflow once on startup
setup_mlflow()


@app.post("/upload")
async def upload_pdf(file: UploadFile):
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = load_pdf(file_path)
    chunks = chunk_text(pages, chunk_size=800, overlap=150)

    build_index(chunks)

    return {"status": "indexed", "file": file.filename}


@app.get("/ask")
def ask(query: str, retriever_type: str = "faiss"):

    start = time.time()

    contexts = retrieve(query, retriever_type=retriever_type, k=3)
    answer = generate_answer(query, contexts)

    latency = time.time() - start

    # ✅ log to MLflow
    log_rag_request(
        query=query,
        retriever=retriever_type,
        contexts=contexts,
        answer=answer,
        latency=latency
    )

    return {
        "answer": answer,
        "sources": contexts,
        "retriever": retriever_type,
        "latency_sec": latency
    }
