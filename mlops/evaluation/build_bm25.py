import os
import pickle
from rank_bm25 import BM25Okapi

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text

DATA_DIR = "data/documents"
OUT_DIR = "data/bm25"
os.makedirs(OUT_DIR, exist_ok=True)

documents = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        pages = load_pdf(os.path.join(DATA_DIR, file))
        chunks = chunk_text(pages)
        documents.extend(chunks)

# ✅ FIX: extract text field
tokenized_docs = [doc["text"].lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

with open(os.path.join(OUT_DIR, "docs.pkl"), "wb") as f:
    pickle.dump(documents, f)

with open(os.path.join(OUT_DIR, "bm25.pkl"), "wb") as f:
    pickle.dump(bm25, f)

print("✅ BM25 index built successfully")

