from fastapi import UploadFile
import shutil
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text
from mlops.mlflow.track_embeddings import track_embedding_run


def upload_and_index(file: UploadFile):
    path = f"data/documents/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = load_pdf(path)
    chunks = chunk_text(pages, chunk_size=500, overlap=100)

    track_embedding_run(
        chunks=chunks,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=500,
        overlap=100
    )

    return {"status": "indexed with mlflow"}
