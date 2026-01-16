import mlflow
import os
from src.embeddings.embedder import embed_chunks
from src.embeddings.index_manager import save_index

from mlops.mlflow.utils import setup_mlflow

def track_embedding_run(
    chunks,
    embedding_model: str,
    chunk_size: int,
    overlap: int
):
    setup_mlflow()

    with mlflow.start_run():
        # -------- PARAMETERS --------
        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("overlap", overlap)

        # -------- EMBEDDING --------
        embeddings = embed_chunks(chunks)

        # -------- METRICS --------
        mlflow.log_metric("num_chunks", len(chunks))
        mlflow.log_metric("embedding_dim", embeddings.shape[1])

        # -------- SAVE FAISS --------
        index_path = save_index(embeddings, chunks)

        # -------- ARTIFACT --------
        mlflow.log_artifacts(index_path, artifact_path="faiss_index")

        print("✅ MLflow run logged successfully")
