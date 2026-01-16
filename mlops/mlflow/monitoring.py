import os
import time
import mlflow


# ✅ Use local sqlite file always
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def setup_mlflow(experiment_name="rag-api-monitoring"):
    """
    Ensures experiment exists and sets it.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)


def log_api_request(
    query: str,
    retriever_type: str,
    answer: str,
    sources: list,
    latency: float,
):
    """
    Logs one /ask request as one MLflow run.
    """

    with mlflow.start_run(run_name=f"api_request={retriever_type}"):

        # --- Params ---
        mlflow.log_param("retriever_type", retriever_type)
        mlflow.log_param("query_length", len(query))
        mlflow.log_param("num_sources", len(sources))

        # --- Metrics ---
        mlflow.log_metric("latency_sec", latency)

        # --- Tags ---
        mlflow.set_tag("component", "fastapi")
        mlflow.set_tag("endpoint", "/ask")

        # --- Artifacts (save answer & sources) ---
        mlflow.log_text(query, "query.txt")
        mlflow.log_text(answer, "answer.txt")
        mlflow.log_text(str(sources), "sources.txt")
