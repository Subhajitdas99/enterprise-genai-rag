import os
import time
import mlflow


EXPERIMENT_NAME = "rag-api-monitoring"


def setup_mlflow():
    # local sqlite tracking
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)


def log_rag_request(query: str, retriever: str, contexts: list, answer: str, latency: float):
    """
    Logs one RAG API call as a MLflow run.
    """
    with mlflow.start_run(run_name=f"ask:{retriever}"):

        # ---- params ----
        mlflow.log_param("retriever", retriever)
        mlflow.log_param("query", query)

        # ---- metrics ----
        mlflow.log_metric("latency_sec", latency)
        mlflow.log_metric("num_contexts", len(contexts))

        # ---- artifacts (debugging) ----
        mlflow.log_text(query, "query.txt")
        mlflow.log_text(answer, "answer.txt")

        mlflow.log_text(
            "\n\n---\n\n".join([str(c) for c in contexts]),
            "contexts.txt"
        )
