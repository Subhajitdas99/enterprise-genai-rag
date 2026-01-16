# ----------------------------
# MLflow bootstrap (CRITICAL)
# ----------------------------
import os
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import json
import time
import mlflow

from src.retrieval.router import retrieve
from src.llm.generator import generate_answer
from mlops.evaluation.metrics import (
    recall_at_k,
    faithfulness,
    hallucination_rate,
)

# ----------------------------
# Safety utils
# ----------------------------
def normalize_context(ctx):
    if ctx is None:
        return ""
    if isinstance(ctx, dict):
        if "text" in ctx:
            return normalize_context(ctx["text"])
        return str(ctx)
    if isinstance(ctx, list):
        return " ".join(normalize_context(x) for x in ctx)
    return str(ctx)


def truncate_text(text: str, max_tokens: int = 450):
    return str(text)[: max_tokens * 4]


# ----------------------------
# Experiment setup
# ----------------------------
EXPERIMENT_NAME = "rag-ab-testing"

if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
    mlflow.create_experiment(EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)

# ----------------------------
# Config
# ----------------------------
RETRIEVERS = ["faiss", "bm25", "hybrid"]

TOP_K = 5
MAX_CONTEXT_TOKENS = 450

with open("data/eval/questions.json") as f:
    questions = json.load(f)

if not questions:
    raise RuntimeError("No evaluation questions found")

# ----------------------------
# A/B Evaluation Loop
# ----------------------------
for retriever_type in RETRIEVERS:

    with mlflow.start_run(run_name=f"retriever={retriever_type}"):

        mlflow.set_tag("experiment_type", "ab_testing")
        mlflow.set_tag("retriever", retriever_type)

        recalls, faiths, hallucinations, latencies = [], [], [], []

        for step, item in enumerate(questions):
            question = item["question"]
            expected = item["expected_keywords"]

            start = time.time()
            raw_contexts = retrieve(question, retriever_type)

            contexts = []
            for ctx in raw_contexts[:TOP_K]:
                text = normalize_context(ctx).strip()
                if not text:
                    continue
                contexts.append(truncate_text(text, MAX_CONTEXT_TOKENS))

            if not contexts:
                contexts = [""]

            answer = generate_answer(question, contexts)
            latency = time.time() - start

            recall = recall_at_k(contexts, expected)
            faith = faithfulness(answer, contexts)
            halluc = hallucination_rate(answer, contexts)

            mlflow.log_metric("recall_at_5", recall, step=step)
            mlflow.log_metric("faithfulness", faith, step=step)
            mlflow.log_metric("hallucination_rate", halluc, step=step)
            mlflow.log_metric("latency_sec", latency, step=step)

            recalls.append(recall)
            faiths.append(faith)
            hallucinations.append(halluc)
            latencies.append(latency)

        n = max(len(recalls), 1)

        mlflow.log_metric("avg_recall_at_5", sum(recalls) / n)
        mlflow.log_metric("avg_faithfulness", sum(faiths) / n)
        mlflow.log_metric("avg_hallucination_rate", sum(hallucinations) / n)
        mlflow.log_metric("avg_latency_sec", sum(latencies) / n)

        mlflow.log_param("retriever", retriever_type)
        mlflow.log_param("eval_samples", len(questions))
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_param("context_truncation", f"{MAX_CONTEXT_TOKENS}_tokens")





