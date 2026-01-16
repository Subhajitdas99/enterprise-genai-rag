import json
import time
import mlflow

from src.retrieval.retriever import retrieve
from src.llm.generator import generate_answer
from mlops.evaluation.metrics import (
    recall_at_k,
    faithfulness,
    hallucination_rate
)

mlflow.set_experiment("rag-evaluation")

with open("data/eval/questions.json") as f:
    questions = json.load(f)

recall_scores = []
faith_scores = []
hallucination_scores = []
latencies = []

with mlflow.start_run(run_name="rag_eval_v1"):

    for step, item in enumerate(questions):
        question = item["question"]
        expected = item["expected_keywords"]

        contexts = retrieve(question)

        start = time.time()
        answer = generate_answer(question, contexts)
        latency = time.time() - start

        recall = recall_at_k(contexts, expected)
        faith = faithfulness(answer, contexts)
        hallucination = hallucination_rate(answer, contexts)

        # ---- per-question metrics ----
        mlflow.log_metric("recall_at_5", recall, step=step)
        mlflow.log_metric("faithfulness", faith, step=step)
        mlflow.log_metric("hallucination_rate", hallucination, step=step)
        mlflow.log_metric("latency_sec", latency, step=step)

        recall_scores.append(recall)
        faith_scores.append(faith)
        hallucination_scores.append(hallucination)
        latencies.append(latency)

        # ---- log artifacts for debugging ----
        mlflow.log_text(
            json.dumps({
                "question": question,
                "answer": answer,
                "contexts": contexts
            }, indent=2),
            artifact_file=f"samples/sample_{step}.json"
        )

    # ---- aggregated metrics ----
    mlflow.log_metric("avg_recall_at_5", sum(recall_scores) / len(recall_scores))
    mlflow.log_metric("avg_faithfulness", sum(faith_scores) / len(faith_scores))
    mlflow.log_metric(
        "avg_hallucination_rate",
        sum(hallucination_scores) / len(hallucination_scores)
    )
    mlflow.log_metric("avg_latency_sec", sum(latencies) / len(latencies))

    # ---- run metadata ----
    mlflow.log_param("retriever", "faiss")
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
    mlflow.log_param("llm", "local-causal-llm")
    mlflow.log_param("eval_samples", len(questions))




