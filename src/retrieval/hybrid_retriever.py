import hashlib
from src.retrieval.retriever import faiss_retrieve
from src.retrieval.bm25_retriever import BM25Retriever

bm25 = BM25Retriever()


def minmax_normalize(scores, reverse=False):
    """
    reverse=True => lower score is better (FAISS distance)
    reverse=False => higher score is better (BM25)
    """
    if not scores:
        return []

    s_min = min(scores)
    s_max = max(scores)

    if s_max == s_min:
        return [1.0] * len(scores)

    normalized = []
    for s in scores:
        v = (s - s_min) / (s_max - s_min)
        if reverse:
            v = 1 - v
        normalized.append(float(v))

    return normalized


def _doc_key(d):
    """Stable dedupe key using page + text hash"""
    text = str(d.get("text", ""))
    page = str(d.get("page", -1))
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{page}_{h}"


def hybrid_retrieve(query: str, k: int = 5, alpha: float = 0.6):
    """
    alpha = weight for FAISS semantic score
    (1-alpha) = weight for BM25 keyword score
    """

    faiss_docs = faiss_retrieve(query, k=k)
    bm25_docs = bm25.retrieve(query, k=k)

    # ---- normalize faiss (distance: smaller is better) ----
    faiss_scores = [float(d.get("score", 0.0)) for d in faiss_docs]
    faiss_norm = minmax_normalize(faiss_scores, reverse=True)

    # ---- normalize bm25 (bigger is better) ----
    bm25_scores = [float(d.get("score", 0.0)) for d in bm25_docs]
    bm25_norm = minmax_normalize(bm25_scores, reverse=False)

    # attach normalized scores
    for i, d in enumerate(faiss_docs):
        d["norm_score"] = faiss_norm[i]
        d["source"] = "faiss"

    for i, d in enumerate(bm25_docs):
        d["norm_score"] = bm25_norm[i]
        d["source"] = "bm25"

    # ---- merge + weighted score ----
    merged = {}

    # FAISS first
    for d in faiss_docs:
        key = _doc_key(d)
        merged[key] = {
            **d,
            "final_score": alpha * d["norm_score"],
        }

    # add BM25
    for d in bm25_docs:
        key = _doc_key(d)

        if key in merged:
            merged[key]["final_score"] += (1 - alpha) * d["norm_score"]
            merged[key]["source"] = "hybrid(faiss+bm25)"
        else:
            merged[key] = {
                **d,
                "final_score": (1 - alpha) * d["norm_score"],
                "source": "hybrid(bm25)",
            }

    ranked = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)

    return ranked[:k]

