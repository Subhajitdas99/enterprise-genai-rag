from src.retrieval.retriever import faiss_retrieve
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import hybrid_retrieve

# ✅ load bm25 retriever only once
bm25 = BM25Retriever()


def retrieve(query: str, retriever_type: str = "faiss", k: int = 3):
    retriever_type = retriever_type.lower()

    if retriever_type == "faiss":
        return faiss_retrieve(query, k=k)

    elif retriever_type == "bm25":
        return bm25.retrieve(query, k=k)

    elif retriever_type == "hybrid":
        return hybrid_retrieve(query, k=k)

    else:
        raise ValueError(f"Unknown retriever_type: {retriever_type}")




