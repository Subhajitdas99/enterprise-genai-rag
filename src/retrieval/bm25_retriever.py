import pickle
import os

BM25_DIR = "data/bm25"
DOC_PATH = os.path.join(BM25_DIR, "docs.pkl")
BM25_PATH = os.path.join(BM25_DIR, "bm25.pkl")


class BM25Retriever:
    def __init__(self):
        if not os.path.exists(DOC_PATH) or not os.path.exists(BM25_PATH):
            raise RuntimeError(
                f"BM25 index not found.\nExpected:\n{DOC_PATH}\n{BM25_PATH}\n"
                "Run: python -m mlops.evaluation.build_bm25"
            )

        with open(DOC_PATH, "rb") as f:
            self.documents = pickle.load(f)

        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

    def retrieve(self, query, k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = []
        for i in top_k:
            doc = self.documents[i]

            # doc can be dict OR string (safety)
            if isinstance(doc, dict):
                text = doc.get("text", "")
                page = doc.get("page", -1)
            else:
                text = str(doc)
                page = -1

            results.append({
                "text": text,
                "page": page,
                "score": float(scores[i]),
                "source": "bm25"
            })

        return results





