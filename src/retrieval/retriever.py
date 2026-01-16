from sentence_transformers import SentenceTransformer
from src.embeddings.index_manager import load_index

model = SentenceTransformer("all-MiniLM-L6-v2")

def faiss_retrieve(query: str, k: int = 5):
    index, metadata = load_index()

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        doc = metadata[idx]

        results.append({
            "text": doc.get("text", ""),
            "page": doc.get("page", -1),
            "score": float(distances[0][rank]),   # faiss gives distance
            "source": "faiss"
        })

    return results



