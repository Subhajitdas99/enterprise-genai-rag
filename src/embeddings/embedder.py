from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("vector_store/faiss_index", exist_ok=True)
    faiss.write_index(index, "vector_store/faiss_index/index.bin")

    with open("vector_store/faiss_index/meta.pkl", "wb") as f:
        pickle.dump(chunks, f)
