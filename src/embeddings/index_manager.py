import faiss
import pickle
import os

INDEX_DIR = "vector_store/faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.bin")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")


def build_index(embeddings, metadata):
    os.makedirs(INDEX_DIR, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def load_index():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(
            "FAISS index not found. Upload a document first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


