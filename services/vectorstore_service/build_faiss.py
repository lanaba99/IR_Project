import numpy as np
import faiss
import os

def build_faiss_index():
    os.makedirs("data/index/faiss", exist_ok=True)
    out_path = "data/index/faiss/doc_index.faiss"

    if os.path.exists(out_path):
        print("FAISS index already built, skipping (cache).")
        return

    data = np.load("data/index/embeddings/doc_embeddings.npz")
    embeddings = data["embeddings"].astype("float32")

    print(f"Building FAISS index for {embeddings.shape[0]} vectors...")
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (تساوي cosine لأنه normalized)
    index.add(embeddings)

    faiss.write_index(index, out_path)
    print("Saved FAISS index.")

if __name__ == "__main__":
    build_faiss_index()