import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def build_embeddings():
    os.makedirs("data/index/embeddings", exist_ok=True)
    out_path = "data/index/embeddings/doc_embeddings.npz"

    if os.path.exists(out_path):
        print("Embeddings already built, skipping (cache).")
        return

    docs = pd.read_parquet("data/processed/clean_documents.parquet")
    print(f"Encoding {len(docs)} documents with SentenceTransformer (CPU)...")

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    texts = docs["text"].fillna("").tolist()  # النص الأصلي أفضل لـ embeddings من النص بعد إزالة stopwords

    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.savez_compressed(out_path, embeddings=embeddings, doc_ids=docs["doc_id"].values)
    print(f"Saved embeddings, shape: {embeddings.shape}")

if __name__ == "__main__":
    build_embeddings()