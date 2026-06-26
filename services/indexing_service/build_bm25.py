import pandas as pd
import pickle
import os
import gzip
from rank_bm25 import BM25Okapi

def build_bm25():
    os.makedirs("data/index/bm25", exist_ok=True)
    out_path = "data/index/bm25/bm25_index.pkl.gz"

    if os.path.exists(out_path):
        print("BM25 already built, skipping (cache).")
        return

    docs = pd.read_parquet("data/processed/clean_documents.parquet")
    print(f"Building BM25 for {len(docs)} documents...")

    tokenized_corpus = [text.split() for text in docs["clean_text"].fillna("")]
    bm25 = BM25Okapi(tokenized_corpus)

    with gzip.open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": docs["doc_id"].tolist()}, f)

    print("Saved BM25 index (compressed).")

if __name__ == "__main__":
    build_bm25()