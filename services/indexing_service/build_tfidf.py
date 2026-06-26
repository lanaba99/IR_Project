import pandas as pd
import os
import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf():
    os.makedirs("data/index/tfidf", exist_ok=True)
    matrix_path = "data/index/tfidf/tfidf_matrix.npz"
    vectorizer_path = "data/index/tfidf/tfidf_vectorizer.joblib"

    if os.path.exists(matrix_path) and os.path.exists(vectorizer_path):
        print("TF-IDF already built, skipping (cache).")
        return

    docs = pd.read_parquet("data/processed/clean_documents.parquet")
    print(f"Building TF-IDF for {len(docs)} documents...")

    vectorizer = TfidfVectorizer(max_features=100000)
    matrix = vectorizer.fit_transform(docs["clean_text"].fillna(""))

    sparse.save_npz(matrix_path, matrix)
    joblib.dump(vectorizer, vectorizer_path, compress=3)
    docs[["doc_id"]].to_csv("data/index/tfidf/doc_ids.csv", index=False)

    print(f"TF-IDF matrix shape: {matrix.shape}")
    print("Saved TF-IDF index (compressed).")

if __name__ == "__main__":
    build_tfidf()