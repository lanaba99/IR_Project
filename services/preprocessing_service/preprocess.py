import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def preprocess_documents():
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/clean_documents.parquet"

    if os.path.exists(out_path):
        print("Clean documents already exist, skipping (cache).")
        return pd.read_parquet(out_path)

    docs = pd.read_parquet("data/raw/documents.parquet")
    print(f"Preprocessing {len(docs)} documents...")

    docs["clean_text"] = docs["text"].apply(clean_text)
    docs.to_parquet(out_path)
    print("Saved clean_documents.parquet")
    return docs

def preprocess_queries():
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/clean_queries.csv"

    if os.path.exists(out_path):
        print("Clean queries already exist, skipping (cache).")
        return pd.read_csv(out_path)

    queries = pd.read_csv("data/raw/test_queries.csv")
    queries["clean_text"] = queries["text"].apply(clean_text)
    queries.to_csv(out_path, index=False)
    print("Saved clean_queries.csv")
    return queries

if __name__ == "__main__":
    preprocess_documents()
    preprocess_queries()