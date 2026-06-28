import pandas as pd

docs = pd.read_parquet("data/raw/documents.parquet")
queries = pd.read_csv("data/raw/test_queries.csv")
qrels = pd.read_csv("data/raw/qrels.csv")

print("=== Documents ===")
print(f"Total: {len(docs)}")
print(docs.head(5))
print()

print("=== Test Queries ===")
print(f"Total: {len(queries)}")
print(queries.head(5))
print()

print("=== Qrels ===")
print(f"Total: {len(qrels)}")
print(qrels.head(5))