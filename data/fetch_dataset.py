import ir_datasets
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

print("Loading Quora dataset via ir_datasets...")
dataset = ir_datasets.load("beir/quora")

# 1) Documents (522K documents) 
docs = []
for doc in dataset.docs_iter():
    docs.append({"doc_id": doc.doc_id, "text": doc.text})
docs_df = pd.DataFrame(docs)
docs_df.to_parquet("data/raw/documents.parquet")
print(f"Documents saved: {len(docs_df)}")

# 2) Test queries + qrels 
test_dataset = ir_datasets.load("beir/quora/test")

queries = [{"query_id": q.query_id, "text": q.text} for q in test_dataset.queries_iter()]
queries_df = pd.DataFrame(queries)
queries_df.to_csv("data/raw/test_queries.csv", index=False)
print(f"Test queries saved: {len(queries_df)}")

qrels = [{"query_id": q.query_id, "doc_id": q.doc_id, "relevance": q.relevance} for q in test_dataset.qrels_iter()]
qrels_df = pd.DataFrame(qrels)
qrels_df.to_csv("data/raw/qrels.csv", index=False)
print(f"Qrels saved: {len(qrels_df)}")