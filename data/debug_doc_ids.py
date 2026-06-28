import pandas as pd
import numpy as np

print("=== TF-IDF doc_ids ===")
tfidf_ids = pd.read_csv("data/index/tfidf/doc_ids.csv")["doc_id"].tolist()
print("Sample:", tfidf_ids[:3])
print("Type:", type(tfidf_ids[0]))
print()

print("=== Embedding doc_ids ===")
emb_data = np.load("data/index/embeddings/doc_embeddings.npz", allow_pickle=True)
emb_ids = emb_data["doc_ids"]
print("Sample:", emb_ids[:3])
print("Type:", type(emb_ids[0]))
print()

print("=== هل في تطابق فعلي؟ ===")
test_id = tfidf_ids[0]
print(f"tfidf_id = {test_id!r} (type: {type(test_id)})")
print(f"موجود بالـ emb_ids كـ set مباشرة؟ -> {test_id in set(emb_ids)}")
print(f"موجود بالـ emb_ids بعد str() على الطرفين؟ -> {str(test_id) in set(str(x) for x in emb_ids)}")