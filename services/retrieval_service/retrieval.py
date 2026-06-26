import numpy as np
import pandas as pd
import joblib
import pickle
import gzip
import faiss
import sqlite3
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_query(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


class RetrievalEngine:
    def __init__(self):
        print("Loading all indexes into memory (one-time)...")
        # TF-IDF
        self.tfidf_vectorizer = joblib.load("data/index/tfidf/tfidf_vectorizer.joblib")
        self.tfidf_matrix = sparse.load_npz("data/index/tfidf/tfidf_matrix.npz")
        self.tfidf_doc_ids = pd.read_csv("data/index/tfidf/doc_ids.csv")["doc_id"].astype(str).tolist()
        
        # BM25
        with gzip.open("data/index/bm25/bm25_index.pkl.gz", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_doc_ids = bm25_data["doc_ids"]

        # Embeddings + FAISS
        emb_data = np.load("data/index/embeddings/doc_embeddings.npz", allow_pickle=True)
        self.doc_embeddings = emb_data["embeddings"]
        self.emb_doc_ids = emb_data["doc_ids"]
        self.faiss_index = faiss.read_index("data/index/faiss/doc_index.faiss")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # نص الوثائق الأصلي (لعرضه كامل بالواجهة)
        #self.documents = pd.read_parquet("data/raw/documents.parquet").set_index("doc_id")
        self.db_path = "data/documents.db"

        print("Retrieval engine ready.")

    def search_tfidf(self, query: str, top_k=10):
        clean_q = clean_query(query)
        query_vec = self.tfidf_vectorizer.transform([clean_q])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.tfidf_doc_ids[i], float(scores[i])) for i in top_idx]

    def search_bm25(self, query: str, top_k=10, k1=1.5, b=0.75):
        self.bm25.k1 = k1
        self.bm25.b = b
        clean_q = clean_query(query)
        tokenized_q = clean_q.split()
        scores = self.bm25.get_scores(tokenized_q)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.bm25_doc_ids[i], float(scores[i])) for i in top_idx]

    def search_embedding(self, query: str, top_k=10, use_faiss=True):
        query_vec = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        if use_faiss:
            scores, idx = self.faiss_index.search(query_vec, top_k)
            return [(self.emb_doc_ids[i], float(scores[0][j])) for j, i in enumerate(idx[0])]
        else:
            sims = cosine_similarity(query_vec, self.doc_embeddings).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(self.emb_doc_ids[i], float(sims[i])) for i in top_idx]

    def _normalize_scores(self, results):
        if not results:
            return results
        scores = np.array([s for _, s in results])
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return [(d, 1.0) for d, s in results]
        return [(d, (s - min_s) / (max_s - min_s)) for d, s in results]

    def search_hybrid_serial(self, query: str, top_k=10, prefilter_k=100):
        # المرحلة 1: TF-IDF يفلتر مجموعة أولية
        prefiltered = self.search_tfidf(query, top_k=prefilter_k)
        prefiltered_ids = set(d for d, _ in prefiltered)

        # المرحلة 2: نعيد ترتيب هاي المجموعة بـ embeddings (re-ranking)
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        id_to_idx = {doc_id: i for i, doc_id in enumerate(self.emb_doc_ids)}
        rescored = []
        for doc_id in prefiltered_ids:
            if doc_id in id_to_idx:
                emb = self.doc_embeddings[id_to_idx[doc_id]]
                sim = float(np.dot(query_vec[0], emb))
                rescored.append((doc_id, sim))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]

    def search_hybrid_parallel(self, query: str, top_k=10, weight_tfidf=0.3, weight_bm25=0.3, weight_embed=0.4, fusion="weighted"):
        tfidf_res = self._normalize_scores(self.search_tfidf(query, top_k=50))
        bm25_res = self._normalize_scores(self.search_bm25(query, top_k=50))
        embed_res = self._normalize_scores(self.search_embedding(query, top_k=50))

        combined_scores = {}

        if fusion == "weighted":
            for doc_id, score in tfidf_res:
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weight_tfidf * score
            for doc_id, score in bm25_res:
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weight_bm25 * score
            for doc_id, score in embed_res:
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + weight_embed * score

        elif fusion == "rrf":  # Reciprocal Rank Fusion
            k_rrf = 60
            for rank_list in [tfidf_res, bm25_res, embed_res]:
                for rank, (doc_id, _) in enumerate(rank_list):
                    combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 1.0 / (k_rrf + rank + 1)

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_document_content(self, doc_id: str):
        """
        استرجاع محتوى الوثيقة الأصلي الكامل من قاعدة بيانات SQLite
        مباشرة بالـ doc_id (بدل تحميل كل الوثائق بالـ RAM).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (str(doc_id),))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None