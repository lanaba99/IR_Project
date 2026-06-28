import pandas as pd
import numpy as np
import pytrec_eval
import time
import json
import os
from services.retrieval_service.retrieval import RetrievalEngine

# ============================================
# إعدادات Sampling 
# ============================================
SAMPLE_SIZE = 1000          # عدد الـ queries يلي رح تُختبر (من أصل 10,000)
RANDOM_SEED = 42           # ثابت عشان النتيجة تتكرر لو شغلتي الكود مرة ثانية
TOP_K = 10                 # عدد النتائج المُسترجعة لكل query


def build_qrels_dict():
    qrels_df = pd.read_csv("data/raw/qrels.csv")
    qrels = {}
    for _, row in qrels_df.iterrows():
        qid, did, rel = str(row["query_id"]), str(row["doc_id"]), int(row["relevance"])
        qrels.setdefault(qid, {})[did] = rel
    return qrels


def get_sampled_queries(qrels):
    """
    نختار عينة عشوائية من test_queries.csv، لكن فقط من الأسئلة
    يلي عندها qrels فعلياً (عشان ما نضيع queries بدون تقييم).
    """
    queries_df = pd.read_csv("data/raw/test_queries.csv")
    queries_df["query_id"] = queries_df["query_id"].astype(str)

    # فلترة: نخلي بس الأسئلة يلي إلها qrels
    queries_with_qrels = queries_df[queries_df["query_id"].isin(qrels.keys())]

    print(f"إجمالي test queries: {len(queries_df)}")
    print(f"Queries عندها qrels: {len(queries_with_qrels)}")

    n_sample = min(SAMPLE_SIZE, len(queries_with_qrels))
    sampled = queries_with_qrels.sample(n=n_sample, random_state=RANDOM_SEED)

    print(f"تم اختيار عينة عشوائية: {n_sample} query (seed={RANDOM_SEED})")
    return sampled


def evaluate_method(engine, method_name, search_fn, queries_df, qrels, top_k=TOP_K):
    run = {}
    start = time.time()

    for i, row in enumerate(queries_df.itertuples(), 1):
        qid, text = str(row.query_id), row.text
        results = search_fn(text, top_k=top_k)
        run[qid] = {str(doc_id): float(score) for doc_id, score in results}

        if i % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{method_name}] {i}/{len(queries_df)} queries - {elapsed:.1f}s elapsed")

    elapsed_total = time.time() - start

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recall", "ndcg", "P_10"})
    results = evaluator.evaluate(run)

    metrics_df = pd.DataFrame(results).T
    summary = {
        "method": method_name,
        "MAP": metrics_df["map"].mean(),
        "Recall": metrics_df.filter(like="recall").mean().mean(),
        "P@10": metrics_df["P_10"].mean(),
        "nDCG": metrics_df["ndcg"].mean(),
        "time_seconds": round(elapsed_total, 2),
        "n_queries": len(queries_df),
    }
    print(f"  -> {method_name} خلص بـ {elapsed_total:.1f} ثانية\n")
    return summary


def run_full_evaluation():
    print("=" * 50)
    print(f"Evaluation Mode: SAMPLED ({SAMPLE_SIZE} queries, seed={RANDOM_SEED})")
    print("=" * 50)

    qrels = build_qrels_dict()
    sampled_queries = get_sampled_queries(qrels)

    engine = RetrievalEngine()
    all_results = []

    methods = [
        ("TF-IDF", engine.search_tfidf),
        ("BM25", engine.search_bm25),
        ("Embedding", engine.search_embedding),
        ("Hybrid-Serial", engine.search_hybrid_serial),
        ("Hybrid-Parallel", engine.search_hybrid_parallel),
    ]

    for name, fn in methods:
        print(f"\n--- تقييم: {name} ---")
        result = evaluate_method(engine, name, fn, sampled_queries, qrels)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    os.makedirs("data/evaluation", exist_ok=True)
    df.to_csv("data/evaluation/results.csv", index=False)

    print("\n" + "=" * 50)
    print("النتائج النهائية:")
    print("=" * 50)
    print(df.to_string(index=False))

    # حفظ معلومات الـ sampling بالتقرير
    with open("data/evaluation/sampling_info.json", "w") as f:
        json.dump({
            "sample_size": SAMPLE_SIZE,
            "random_seed": RANDOM_SEED,
            "top_k": TOP_K,
            "note": "Evaluation conducted on a random sample of queries from the official BEIR/Quora test set due to computational constraints. Indexing was performed on the complete corpus (522K documents) without any reduction."
        }, f, indent=2)

    return df


if __name__ == "__main__":
    run_full_evaluation()