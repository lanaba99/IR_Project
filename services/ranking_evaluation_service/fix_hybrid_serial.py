import pandas as pd
import pytrec_eval
import time
from services.retrieval_service.retrieval import RetrievalEngine

SAMPLE_SIZE = 1000
RANDOM_SEED = 42
TOP_K = 10


def build_qrels_dict():
    qrels_df = pd.read_csv("data/raw/qrels.csv")
    qrels = {}
    for _, row in qrels_df.iterrows():
        qid, did, rel = str(row["query_id"]), str(row["doc_id"]), int(row["relevance"])
        qrels.setdefault(qid, {})[did] = rel
    return qrels


def get_sampled_queries(qrels):
    queries_df = pd.read_csv("data/raw/test_queries.csv")
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    queries_with_qrels = queries_df[queries_df["query_id"].isin(qrels.keys())]
    n_sample = min(SAMPLE_SIZE, len(queries_with_qrels))
    sampled = queries_with_qrels.sample(n=n_sample, random_state=RANDOM_SEED)
    return sampled


def main():
    qrels = build_qrels_dict()
    sampled_queries = get_sampled_queries(qrels)

    print("Loading retrieval engine...")
    engine = RetrievalEngine()

    print(f"\nإعادة تقييم Hybrid-Serial على نفس {len(sampled_queries)} query (seed={RANDOM_SEED})...")

    run = {}
    start = time.time()
    for i, row in enumerate(sampled_queries.itertuples(), 1):
        qid, text = str(row.query_id), row.text
        results = engine.search_hybrid_serial(text, top_k=TOP_K)
        run[qid] = {str(doc_id): float(score) for doc_id, score in results}

        if i % 50 == 0:
            elapsed = time.time() - start
            print(f"  [Hybrid-Serial] {i}/{len(sampled_queries)} - {elapsed:.1f}s elapsed")

    elapsed_total = time.time() - start

    empty_count = sum(1 for r in run.values() if len(r) == 0)
    print(f"\nعدد الـ queries يلي رجعوا نتائج فاضية: {empty_count}/{len(run)}")
    if empty_count == len(run):
        print("⚠️ المشكلة لسا موجودة! تأكدي من تطبيق التعديل على retrieval.py صحيح.")
        return

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recall", "ndcg", "P_10"})
    results = evaluator.evaluate(run)
    metrics_df = pd.DataFrame(results).T

    new_row = {
        "method": "Hybrid-Serial",
        "MAP": metrics_df["map"].mean(),
        "Recall": metrics_df.filter(like="recall").mean().mean(),
        "P@10": metrics_df["P_10"].mean(),
        "nDCG": metrics_df["ndcg"].mean(),
        "time_seconds": round(elapsed_total, 2),
        "n_queries": len(sampled_queries),
    }

    print("\n" + "=" * 50)
    print("النتيجة المصححة لـ Hybrid-Serial:")
    print("=" * 50)
    for k, v in new_row.items():
        print(f"  {k}: {v}")

    results_path = "data/evaluation/results.csv"
    df = pd.read_csv(results_path)
    df = df[df["method"] != "Hybrid-Serial"]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(results_path, index=False)

    print(f"\nتم تحديث {results_path} بالنتيجة المصححة.")
    print("\nالجدول النهائي الكامل:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()