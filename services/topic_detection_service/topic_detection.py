"""
Topic Detection Service
========================
بونص: اكتشاف المواضيع (Topics) الكامنة بمجموعة الوثائق باستخدام LDA
(Latent Dirichlet Allocation) من مكتبة scikit-learn.

ليش LDA بالذات؟
- مكتبة جاهزة وموثوقة (sklearn.decomposition.LatentDirichletAllocation)
- بيشتغل مباشرة على نفس التمثيل المبني مسبقاً (CountVectorizer مشابه لـ TF-IDF)
- نتائجه قابلة للتفسير: كل Topic = توزيع احتمالي على الكلمات

طريقة العمل:
1) نبني CountVectorizer جديد (LDA بالأصل مصمم للعمل على raw counts، مش TF-IDF
   لأنه TF-IDF بيقلل الترددات وهذا يأثر على افتراضات Dirichlet الإحصائية)
2) نطبق LDA لاستخراج N topics
3) نطبع أهم الكلمات لكل topic
4) نحسب توزيع الوثائق على الـ Topics (لكل وثيقة، أي Topic هو الغالب)
5) نرسم Bar chart لأهم كلمات كل Topic (مطلوب بالتوصيف: "شارتات التوبيكس")
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ============================================
# إعدادات قابلة للتعديل
# ============================================
N_TOPICS = 10          # عدد المواضيع المطلوب استخراجها
N_TOP_WORDS = 10        # عدد الكلمات الممثلة لكل Topic بالعرض
MAX_FEATURES = 5000     # تحديد حجم القاموس (vocabulary) لتسريع LDA على 522K مستند
RANDOM_STATE = 42


def build_topic_model(n_topics=N_TOPICS):
    os.makedirs("data/topics", exist_ok=True)

    model_path = "data/topics/lda_model.joblib"
    vectorizer_path = "data/topics/count_vectorizer.joblib"
    doc_topics_path = "data/topics/doc_topics.csv"

    docs = pd.read_parquet("data/processed/clean_documents.parquet")
    print(f"Building Topic Model (LDA) on {len(docs)} documents, k={n_topics} topics...")

    # -------------------------------
    # خطوة 1: Count Vectorizer (مش TF-IDF)
    # -------------------------------
    if os.path.exists(vectorizer_path):
        print("Count vectorizer موجودة مسبقاً (cache)، عم نحملها...")
        vectorizer = joblib.load(vectorizer_path)
        doc_term_matrix = vectorizer.transform(docs["clean_text"].fillna(""))
    else:
        vectorizer = CountVectorizer(
            max_features=MAX_FEATURES,
            max_df=0.95,   # نتجاهل كلمات موجودة بأكثر من 95% من الوثائق (شائعة جداً)
            min_df=5       # نتجاهل كلمات موجودة بأقل من 5 وثائق (نادرة جداً/أخطاء)
        )
        doc_term_matrix = vectorizer.fit_transform(docs["clean_text"].fillna(""))
        joblib.dump(vectorizer, vectorizer_path, compress=3)
        print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # -------------------------------
    # خطوة 2: تدريب LDA
    # -------------------------------
    if os.path.exists(model_path):
        print("LDA model موجود مسبقاً (cache)، عم نحمله...")
        lda = joblib.load(model_path)
    else:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=RANDOM_STATE,
            learning_method="online",  # أسرع بكثير على datasets كبيرة (522K)
            batch_size=2048,
            max_iter=10,
            n_jobs=-1  # استخدام كل الـ CPU cores المتاحة
        )
        print("Training LDA... (هاي الخطوة ممكن تاخد عدة دقائق على 522K مستند)")
        lda.fit(doc_term_matrix)
        joblib.dump(lda, model_path, compress=3)

    # -------------------------------
    # خطوة 3: توزيع الوثائق على الـ Topics
    # -------------------------------
    if os.path.exists(doc_topics_path):
        print("Document-topic assignments موجودة مسبقاً (cache).")
        doc_topics_df = pd.read_csv(doc_topics_path)
    else:
        print("Computing topic distribution per document...")
        doc_topic_dist = lda.transform(doc_term_matrix)  # (n_docs, n_topics)
        dominant_topic = doc_topic_dist.argmax(axis=1)

        doc_topics_df = pd.DataFrame({
            "doc_id": docs["doc_id"].values,
            "dominant_topic": dominant_topic
        })
        doc_topics_df.to_csv(doc_topics_path, index=False)

    return lda, vectorizer, doc_topics_df


def get_top_words_per_topic(lda, vectorizer, n_top_words=N_TOP_WORDS):
    """يرجع dict: {topic_id: [(word, weight), ...]}"""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        topics[topic_idx] = [(feature_names[i], float(topic[i])) for i in top_indices]
    return topics


def plot_topics(topics_dict, save_path="data/topics/topics_visualization.png"):
    """
    رسم Bar chart لكل Topic يعرض أهم الكلمات وأوزانها.
    هذا الرسم مطلوب صراحة بملاحظات الدكتورة: "شارتات التوبيكس"
    """
    n_topics = len(topics_dict)
    n_cols = 5
    n_rows = (n_topics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for topic_idx, words_weights in topics_dict.items():
        words = [w for w, _ in words_weights][::-1]
        weights = [wt for _, wt in words_weights][::-1]

        ax = axes[topic_idx]
        ax.barh(words, weights, color="steelblue")
        ax.set_title(f"Topic #{topic_idx}", fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)

    # حذف أي subplot فاضي زيادة
    for i in range(n_topics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved topics visualization: {save_path}")


def plot_topic_distribution(doc_topics_df, save_path="data/topics/topic_distribution.png"):
    """
    رسم Bar chart يوضح كم وثيقة تتبع كل Topic (توزيع الوثائق على المواضيع).
    """
    counts = doc_topics_df["dominant_topic"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values, color="coral")
    plt.xlabel("Topic ID")
    plt.ylabel("عدد الوثائق (Number of Documents)")
    plt.title("توزيع الوثائق على المواضيع (Document Distribution Across Topics)")
    plt.xticks(counts.index)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved topic distribution chart: {save_path}")


def run_topic_detection(n_topics=N_TOPICS):
    lda, vectorizer, doc_topics_df = build_topic_model(n_topics)
    topics_dict = get_top_words_per_topic(lda, vectorizer)

    print("\n" + "=" * 50)
    print("أهم الكلمات لكل Topic:")
    print("=" * 50)
    for topic_idx, words_weights in topics_dict.items():
        words_str = ", ".join([w for w, _ in words_weights])
        print(f"Topic #{topic_idx}: {words_str}")

    plot_topics(topics_dict)
    plot_topic_distribution(doc_topics_df)

    print("\nTopic detection خلصت. لازم تشوفي:")
    print("  - data/topics/topics_visualization.png")
    print("  - data/topics/topic_distribution.png")
    print("  - data/topics/doc_topics.csv")

    return lda, vectorizer, doc_topics_df, topics_dict


if __name__ == "__main__":
    run_topic_detection()