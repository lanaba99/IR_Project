"""
UI Service - Streamlit
========================
واجهة المستخدم الرئيسية لمحرك البحث (IR System).

تحقق الشروط الإلزامية بالتوصيف:
- اختيار dataset قبل البحث
- قبول query من المستخدم
- اختيار طريقة التمثيل (TF-IDF / BM25 / Embedding / Hybrid Serial / Hybrid Parallel)
- التحكم بمعاملات BM25 (k1, b) من الواجهة مباشرة
- التحكم بأوزان الـ Hybrid Parallel (weights) من الواجهة
- اختيار Hybrid Serial أو Parallel
- عرض نتائج البحث مع ID الوثيقة + المحتوى الأصلي الكامل
- خيار "أساسي فقط" مقابل "أساسي + إضافي" (تفعيل/تعطيل الميزات الإضافية)

تشغيل الواجهة:
    streamlit run ui/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os

# عشان نقدر نستورد من services/ مهما كان مكان تشغيل streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.retrieval_service.retrieval import RetrievalEngine
from services.query_refinement_service.refinement import refine_query

st.set_page_config(page_title="IR System - محرك البحث", layout="wide")


# ============================================
# تحميل الـ Engine مرة وحدة فقط (Streamlit caching)
# ============================================
@st.cache_resource
def load_engine():
    return RetrievalEngine()


@st.cache_resource
def load_clustering_data():
    try:
        clusters_df = pd.read_csv("data/clustering/doc_clusters.csv")
        clusters_df["doc_id"] = clusters_df["doc_id"].astype(str)
        return clusters_df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_topics_data():
    try:
        topics_df = pd.read_csv("data/topics/doc_topics.csv")
        topics_df["doc_id"] = topics_df["doc_id"].astype(str)
        return topics_df
    except FileNotFoundError:
        return None


# ============================================
# الشريط الجانبي (Sidebar) - كل عناصر التحكم
# ============================================
st.sidebar.title(" إعدادات البحث")

# 1) اختيار Dataset (شرط إلزامي بالتوصيف)
dataset_choice = st.sidebar.selectbox(
    "اختر مجموعة البيانات (Dataset)",
    ["BEIR / Quora (522K documents)"],
)

st.sidebar.markdown("---")

# 2) اختيار طريقة التمثيل
method = st.sidebar.radio(
    "طريقة التمثيل والبحث (Representation Method)",
    ["TF-IDF", "BM25", "Embedding", "Hybrid - Serial", "Hybrid - Parallel"],
)

st.sidebar.markdown("---")

# 3) خيار أساسي فقط / أساسي + إضافي (شرط إلزامي بالتوصيف)
execution_mode = st.sidebar.radio(
    "وضع التنفيذ (Execution Mode)",
    ["أساسي فقط (Core Only)", "أساسي + إضافي (Core + Bonus Features)"],
)
enable_bonus = execution_mode.startswith("أساسي +")

# 4) معاملات BM25 (شرط إلزامي - لازم تتحكم بيها بالواجهة)
bm25_k1, bm25_b = 1.5, 0.75
if method == "BM25":
    st.sidebar.markdown("### معاملات BM25")
    bm25_k1 = st.sidebar.slider(
        "k1 (تشبع تكرار الكلمة - term frequency saturation)",
        min_value=0.1, max_value=3.0, value=1.5, step=0.1,
        help="قيمة أعلى = تأثير أكبر لتكرار الكلمة بالوثيقة على الدرجة النهائية"
    )
    bm25_b = st.sidebar.slider(
        "b (تأثير طول الوثيقة - length normalization)",
        min_value=0.0, max_value=1.0, value=0.75, step=0.05,
        help="قيمة أعلى = عقاب أكبر للوثائق الطويلة (تطبيع حسب الطول)"
    )

# 5) أوزان Hybrid Parallel (شرط إلزامي - لازم تتحكم بيها بالواجهة)
w_tfidf, w_bm25, w_embed = 0.3, 0.3, 0.4
fusion_method = "weighted"
if method == "Hybrid - Parallel":
    st.sidebar.markdown("### إعدادات الدمج (Fusion) - Hybrid Parallel")
    fusion_method = st.sidebar.selectbox(
        "طريقة دمج النتائج (Fusion Method)",
        ["weighted", "rrf"],
        help="weighted = دمج الدرجات بأوزان | rrf = Reciprocal Rank Fusion (حسب الرتبة بدل الدرجة)"
    )
    if fusion_method == "weighted":
        w_tfidf = st.sidebar.slider("وزن TF-IDF", 0.0, 1.0, 0.3, 0.05)
        w_bm25 = st.sidebar.slider("وزن BM25", 0.0, 1.0, 0.3, 0.05)
        w_embed = st.sidebar.slider("وزن Embedding", 0.0, 1.0, 0.4, 0.05)
        total_w = w_tfidf + w_bm25 + w_embed
        if total_w > 0:
            st.sidebar.caption(f"مجموع الأوزان الحالي: {total_w:.2f} (يفضل = 1.0)")

# 6) عدد النتائج
top_k = st.sidebar.slider("عدد النتائج (Top-K)", min_value=5, max_value=50, value=10)

st.sidebar.markdown("---")

# Query Refinement (شرط أساسي بالتوصيف - تصحيح لغوي/مرادفات)
st.sidebar.markdown("###  تحسين الاستعلام (Query Refinement)")
enable_spell_correction = st.sidebar.checkbox("تصحيح إملائي (Spell Correction)", value=False)
enable_synonym_expansion = st.sidebar.checkbox("إضافة مرادفات (Synonym Expansion)", value=False)

st.sidebar.markdown("---")

# 7) فلاتر إضافية (بونص: Clustering + Topic Detection) - تظهر فقط بوضع "أساسي + إضافي"
selected_cluster = None
selected_topic = None
if enable_bonus:
    clusters_df = load_clustering_data()
    topics_df = load_topics_data()

    st.sidebar.markdown("###  ميزات إضافية (Bonus)")

    if clusters_df is not None:
        cluster_options = ["الكل (All Clusters)"] + sorted(clusters_df["cluster"].unique().tolist())
        selected_cluster_label = st.sidebar.selectbox("فلترة حسب Cluster", cluster_options)
        if selected_cluster_label != "الكل (All Clusters)":
            selected_cluster = selected_cluster_label
    else:
        st.sidebar.caption(" نتائج Clustering غير موجودة - شغّلي cluster_documents.py أولاً")

    if topics_df is not None:
        topic_options = ["الكل (All Topics)"] + sorted(topics_df["dominant_topic"].unique().tolist())
        selected_topic_label = st.sidebar.selectbox("فلترة حسب Topic", topic_options)
        if selected_topic_label != "الكل (All Topics)":
            selected_topic = selected_topic_label
    else:
        st.sidebar.caption(" نتائج Topic Detection غير موجودة - شغّلي topic_detection.py أولاً")

    use_faiss = st.sidebar.checkbox("استخدام FAISS Vector Store (بونص)", value=True,
                                      help="إلغاء التفعيل = بحث embedding بـ brute-force عادي (cosine_similarity) للمقارنة")
else:
    use_faiss = True


# ============================================
# المحتوى الرئيسي
# ============================================
st.title(" محرك استرجاع المعلومات (IR System)")
st.caption(f"Dataset: {dataset_choice}  |  Method: {method}  |  Mode: {execution_mode}")

query = st.text_input("اكتبي استعلام البحث (Query):", placeholder="مثال: how to learn machine learning")

col1, col2 = st.columns([1, 4])
with col1:
    search_clicked = st.button(" بحث", type="primary", use_container_width=True)

if search_clicked and query.strip():
    engine = load_engine()

    # -------------------------------
    # تطبيق Query Refinement (إذا مفعّلة)
    # -------------------------------
    original_query = query
    if enable_spell_correction or enable_synonym_expansion:
        refined_query, refinement_report = refine_query(
            query,
            use_spell_correction=enable_spell_correction,
            use_synonym_expansion=enable_synonym_expansion
        )

        with st.expander(" تفاصيل تحسين الاستعلام (Query Refinement)", expanded=True):
            st.write(f"**الاستعلام الأصلي:** `{refinement_report['original']}`")
            st.write(f"**بعد التطبيع الأساسي** (lowercase + إزالة رموز + إزالة stop words): `{refinement_report['normalized']}`")

            if enable_spell_correction:
                if refinement_report["spelling_changes"]:
                    changes_str = ", ".join([f"`{old}` → `{new}`" for old, new in refinement_report["spelling_changes"]])
                    st.write(f"**تصحيحات إملائية:** {changes_str}")
                else:
                    st.write("**تصحيحات إملائية:** لا يوجد أخطاء إملائية مكتشفة")

            if enable_synonym_expansion:
                if refinement_report["synonyms_added"]:
                    syn_str = " | ".join([f"`{w}` → {', '.join(syns)}" for w, syns in refinement_report["synonyms_added"].items()])
                    st.write(f"**مرادفات مُضافة:** {syn_str}")
                else:
                    st.write("**مرادفات مُضافة:** لا يوجد")

            st.write(f"**الاستعلام النهائي المستخدم بالبحث:** `{refinement_report['final_query']}`")

        query = refined_query

    start_time = time.time()

    # -------------------------------
    # تنفيذ البحث حسب الطريقة المختارة
    # -------------------------------
    if method == "TF-IDF":
        results = engine.search_tfidf(query, top_k=top_k)
    elif method == "BM25":
        results = engine.search_bm25(query, top_k=top_k, k1=bm25_k1, b=bm25_b)
    elif method == "Embedding":
        results = engine.search_embedding(query, top_k=top_k, use_faiss=use_faiss)
    elif method == "Hybrid - Serial":
        results = engine.search_hybrid_serial(query, top_k=top_k)
    elif method == "Hybrid - Parallel":
        results = engine.search_hybrid_parallel(
            query, top_k=top_k,
            weight_tfidf=w_tfidf, weight_bm25=w_bm25, weight_embed=w_embed,
            fusion=fusion_method
        )

    elapsed = time.time() - start_time

    # -------------------------------
    # فلترة حسب Cluster / Topic (إذا مفعّلة)
    # -------------------------------
    if enable_bonus:
        if selected_cluster is not None:
            clusters_df = load_clustering_data()
            allowed_ids = set(clusters_df[clusters_df["cluster"] == selected_cluster]["doc_id"])
            results = [(d, s) for d, s in results if str(d) in allowed_ids]
        if selected_topic is not None:
            topics_df = load_topics_data()
            allowed_ids = set(topics_df[topics_df["dominant_topic"] == selected_topic]["doc_id"])
            results = [(d, s) for d, s in results if str(d) in allowed_ids]

    st.success(f"تم إيجاد {len(results)} نتيجة بزمن {elapsed:.3f} ثانية")

    if len(results) == 0:
        st.info("لم يتم إيجاد نتائج. حاولي استعلاماً أطول أو أكثر تحديداً "
                 "(الكلمات القصيرة جداً أو الشائعة جداً قد لا تكفي للبحث).")

    # -------------------------------
    # عرض النتائج (ID + المحتوى الأصلي الكامل - شرط إلزامي)
    # -------------------------------
    for rank, (doc_id, score) in enumerate(results, 1):
        content = engine.get_document_content(doc_id)
        with st.expander(f"#{rank} | Doc ID: `{doc_id}` | Score: {score:.4f}"):
            st.write(content if content else " محتوى الوثيقة غير موجود")

elif search_clicked:
    st.warning("لازم تكتبي استعلام بحث أولاً")


# ============================================
# تبويب التقييم (عرض نتائج evaluate.py لو موجودة)
# ============================================
st.markdown("---")
with st.expander(" عرض نتائج التقييم (Evaluation Results)"):
    eval_path = "data/evaluation/results.csv"
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        st.dataframe(eval_df, use_container_width=True)
    else:
        st.info("لسا ما تم تشغيل evaluate.py")