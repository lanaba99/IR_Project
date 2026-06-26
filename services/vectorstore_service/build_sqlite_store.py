"""
Document Store Service - SQLite
=================================
بناء قاعدة بيانات SQLite لتخزين الوثائق الأصلية (documents) بشكل دائم،
بحيث نقدر نسترجع محتوى أي وثيقة (Top-K results) مباشرة من قاعدة البيانات
بالـ doc_id، بدل تحميل ملف parquet كامل بالـ RAM في كل مرة.

ليش SQLite؟
- ملف واحد (.db) بدون حاجة لسيرفر منفصل (مناسب لمشروع أكاديمي/SOA بسيط)
- مدعوم built-in بمكتبة sqlite3 الجاهزة بـ Python
- استعلام سريع جداً بالـ doc_id لأنه عمود مفهرس (INDEX)
- يحقق متطلب "التخزين بقاعدة بيانات" بدون تعقيد إضافي (PostgreSQL/MySQL غير ضروريين هنا)

شغليه مرة واحدة فقط لبناء القاعدة:
    python services\\vectorstore_service\\build_sqlite_store.py
"""

import sqlite3
import pandas as pd
import os
import time

DB_PATH = "data/documents.db"


def build_sqlite_store():
    if os.path.exists(DB_PATH):
        print(f"قاعدة البيانات موجودة مسبقاً: {DB_PATH} (cache) — حذفيها يدوياً لو بدك تبنيها من جديد.")
        return

    print("Loading documents.parquet...")
    docs = pd.read_parquet("data/raw/documents.parquet")
    print(f"عدد الوثائق: {len(docs)}")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    print(f"Building SQLite database at {DB_PATH}...")
    start = time.time()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # إنشاء الجدول
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            content TEXT NOT NULL
        )
    """)

    # إدخال البيانات (executemany أسرع بكثير من insert لكل صف لحاله)
    docs_records = list(zip(docs["doc_id"].astype(str), docs["text"].astype(str)))
    cursor.executemany("INSERT OR REPLACE INTO documents (doc_id, content) VALUES (?, ?)", docs_records)

    # إنشاء index على doc_id لتسريع البحث (رغم إنه PRIMARY KEY بالفعل بيعمل index ضمنياً، نوضحه صراحة)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)")

    conn.commit()
    conn.close()

    elapsed = time.time() - start
    print(f"تم بناء قاعدة البيانات بـ {elapsed:.2f} ثانية")
    print(f"عدد السجلات المخزّنة: {len(docs_records)}")


def test_lookup(sample_doc_id=None):
    """دالة سريعة للتأكد إنه القاعدة شغالة صحيح"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if sample_doc_id is None:
        cursor.execute("SELECT doc_id, content FROM documents LIMIT 1")
    else:
        cursor.execute("SELECT doc_id, content FROM documents WHERE doc_id = ?", (str(sample_doc_id),))

    row = cursor.fetchone()
    conn.close()

    if row:
        print(f"\nاختبار الاسترجاع:")
        print(f"  doc_id: {row[0]}")
        print(f"  content (أول 200 حرف): {row[1][:200]}...")
    else:
        print("لم يتم إيجاد الوثيقة")


if __name__ == "__main__":
    build_sqlite_store()
    test_lookup()