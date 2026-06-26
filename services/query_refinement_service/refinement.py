"""
Query Refinement Service
==========================
بونص/متطلب أساسي: تحسين استعلام المستخدم قبل البحث عبر:

1) تصحيح إملائي (Spell Correction) - باستخدام مكتبة pyspellchecker الجاهزة
   مثال: "algoritmhs" -> "algorithms"

2) توسيع بالمرادفات (Synonym Expansion) - باستخدام WordNet (مكتبة NLTK جاهزة)
   مثال: "best" -> يضيف مرادفات مثل "good", "finest"

كل التحسينات اختيارية (toggle) وقابلة للتجربة كل واحدة لحالها من الواجهة،
حسب طلب التوصيف: "تطبيق تحسينات على الاستعلام... تصحيح الاستعلام لغوياً
أو إضافة مرادفات".
"""

import re
from spellchecker import SpellChecker
from nltk.corpus import wordnet, stopwords

spell = SpellChecker()
stop_words = set(stopwords.words("english"))


def basic_normalize(query: str) -> str:
    """
    خطوة تطبيع أساسية (نفس منطق clean_query بملف retrieval.py):
    lowercase + إزالة الرموز/الترقيم + إزالة stop words.
    نطبقها هنا أولاً عشان التقرير المعروض بالواجهة يعكس فعلياً
    النص يلي رح يدخل على البحث (بدل ما يظهر "the" أو "@" بالتقرير
    بينما هم أصلاً محذوفين بمرحلة لاحقة من clean_query).
    """
    text = query.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def correct_spelling(query: str) -> tuple[str, list[tuple[str, str]]]:
    """
    تصحيح إملائي لكل كلمة بالاستعلام.
    يرجع: (النص المصحح، قائمة التغييرات [(كلمة قديمة, كلمة جديدة), ...])
    """
    words = re.findall(r"[a-zA-Z]+", query.lower())
    corrected_words = []
    changes = []

    for word in words:
        if len(word) <= 2:  # كلمات قصيرة جداً (مثل "is", "a") نتجاهل تصحيحها
            corrected_words.append(word)
            continue

        corrected = spell.correction(word)
        if corrected and corrected != word:
            corrected_words.append(corrected)
            changes.append((word, corrected))
        else:
            corrected_words.append(word)

    corrected_query = " ".join(corrected_words)
    return corrected_query, changes


def expand_with_synonyms(query: str, max_synonyms_per_word=1) -> tuple[str, dict]:
    """
    توسيع الاستعلام بمرادفات من WordNet.
    يرجع: (النص الموسّع، dict {كلمة: [مرادفات]})
    """
    words = query.split()
    expanded_words = list(words)  # نبدأ بنسخة من الكلمات الأصلية
    synonyms_found = {}

    for word in words:
        synsets = wordnet.synsets(word)
        word_synonyms = set()

        for syn in synsets[:2]:  # نقتصر على أول 2 synsets لتجنب ضوضاء كتيرة
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower() and synonym.isalpha():
                    word_synonyms.add(synonym.lower())

        word_synonyms = list(word_synonyms)[:max_synonyms_per_word]
        if word_synonyms:
            synonyms_found[word] = word_synonyms
            expanded_words.extend(word_synonyms)

    expanded_query = " ".join(expanded_words)
    return expanded_query, synonyms_found


def refine_query(query: str, use_spell_correction=True, use_synonym_expansion=False):
    """
    الدالة الرئيسية: تطبق التحسينات المختارة بالترتيب وترجع تقرير كامل
    عن كل خطوة (مفيد لعرضها بالواجهة كـ "before/after").

    الترتيب:
    0) Normalization أساسي (lowercase + إزالة رموز + إزالة stop words)
    1) تصحيح إملائي (اختياري)
    2) توسيع بمرادفات (اختياري)
    """
    report = {"original": query}

    normalized_query = basic_normalize(query)
    report["normalized"] = normalized_query
    current_query = normalized_query

    if use_spell_correction:
        corrected_query, spelling_changes = correct_spelling(current_query)
        report["spell_corrected"] = corrected_query
        report["spelling_changes"] = spelling_changes
        current_query = corrected_query
    else:
        report["spell_corrected"] = current_query
        report["spelling_changes"] = []

    if use_synonym_expansion:
        expanded_query, synonyms = expand_with_synonyms(current_query)
        report["synonym_expanded"] = expanded_query
        report["synonyms_added"] = synonyms
        current_query = expanded_query
    else:
        report["synonym_expanded"] = current_query
        report["synonyms_added"] = {}

    report["final_query"] = current_query
    return current_query, report


if __name__ == "__main__":
    # اختبار سريع بنفس استعلام لانا يلي فيه typo
    test_query = "the machine learning algoritmhs for data science"
    final_q, report = refine_query(test_query, use_spell_correction=True, use_synonym_expansion=True)

    print("الاستعلام الأصلي:", report["original"])
    print("بعد التصحيح الإملائي:", report["spell_corrected"])
    print("التغييرات الإملائية:", report["spelling_changes"])
    print("بعد إضافة المرادفات:", report["synonym_expanded"])
    print("المرادفات المضافة:", report["synonyms_added"])
    print("\nالاستعلام النهائي:", final_q)