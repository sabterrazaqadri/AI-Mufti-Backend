"""
Retrieval smoke-test: run known masail questions (Urdu, Roman Urdu, English)
against the RAG store and print what gets retrieved with scores, so citation
quality can be judged before deploying.

Usage: python verify_rag.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import rag  # noqa: E402

QUESTIONS = [
    # Urdu script
    "غسل کے فرائض کتنے ہیں؟",
    "پانی کی کتنی قسمیں ہیں اور کون سا پانی وضو کے لیے جائز ہے؟",
    "تیمم کب جائز ہے؟",
    "موزوں پر مسح کی مدت کتنی ہے؟",
    "نماز میں قبلہ کی طرف منہ کرنا کیوں ضروری ہے؟",
    # Roman Urdu
    "ghusl ke faraiz kitne hain?",
    "musafir ki namaz ka kya hukum hai?",
    "azan ka jawab dena kaisa hai?",
    "masjid mein dunya ki baatein karna kaisa hai?",
    "zakat kis par farz hai?",
    "roza kin cheezon se toot jata hai?",
    "mayyat ko ghusl dene ka tareeqa kya hai?",
    "eid ki namaz ka waqt kya hai?",
    # English
    "What breaks wudu?",
    "Is sajda sahw required if I forget qa'da?",
]


def main():
    for q in QUESTIONS:
        print("=" * 70)
        print("Q:", q)
        passages = rag.retrieve(q)
        if not passages:
            print("  (no passages above threshold)")
            continue
        for p in passages:
            print(f"  [{p['score']:.3f}] {p['reference']}")
            print(f"          {p['content'][:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
