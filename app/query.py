"""Turn a free-text request (Arabic or English) into a structured query.

Rule-based by default (free, no API). If GEMINI_API_KEY is set, app.llm
can be used first and this module is the fallback.
"""
import re

from .models import ParsedQuery
from .utils import normalize

# Small Arabic -> English vocabulary so stores (which index English titles) find results.
AR_VOCAB = {
    "باور بانك": "power bank", "باوربانك": "power bank", "شاحن متنقل": "power bank",
    "بطارية متنقلة": "power bank", "بطاريه متنقله": "power bank",
    "شاحن": "charger", "كابل": "cable", "وصلة": "cable", "سماعة": "earbuds", "سماعه": "earbuds",
    "ايربودز": "earbuds", "سماعات": "earbuds", "ساعة ذكية": "smart watch", "ساعه ذكيه": "smart watch",
    "موبايل": "phone", "تليفون": "phone", "لابتوب": "laptop", "ماوس": "mouse", "كيبورد": "keyboard",
    "سريع": "fast", "لاسلكي": "wireless", "وايرلس": "wireless",
}
STOPWORDS = {
    "عايز", "عاوز", "محتاج", "ابغى", "اريد", "ارخص", "أرخص", "رخيص", "حلو", "كويس", "في", "من", "على",
    "علي", "لو", "سمحت", "ممكن", "يا", "ال", "the", "a", "an", "i", "want", "need", "cheap", "cheapest",
    "best", "for", "with", "please", "find", "me", "جنيه", "ج", "egp", "le",
}
BUDGET_MAX = r"(?:تحت|اقل من|أقل من|ميزانيه|ميزانية|لحد|under|below|less than|max|<)\s*(\d[\d,]*)"
BUDGET_MIN = r"(?:فوق|اكتر من|أكثر من|over|above|more than|min|>)\s*(\d[\d,]*)"
UNIT_TOKEN = re.compile(r"^\d+(?:\.\d+)?(mah|w|gb|tb|mp|hz|inch)$")


def parse_query(raw: str) -> ParsedQuery:
    text = normalize(raw.strip())
    max_price = min_price = None
    if m := re.search(BUDGET_MAX, text):
        max_price = float(m.group(1).replace(",", ""))
        text = text.replace(m.group(0), " ")
    if m := re.search(BUDGET_MIN, text):
        min_price = float(m.group(1).replace(",", ""))
        text = text.replace(m.group(0), " ")

    # Arabic capacity phrasing: "10000 مللي" / "10 الاف مللي"
    text = re.sub(r"(\d+)\s*(?:الاف|آلاف|ألف|الف)\s*(?:مللي|ملي|مل)", lambda m: f" {int(m.group(1)) * 1000}mah ", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:مللي|ملي)(?:\s*(?:امبير|أمبير))?", r" \1mah ", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*وات", r" \1w ", text)
    text = re.sub(r"(\d+)\s*جيجا", r" \1gb ", text)
    text = re.sub(r"\bpowerbank\b", "power bank", text)

    for ar in sorted(AR_VOCAB, key=len, reverse=True):
        text = text.replace(ar, f" {AR_VOCAB[ar]} ")

    tokens = [t.strip(".,!?؟،") for t in text.split()]
    tokens = [t for t in tokens if t and t not in STOPWORDS]

    must_have = [t for t in tokens if UNIT_TOKEN.match(t)]
    keywords = [t for t in tokens if t not in must_have and not re.search(r"[؀-ۿ]", t)]
    # keep untranslated Arabic words only if nothing else is left
    if not keywords:
        keywords = [t for t in tokens if t not in must_have]
    return ParsedQuery(raw=raw, keywords=keywords, must_have=must_have, max_price=max_price, min_price=min_price)
