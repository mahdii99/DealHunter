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


# Words shoppers write in Arabic for brands, kinds of products and units, with the English word stores use in their
# titles. Matched word by word (never inside another word). Used when no model is free to translate the request:
# the big stores' own search pages (English titles) are searched with these words.
AR_WORDS = {
    # brands
    "سامسونج": "samsung", "شاومي": "xiaomi", "ريلمي": "realme", "اوبو": "oppo", "أوبو": "oppo", "ابل": "apple",
    "آبل": "apple", "ايفون": "iphone", "آيفون": "iphone", "هواوي": "huawei", "انفنكس": "infinix",
    "إنفينكس": "infinix", "انفينكس": "infinix", "تكنو": "tecno", "هونر": "honor", "نوكيا": "nokia", "فيفو": "vivo",
    "موتورولا": "motorola", "لينوفو": "lenovo", "ديل": "dell", "ايسر": "acer", "أيسر": "acer", "اسوس": "asus",
    "أسوس": "asus", "توشيبا": "toshiba", "شارب": "sharp", "تورنيدو": "tornado", "فريش": "fresh",
    "زانوسي": "zanussi", "بيكو": "beko", "اريستون": "ariston", "يونيون": "unionaire", "كاريير": "carrier",
    "جري": "gree", "هايسنس": "hisense", "سوني": "sony", "باناسونيك": "panasonic", "فيليبس": "philips",
    "براون": "braun", "انكر": "anker", "أنكر": "anker", "ساوندكور": "soundcore", "بوش": "bosch",
    "كينوود": "kenwood", "تيفال": "tefal", "هيتاشي": "hitachi", "ميديا": "midea", "شاومى": "xiaomi",
    # kinds of products
    "شاشة": "tv", "شاشه": "tv", "تلفزيون": "tv", "تليفزيون": "tv", "غسالة": "washing machine",
    "غساله": "washing machine", "تلاجة": "refrigerator", "تلاجه": "refrigerator", "ثلاجة": "refrigerator",
    "ثلاجه": "refrigerator", "تكييف": "air conditioner", "مكيف": "air conditioner", "ميكروويف": "microwave",
    "سخان": "water heater", "بوتاجاز": "cooker", "مكنسة": "vacuum", "مكنسه": "vacuum", "قلاية": "air fryer",
    "قلايه": "air fryer", "تابلت": "tablet", "ايباد": "ipad", "موبايل": "phone", "تليفون": "phone",
    "لابتوب": "laptop", "لاب": "laptop", "سماعة": "earbuds", "سماعه": "earbuds", "سماعات": "earbuds",
    "ساعة": "watch", "ساعه": "watch", "شاحن": "charger", "خلاط": "blender", "مكواة": "iron", "مكواه": "iron",
    "ديب": "freezer", "فريزر": "freezer", "مروحة": "fan", "مروحه": "fan", "كاميرا": "camera", "طابعة": "printer",
    "طابعه": "printer", "راوتر": "router", "بلايستيشن": "playstation",
    # units and model words
    "بوصة": "inch", "بوصه": "inch", "انش": "inch", "كيلو": "kg", "جيجا": "gb", "جيجابايت": "gb", "تيرا": "tb",
    "حصان": "hp", "لتر": "liter", "برو": "pro", "بلس": "plus", "ماكس": "max", "الترا": "ultra", "ألترا": "ultra",
    "لايت": "lite", "ميني": "mini", "اوتوماتيك": "automatic", "أوتوماتيك": "automatic", "اتوماتيك": "automatic",
    "كريستال": "crystal", "سمارت": "smart", "انفرتر": "inverter", "ديجيتال": "digital", "وايرلس": "wireless",
}
_UNITS = {"inch", "kg", "gb", "tb", "hp", "liter"}


def english_words(text: str) -> str:
    """The shopper's request in the English words stores use in their titles, with no model: each Arabic word we
    know replaced by its English word, Latin words and numbers kept, numbers joined to their unit ("55 بوصة" ->
    "55inch"); Arabic words we don't know are left out. '' when nothing is left."""
    out: list[str] = []
    for w in normalize(text).split():
        w = w.strip(".,!?؟،()")
        w = w[2:] if w.startswith("ال") and w[2:] in AR_WORDS else w
        eng = AR_WORDS.get(w) or (w if re.fullmatch(r"[a-z0-9][a-z0-9+.\-]*", w) else "")
        if not eng or eng in STOPWORDS:
            continue
        if eng in _UNITS and out and out[-1].isdigit():
            out[-1] += " " + eng
        elif eng not in out:
            out.append(eng)
    return " ".join(out)
