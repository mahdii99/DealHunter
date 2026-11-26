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
    # product lines written alone ("ريدمي نوت 14")
    "ريدمي": "redmi", "ردمي": "redmi", "بوكو": "poco", "جالاكسي": "galaxy", "جلاكسي": "galaxy", "جالكسي": "galaxy",
    "ايربودز": "airpods",
    # kinds of products
    "شاشة": "tv", "شاشه": "tv", "تلفزيون": "tv", "تليفزيون": "tv", "غسالة": "washing machine",
    "غساله": "washing machine", "تلاجة": "refrigerator", "تلاجه": "refrigerator", "ثلاجة": "refrigerator",
    "ثلاجه": "refrigerator", "تكييف": "air conditioner", "مكيف": "air conditioner", "ميكروويف": "microwave",
    "سخان": "water heater", "بوتاجاز": "cooker", "مكنسة": "vacuum", "مكنسه": "vacuum", "قلاية": "air fryer",
    "قلايه": "air fryer", "جراب": "case", "جرابات": "case", "كفر": "cover", "حافظة": "case", "اسكرينة": "protector",
    "اسكرينه": "protector", "لاصقة": "protector", "تابلت": "tablet", "ايباد": "ipad", "موبايل": "phone", "تليفون": "phone",
    "لابتوب": "laptop", "لاب": "laptop", "سماعة": "earbuds", "سماعه": "earbuds", "سماعات": "earbuds",
    "ساعة": "watch", "ساعه": "watch", "شاحن": "charger", "خلاط": "blender", "مكواة": "iron", "مكواه": "iron",
    "ديب": "freezer", "فريزر": "freezer", "مروحة": "fan", "مروحه": "fan", "كاميرا": "camera", "طابعة": "printer",
    "طابعه": "printer", "راوتر": "router", "بلايستيشن": "playstation",
    # units and model words
    "بوصة": "inch", "بوصه": "inch", "انش": "inch", "كيلو": "kg", "جيجا": "gb", "جيجابايت": "gb", "تيرا": "tb",
    "حصان": "hp", "لتر": "liter", "برو": "pro", "بلس": "plus", "ماكس": "max", "الترا": "ultra", "ألترا": "ultra",
    "لايت": "lite", "ميني": "mini", "نوت": "note", "نووت": "note", "اوتوماتيك": "automatic", "أوتوماتيك": "automatic", "اتوماتيك": "automatic",
    "ساخن": "heat", "سخن": "heat", "بارد": "cool", "نوفروست": "no frost", "ديفروست": "defrost",
    "كريستال": "crystal", "سمارت": "smart", "انفرتر": "inverter", "ديجيتال": "digital", "وايرلس": "wireless",
    "لاسلكي": "wireless", "لاسلكيه": "wireless", "بلوتوث": "bluetooth", "بلوتووث": "bluetooth",
    "العاب": "gaming", "الالعاب": "gaming", "للالعاب": "gaming", "للعاب": "gaming", "جيمنج": "gaming",
    "جيمينج": "gaming", "جيمنغ": "gaming", "جيمر": "gaming", "قيمنق": "gaming",
    "هيدفون": "headphones", "هيدسيت": "headset", "سبيكر": "speaker", "اسبيكر": "speaker",
    # accessories, so a request for one is understood as one (and its results aren't left out as accessories)
    "شنطة": "bag", "شنطه": "bag", "حقيبة": "bag", "حقيبه": "bag", "ستاند": "stand", "حامل": "holder",
    "منظف": "cleaner", "كابل": "cable", "وصلة": "cable", "وصله": "cable", "ماوس": "mouse", "كيبورد": "keyboard",
}
# Brands and words shoppers write as several Arabic words ("جي بي ال" = JBL): made one word before anything else.
AR_PHRASES = {
    "جي بي ال": "jbl", "جى بى ال": "jbl", "جي بي إل": "jbl", "ال جي": "lg", "ال جى": "lg", "إل جي": "lg",
    "اتش بي": "hp", "اتش بى": "hp", "إتش بي": "hp", "تي سي ال": "tcl", "تى سى ال": "tcl", "ام اس اي": "msi",
    "إم إس آي": "msi", "ون بلس": "oneplus", "وان بلس": "oneplus", "لاب توب": "لابتوب", "نو فروست": "نوفروست",
    "مكبر صوت": "speaker", "هيد فون": "هيدفون", "سمارت واتش": "smartwatch", "ساعة ذكية": "smartwatch",
    "ساعه ذكيه": "smartwatch",
}
_PHRASE_RE = re.compile("|".join(rf"(?<![\u0600-\u06ff]){re.escape(k)}(?![\u0600-\u06ff])"
                                 for k in sorted(AR_PHRASES, key=len, reverse=True)))


def phrased(text: str) -> str:
    """The shopper's text with multi-word names made one word ("سماعة جي بي ال" -> "سماعة jbl")."""
    return _PHRASE_RE.sub(lambda m: AR_PHRASES[m.group(0)], text or "")
_UNITS = {"inch", "kg", "gb", "tb", "hp", "liter"}
BRANDS = {"samsung", "xiaomi", "realme", "oppo", "apple", "iphone", "huawei", "infinix", "tecno", "honor", "nokia",
          "vivo", "motorola", "lenovo", "dell", "acer", "asus", "toshiba", "sharp", "tornado", "fresh", "zanussi",
          "beko", "ariston", "unionaire", "carrier", "gree", "hisense", "sony", "panasonic", "philips", "braun",
          "anker", "soundcore", "bosch", "kenwood", "tefal", "hitachi", "midea", "lg", "hp", "tcl", "jbl", "oneplus",
          "redmi", "poco", "msi", "haier", "white", "whirlpool", "ikon", "kiriazi", "olympic", "jac"} - {"white"}
_AR_UNITS = {"inch": "بوص|بوصه|بوصة|انش", "kg": "كيلو|كجم|كغ", "gb": "جيجا|جيجابايت", "tb": "تيرا", "hp": "حصان",
             "liter": "لتر"}
AR_UNITS = _AR_UNITS  # (the Arabic words of each unit, for matching sizes elsewhere)


def wanted_sizes(eng: str) -> list[tuple[str, str]]:
    """The sizes the request names in english_words' output ("55 inch", "10 kg"): (number, unit)."""
    return re.findall(r"(\d+(?:\.\d+)?) (" + "|".join(sorted(_UNITS)) + r")\b", eng)


def shows_size(title: str, number: str, unit: str) -> bool:
    """The title names this size, in English or Arabic ("55 Inch", "55-inch", '55"', "55 بوصة", "10 كجم")."""
    t = normalize(title)
    ar = _AR_UNITS.get(unit, unit)
    return bool(re.search(rf"(?<![\d.]){re.escape(number)}\s*(?:-\s*)?(?:{unit}|{ar}|\"|''|”)", t))


def english_words(text: str) -> str:
    """The shopper's request in the English words stores use in their titles, with no model: each Arabic word we
    know replaced by its English word, Latin words and numbers kept, numbers joined to their unit ("55 بوصة" ->
    "55inch"); Arabic words we don't know are left out. '' when nothing is left."""
    out: list[str] = []
    for w in normalize(phrased(text)).split():
        w = w.strip(".,!?؟،()")
        plus = len(w) > 1 and w.endswith("+")  # "برو+" / "pro+": the plus model
        w = w.rstrip("+") if plus else w
        w = w[2:] if w.startswith("ال") and w[2:] in AR_WORDS else w
        w = w[1:] if w.startswith("و") and w not in AR_WORDS and w[1:] in AR_WORDS else w  # "بارد وساخن": "and hot"
        eng = AR_WORDS.get(w) or (w if re.fullmatch(r"[a-z0-9][a-z0-9+.\-]*", w) else "")
        if not eng or eng in STOPWORDS:
            continue
        if eng in _UNITS and out and out[-1].isdigit():
            out[-1] += " " + eng
        elif eng not in out:
            out.append(eng)
        if plus and "plus" not in out:
            out.append("plus")
    return " ".join(out)


# The kinds of products a shopper names, with the words store titles use for each (English and Arabic, folded:
# ة->ه, أ->ا). With no model to judge, an offer whose title names another kind and not the one asked for is not
# what the shopper wants: a Sharp TV for "ثلاجة شارب", a fridge's "freezer" is fine since it says refrigerator too.
KIND_WORDS = {
    "tv": ("tv", "television", "شاشه", "تلفزيون", "تليفزيون"),
    "refrigerator": ("refrigerator", "fridge", "ثلاجه", "تلاجه", "ثلاجات", "تلاجات"),
    "washing machine": ("washing machine", "washer", "غساله"),
    "air conditioner": ("air conditioner", "conditioner", "split", "تكييف", "مكيف"),
    "microwave": ("microwave", "ميكروويف"),
    "water heater": ("water heater", "heater", "سخان"),
    "cooker": ("cooker", "gas stove", "بوتاجاز"),
    "vacuum": ("vacuum", "مكنسه"),
    "air fryer": ("air fryer", "fryer", "قلايه"),
    "tablet": ("tablet", "ipad", "تابلت", "ايباد"),
    "phone": ("phone", "smartphone", "mobile", "iphone", "ايفون", "موبايل", "تليفون", "هاتف"),
    "laptop": ("laptop", "notebook", "macbook", "لابتوب", "لاب توب"),
    "earbuds": ("earbuds", "earphones", "headphones", "headset", "earphone", "headphone", "سماعه", "سماعات"),
    "speaker": ("speaker", "speakers", "soundbar", "سبيكر", "مكبر صوت", "مكبرات صوت"),
    "watch": ("smart watch", "smartwatch", "watch", "ساعه"),
    "freezer": ("freezer", "ديب", "فريزر"),
    "fan": ("fan", "مروحه"),
    "printer": ("printer", "طابعه"),
    "router": ("router", "راوتر"),
    "blender": ("blender", "خلاط"),
    "iron": ("iron", "مكواه"),
    "camera": ("camera", "كاميرا"),
}
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


def _folded(text: str) -> str:
    return (text or "").lower().translate(_FOLD)


def _says(title: str, word: str) -> bool:
    """The title has this word or phrase as a whole word (never inside another: "lg" is not in "50LG8070Q", "tv" not
    in "ctv"). Arabic words may carry ال before them."""
    w = re.escape(_folded(word))
    return bool(re.search(rf"(?<![a-z0-9؀-ۿ])(?:ال)?{w}(?![a-z0-9؀-ۿ])", _folded(title)))


def kinds_in(text: str) -> set[str]:
    """The kinds of products a text names (by KIND_WORDS)."""
    return {k for k, words in KIND_WORDS.items() if any(_says(text, w) for w in words)}


def same_kind(title: str, wanted: set[str]) -> bool:
    """With no model to judge: an offer is not another kind of product than the one the shopper named. A title that
    names no kind at all ("Samsung Galaxy A36 256GB") is kept; one that names only other kinds is not."""
    if not wanted:
        return True
    named = kinds_in(title)
    # a fridge's "top freezer" and a phone's "watch face" don't make it another product while it names the one asked
    return not named or bool(named & wanted)


def names_brand(title: str, brand: str) -> bool:
    """The title names this brand as a whole word, in English or in the Arabic shoppers write."""
    return _says(title, brand) or any(_says(title, a) for a, e in AR_WORDS.items() if e == brand)


# Features a title can say it does NOT have: with no model to judge, an offer that says it lacks what the shopper
# asked for is left out (a "cool only" air conditioner for "بارد ساخن", a non-inverter one for "انفرتر").
LACKS = {
    "heat": r"cool(?:ing)?\s*only|بارد\s*فقط|تبريد\s*فقط|بارد\s*بس",
    "inverter": r"non[\s-]*inverter|بدون\s*انفرتر",
    "no frost": r"\bdefrost\b|ديفروست",
}


def lacks_wanted(title: str, eng: str) -> bool:
    """The title says it lacks a feature the request (english_words output) asks for."""
    t = _folded(title)
    return any(re.search(pat, t) for feat, pat in LACKS.items() if re.search(rf"(?<![a-z]){feat}(?![a-z])", eng))


# Words that make a title an accessory or a part for the product it names ("Laptop Stand", "Cleaning Kit for
# AirPods", "حقيبة لابتوب"): with no model to judge, never shown for a search of the product itself. Not when the
# word follows "with"/"مع" (a laptop "with backpack" is still the laptop) or the shopper asked for an accessory.
ACCESSORY_WORDS = {
    "case", "cases", "cover", "covers", "protector", "tempered", "film", "skin", "skins", "holder", "mount",
    "replacement", "compatible", "sticker", "stickers", "pouch", "sleeve", "spare", "bracket", "tpu", "bumper",
    "shockproof", "stand", "bag", "backpack", "cleaning", "cleaner", "kit",
    "adapter", "charger", "cable", "dock", "hub", "decal", "lanyard", "organizer", "keycaps", "guard",
    "جراب", "جرابات", "كفر", "غطاء", "اسكرينه", "لاصقه", "حامل", "ستاند", "شنطه", "حقيبه", "منظف", "تنظيف",
    "شاحن", "كابل", "وصله", "بديل", "متوافق", "استيكر",
}
ACCESSORY_PHRASES = ("cooling pad", "mouse pad", "laptop pad", "قاعده تبريد")
_WITH = {"with", "w/", "+", "مع", "including", "includes"}
_FREE_PHRASES = ("vacuum cleaner", "robot cleaner", "charging case", "wireless charging", "fast charging", "charging", "cleaning mode", "self cleaning",
                 "self-cleaning", "تنظيف ذاتي")


def wants_accessory(eng: str) -> bool:
    """The request (english_words output) names an accessory: then accessories are what they want."""
    return bool(set(eng.lower().split()) & ACCESSORY_WORDS) or any(p in eng.lower() for p in ACCESSORY_PHRASES)


def is_accessory(title: str) -> bool:
    """The title is an accessory or a part (by ACCESSORY_WORDS), not a product sold with one."""
    t = _folded(title)
    for phrase in _FREE_PHRASES:
        t = t.replace(phrase, " ")
    if any(p in t for p in ACCESSORY_PHRASES):
        return True
    words = [w for w in re.split(r"[^a-z0-9؀-ۿ/+&]+", t) if w]
    for i, w in enumerate(words):
        bare = w[2:] if w.startswith("ال") and w[2:] in ACCESSORY_WORDS else w
        bare = bare[1:] if bare.startswith("و") and bare[1:] in ACCESSORY_WORDS else bare
        if bare in ("compatible", "متوافق", "replacement", "بديل"):
            return True  # "... compatible with iPhone 15": made for the product, never the product
        if bare in ACCESSORY_WORDS and not (set(words[max(0, i - 4):i]) & _WITH or w.startswith("و") and bare != w):
            return True
    return False


# Features a shopper names that a title must say (in any of these words) when no model can judge: a "gaming laptop"
# request never shows a Celeron office laptop, an "inverter" one never a plain AC. Folded spelling (ة->ه, أ->ا).
FEATURES = {
    "gaming": r"gaming|gamer|game|جيمنج|جيمينج|العاب|للالعاب|rtx|gtx|radeon rx",
    "inverter": r"inverter|انفرتر|انفرتير",
    "no frost": r"no\s*-?\s*frost|nofrost|نو\s*فروست|نوفروست",
    "automatic": r"automatic|auto|اوتوماتيك|اتوماتيك|اوتوماتك",
    "wireless": r"wireless|bluetooth|tws|لاسلكي|بلوتوث|وايرلس",
    "bluetooth": r"bluetooth|wireless|tws|بلوتوث|لاسلكي|وايرلس",
    "smart": r"smart|سمارت|ذكي|android|google tv|webos|tizen|vidaa",
}


def misses_feature(title: str, eng: str) -> bool:
    """The request (english_words output) names a feature (FEATURES) the title never says."""
    t = _folded(title)
    words = set(eng.lower().split())
    return any(feat in words or (" " in feat and feat in eng.lower())
               for feat in FEATURES if not re.search(FEATURES[feat], t))


# Words that make a phone or a device another model of the same number: a "Realme 12 Pro" request never shows the
# "12 Pro+" or the plain "12", an "iPhone 16 Pro Max" request never the "16 Pro".
VARIANTS = {"pro", "plus", "max", "ultra", "lite", "mini", "fe", "neo", "prime", "edge", "air"}


def _variants_after(words: list[str], i: int) -> frozenset:
    got = set()
    for w in words[i + 1:i + 4]:
        if w in VARIANTS:
            got.add(w)
        elif w not in ("5g", "4g"):
            break
    return frozenset(got)


def other_variant(title: str, eng: str) -> bool:
    """The request names a model number ("12" in "realme 12 pro", not a size like "55 inch") and the title never
    names that model: the number isn't in it (a "C85 Pro"), or every time it is it's another variant of it (Pro+ for
    Pro, Pro for Pro Max, Pro for the plain one)."""
    req = eng.lower().split()
    t = re.sub(r"\+", " plus ", _folded(title))
    t_words = [w for w in re.split(r"[^a-z0-9.]+", t) if w]
    for i, w in enumerate(req):
        if not w.isdigit() or (i + 1 < len(req) and req[i + 1] in _UNITS) or len(w) >= 4:
            continue
        want = _variants_after(req, i)
        seen = [_variants_after(t_words, j) for j, x in enumerate(t_words) if x == w]
        if want not in seen:
            return True
    return False


_FOR = {"for", "fits", "fit", "compatible", "لـ", "متوافق"}


def made_for(title: str, eng: str) -> bool:
    """The title names the requested brand or product line only after "for" ("Power Bank for iPhone 15"): it's made
    for that product, not that product."""
    words = [w for w in re.split(r"[^a-z0-9\u0600-\u06ff]+", _folded(title)) if w]
    brands = [b for b in eng.lower().split() if b in BRANDS]
    if not brands:
        return False
    at = [i for i, w in enumerate(words) if w in brands]
    return bool(at) and all(set(words[max(0, i - 3):i]) & _FOR for i in at)
