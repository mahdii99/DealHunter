"""Turn a free-text request (Arabic or English) into a structured query.

Rule-based by default (free, no API). If GEMINI_API_KEY is set, app.llm
can be used first and this module is the fallback.
"""
import re
import unicodedata
from functools import lru_cache

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
UNIT_TOKEN = re.compile(r"^\d+(?:\.\d+)?(mah|w|gb|tb|mp|hz|inch|ml)$")
_BATTERY = re.compile(r"power\s*bank|powerbank|باور|بطاري|شاحن|امبير|أمبير|mah", re.I)


def battery_request(text: str) -> bool:
    """The request is for a battery or a power bank: its "مللي" is mAh, not millilitres."""
    return bool(_BATTERY.search(text or ""))


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
    # "مللي" is mAh for a battery ("باور بانك 10 الاف مللي") and millilitres for anything else ("برفان 100 مللي")
    unit = "mah" if battery_request(text) else "ml"
    text = re.sub(r"(\d+)\s*(?:الاف|آلاف|ألف|الف)\s*(?:مللي|ملي|مل)", lambda m: f" {int(m.group(1)) * 1000}{unit} ", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:مللي|ملي|ملل|مل)(?![\u0600-\u06ff])(?:\s*(?:امبير|أمبير))?", rf" \1{unit} ", text)
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
    # beauty and personal care: brands
    "لانكوم": "lancome", "لوريال": "loreal", "ميبلين": "maybelline", "مايبلين": "maybelline", "نيفيا": "nivea",
    "دوف": "dove", "بانتين": "pantene", "جارنييه": "garnier", "جارنيه": "garnier", "سيرافي": "cerave",
    "سيرافيه": "cerave", "فيشي": "vichy", "بيوديرما": "bioderma", "ايفا": "eva", "إيفا": "eva", "ستارفيل": "starville",
    "نيوتروجينا": "neutrogena", "اسنس": "essence", "إسنس": "essence", "بورجوا": "bourjois", "ريميل": "rimmel",
    "فلورمار": "flormar", "نارس": "nars", "شانيل": "chanel", "ديور": "dior", "فيرساتشي": "versace",
    "فرزاتشي": "versace", "ارماني": "armani", "أرماني": "armani", "جوتشي": "gucci", "غوتشي": "gucci",
    "بربري": "burberry", "زارا": "zara", "بالموليف": "palmolive", "صانسيلك": "sunsilk", "سانسيلك": "sunsilk",
    "تريزيمي": "tresemme", "فاتيكا": "vatika", "ريمنجتون": "remington", "بيبيليس": "babyliss",
    "بابليس": "babyliss", "سيتافيل": "cetaphil", "اوليه": "olay", "أوليه": "olay", "جيليت": "gillette",
    "ريكسونا": "rexona", "لاتافا": "lattafa", "لطافة": "lattafa", "كيكو": "kiko", "كاتريس": "catrice",
    "ريفلون": "revlon", "كلينيك": "clinique", "ايسترلودر": "esteelauder", "اوردينري": "ordinary",
    "إيفرز": "evers", "شيجلام": "sheglam", "انفيوزد": "infused", "يوسيرين": "eucerin",
    "ايسدين": "isdin", "سباستيان": "sebastian", "كيراستاس": "kerastase", "شوارزكوف": "schwarzkopf", "جليس": "gliss",
    "كامينا": "kamena", "هيمالايا": "himalaya", "فازلين": "vaseline", "جونسون": "johnsons",
    # beauty and personal care: kinds of products
    "برفان": "perfume", "بارفان": "perfume", "برفيوم": "perfume", "بيرفيوم": "perfume", "عطر": "perfume",
    "عطور": "perfume", "سبلاش": "splash", "ديودرانت": "deodorant", "شامبو": "shampoo", "بلسم": "conditioner",
    "كونديشنر": "conditioner", "سيروم": "serum", "سيرم": "serum", "كريم": "cream", "مرطب": "moisturizer",
    "لوشن": "lotion", "غسول": "cleanser", "تونر": "toner", "روج": "lipstick", "ماسكرا": "mascara",
    "ماسكارا": "mascara", "مسكرة": "mascara", "ايلاينر": "eyeliner", "ايلانر": "eyeliner", "كحل": "eyeliner",
    "فاونديشن": "foundation", "كونسيلر": "concealer", "بودرة": "powder", "بودره": "powder", "بلاشر": "blush",
    "بلشر": "blush", "هايلايتر": "highlighter", "ايشادو": "eyeshadow", "باليت": "palette", "جلوس": "gloss",
    "مانيكير": "nail polish", "مناكير": "nail polish", "استشوار": "hair dryer", "سشوار": "hair dryer",
    "سيشوار": "hair dryer", "اسشوار": "hair dryer", "ابيلاتور": "epilator", "تريمر": "trimmer", "صابون": "soap",
    "صابونة": "soap", "ماسك": "mask", "صبغة": "hair color", "صبغه": "hair color", "ميكب": "makeup",
    "مكياج": "makeup", "ميكاب": "makeup", "سكرب": "scrub",
    "مقشر": "scrub", "زيت": "oil", "جل": "gel", "رموش": "lashes", "فرشاة": "brush", "فرشاه": "brush",
    "مزيل": "remover", "تستر": "tester", "مات": "matte", "هيالورونيك": "hyaluronic", "نياسيناميد": "niacinamide",
    "ريتينول": "retinol", "كولاجين": "collagen", "كيراتين": "keratin", "ارجان": "argan", "أرجان": "argan",
    "بشرة": "skin", "بشره": "skin", "للبشرة": "skin", "للبشره": "skin", "شعر": "hair", "للشعر": "hair",
    "حريمي": "women", "نسائي": "women", "رجالي": "men", "للرجال": "men", "للستات": "women", "للبنات": "women",
    # beauty units: millilitres and grams
    "مل": "ml", "ملي": "ml", "مللي": "ml", "ملل": "ml", "ميلي": "ml", "جرام": "gram", "جم": "gram", "غرام": "gram",
}
# Brands and words shoppers write as several Arabic words ("جي بي ال" = JBL): made one word before anything else.
AR_PHRASES = {
    "جي بي ال": "jbl", "جى بى ال": "jbl", "جي بي إل": "jbl", "ال جي": "lg", "ال جى": "lg", "إل جي": "lg",
    "اتش بي": "hp", "اتش بى": "hp", "إتش بي": "hp", "تي سي ال": "tcl", "تى سى ال": "tcl", "ام اس اي": "msi",
    "إم إس آي": "msi", "ون بلس": "oneplus", "وان بلس": "oneplus", "لاب توب": "لابتوب", "نو فروست": "نوفروست",
    "مكبر صوت": "speaker", "هيد فون": "هيدفون", "سمارت واتش": "smartwatch", "ساعة ذكية": "smartwatch",
    "ساعه ذكيه": "smartwatch",
    # beauty and personal care
    "مكواة شعر": "hair straightener", "مكواه شعر": "hair straightener", "مكواة الشعر": "hair straightener",
    "مكواه الشعر": "hair straightener", "مكوة شعر": "hair straightener", "فير شعر": "hair straightener",
    "مجفف شعر": "hair dryer", "مجفف الشعر": "hair dryer", "ماكينة حلاقة": "shaver", "ماكينه حلاقه": "shaver",
    "مكنة حلاقة": "shaver", "ماكينة ازالة الشعر": "epilator", "ماكينه ازاله الشعر": "epilator",
    "جهاز ازالة الشعر": "epilator", "جهاز ازاله الشعر": "epilator", "احمر شفاه": "lipstick", "أحمر شفاه": "lipstick",
    "احمر شفايف": "lipstick", "أحمر شفايف": "lipstick", "كريم اساس": "foundation", "كريم أساس": "foundation",
    "واقي شمس": "sunscreen", "واقى شمس": "sunscreen", "صن بلوك": "sunscreen", "صن سكرين": "sunscreen",
    "مزيل عرق": "deodorant", "مزيل العرق": "deodorant", "رول اون": "deodorant", "بودي سبلاش": "splash",
    "بودي ميست": "splash", "بودى سبلاش": "splash", "طلاء اظافر": "nail polish", "طلاء أظافر": "nail polish",
    "غسول وجه": "cleanser", "غسول الوجه": "cleanser", "شاور جل": "shower gel", "جل استحمام": "shower gel",
    "مرطب شفايف": "lip balm", "مرطب شفاه": "lip balm", "ليب بالم": "lip balm", "ليب جلوس": "gloss",
    "مزيل مكياج": "makeup remover", "مزيل ميكب": "makeup remover", "ميسيلار": "micellar", "ميك اب": "makeup",
    "صبغة شعر": "hair color", "صبغه شعر": "hair color", "زيت شعر": "hair oil", "ماسك شعر": "hair mask",
    "هيد اند شولدرز": "head-shoulders", "هيد آند شولدرز": "head-shoulders", "هيد اند شولدر": "head-shoulders",
    "لاروش بوزيه": "la-roche-posay", "لا روش بوزيه": "la-roche-posay", "ذا اوردينري": "ordinary",
    "ذا أوردينري": "ordinary", "هدى بيوتي": "huda-beauty", "هدي بيوتي": "huda-beauty",
    "كالفن كلاين": "calvin-klein", "كارولينا هيريرا": "carolina-herrera", "كارولينا هيرera": "carolina-herrera",
    "ايف سان لوران": "ysl", "إيف سان لوران": "ysl", "سان لوران": "ysl", "هوجو بوس": "hugo-boss",
    "فيكتوريا سيكريت": "victorias-secret", "فكتوريا سيكريت": "victorias-secret", "ماري كي": "mary-kay",
    "استي لودر": "esteelauder", "إستي لودر": "esteelauder", "جونسون اند جونسون": "johnsons", "لا في ايبل": "la vie est belle",
    "فيتامين سي": "vitamin c", "فيتامين c": "vitamin c", "فيتامين اي": "vitamin e", "فيتامين إي": "vitamin e",
    "للبشرة الدهنية": "oily skin", "للبشره الدهنيه": "oily skin", "بشرة دهنية": "oily skin", "البشرة الدهنية": "oily skin",
    "للبشرة الجافة": "dry skin", "للبشره الجافه": "dry skin", "بشرة جافة": "dry skin", "البشرة الجافة": "dry skin",
    "للبشرة الحساسة": "sensitive skin", "للبشره الحساسه": "sensitive skin", "بشرة حساسة": "sensitive skin",
    "للبشرة المختلطة": "combination skin", "للشعر الجاف": "dry hair", "للشعر الدهني": "oily hair",
    "للشعر المجعد": "curly hair", "للشعر الكيرلي": "curly hair", "ضد القشرة": "anti dandruff", "ضد القشره": "anti dandruff",
    "لافي ايبل": "la vie est belle", "لا في إيبل": "la vie est belle", "سوبر ستاي": "superstay", "فيت مي": "fit me",
}
_PHRASE_RE = re.compile("|".join(rf"(?<![\u0600-\u06ff]){re.escape(k)}(?![\u0600-\u06ff])"
                                 for k in sorted(AR_PHRASES, key=len, reverse=True)))


def phrased(text: str) -> str:
    """The shopper's text with multi-word names made one word ("سماعة جي بي ال" -> "سماعة jbl")."""
    return _PHRASE_RE.sub(lambda m: AR_PHRASES[m.group(0)], text or "")
_UNITS = {"inch", "kg", "gb", "tb", "hp", "liter", "ml", "gram"}
BRANDS = {"samsung", "xiaomi", "realme", "oppo", "apple", "iphone", "huawei", "infinix", "tecno", "honor", "nokia",
          "vivo", "motorola", "lenovo", "dell", "acer", "asus", "toshiba", "sharp", "tornado", "fresh", "zanussi",
          "beko", "ariston", "unionaire", "carrier", "gree", "hisense", "sony", "panasonic", "philips", "braun",
          "anker", "soundcore", "bosch", "kenwood", "tefal", "hitachi", "midea", "lg", "hp", "tcl", "jbl", "oneplus",
          "redmi", "poco", "msi", "haier", "white", "whirlpool", "ikon", "kiriazi", "olympic", "jac",
          # beauty and personal care
          "lancome", "loreal", "maybelline", "nivea", "dove", "pantene", "garnier", "cerave", "vichy", "bioderma",
          "eva", "starville", "neutrogena", "essence", "bourjois", "rimmel", "flormar", "nars", "chanel", "dior",
          "versace", "armani", "gucci", "burberry", "zara", "palmolive", "sunsilk", "tresemme", "vatika", "remington",
          "babyliss", "cetaphil", "olay", "gillette", "rexona", "lattafa", "kiko", "catrice", "revlon", "clinique",
          "esteelauder", "ordinary", "sheglam", "eucerin", "isdin", "kerastase", "schwarzkopf", "gliss", "kamena",
          "himalaya", "vaseline", "johnsons", "head-shoulders", "la-roche-posay", "huda-beauty", "calvin-klein",
          "carolina-herrera", "ysl", "hugo-boss", "victorias-secret", "mary-kay"} - {"white"}
# How store titles write the brands whose name isn't one plain word ("L'Oréal Paris", "Head & Shoulders"): a title
# names the brand when it has any of these (matched as whole words, accents and apostrophes ignored).
BRAND_ALIASES = {
    "loreal": ("loreal", "l'oreal", "l oreal"), "head-shoulders": ("head & shoulders", "head and shoulders",
    "head&shoulders", "head shoulders"), "la-roche-posay": ("la roche-posay", "la roche posay", "laroche posay"),
    "huda-beauty": ("huda beauty",), "calvin-klein": ("calvin klein",), "carolina-herrera": ("carolina herrera",),
    "ysl": ("ysl", "yves saint laurent", "saint laurent"), "hugo-boss": ("hugo boss", "boss"),
    "victorias-secret": ("victoria's secret", "victorias secret", "victoria secret"), "mary-kay": ("mary kay",),
    "esteelauder": ("estee lauder", "esteelauder"), "ordinary": ("the ordinary",), "johnsons": ("johnson's",
    "johnsons", "johnson"), "tresemme": ("tresemme",), "lancome": ("lancome",), "kerastase": ("kerastase",),
}
_AR_UNITS = {"inch": "بوص|بوصه|بوصة|انش", "kg": "كيلو|كجم|كغ", "gb": "جيجا|جيجابايت", "tb": "تيرا", "hp": "حصان",
             "liter": "لتر", "ml": "مل|ملي|مللي|ملل|milliliters?|millilitres?|ml",
             "gram": "grams?|gms?|gr|g|جرام|جم|غرام"}
AR_UNITS = _AR_UNITS  # (the Arabic words of each unit, for matching sizes elsewhere)


def wanted_sizes(eng: str) -> list[tuple[str, str]]:
    """The sizes the request names in english_words' output ("55 inch", "10 kg"): (number, unit)."""
    return re.findall(r"(\d+(?:\.\d+)?) (" + "|".join(sorted(_UNITS)) + r")\b", eng)


def shows_size(title: str, number: str, unit: str) -> bool:
    """The title names this size, in English or Arabic ("55 Inch", "55-inch", '55"', "55 بوصة", "10 كجم")."""
    t = normalize(title)
    ar = _AR_UNITS.get(unit, unit)
    if unit in ("ml", "gram"):  # "100 مل" is not "100 ملم" (mm), "50 g" is not "50 gb"
        return bool(re.search(rf"(?<![\d.]){re.escape(number)}\s*(?:-\s*)?(?:{ar})(?![a-z\u0600-\u06ff])", t))
    return bool(re.search(rf"(?<![\d.]){re.escape(number)}\s*(?:-\s*)?(?:{unit}|{ar}|\"|''|”)", t))


def english_words(text: str) -> str:
    """The shopper's request in the English words stores use in their titles, with no model: each Arabic word we
    know replaced by its English word, Latin words and numbers kept, numbers joined to their unit ("55 بوصة" ->
    "55inch"); Arabic words we don't know are left out. '' when nothing is left."""
    out: list[str] = []
    battery = battery_request(text)
    for w in normalize(phrased(text)).split():
        w = w.strip(".,!?؟،()")
        if m := re.fullmatch(r"(\d+(?:\.\d+)?)(ml|kg|gb|tb|inch)", w):  # "100ml" typed as one word: "100 ml"
            if m.group(1) not in out:
                out.append(m.group(1))
            w = m.group(2)
        plus = len(w) > 1 and w.endswith("+")  # "برو+" / "pro+": the plus model
        w = w.rstrip("+") if plus else w
        w = w[2:] if w.startswith("ال") and w[2:] in AR_WORDS else w
        w = w[1:] if w.startswith("و") and w not in AR_WORDS and w[1:] in AR_WORDS else w  # "بارد وساخن": "and hot"
        eng = AR_WORDS.get(w) or (w if re.fullmatch(r"[a-z0-9][a-z0-9+.\-]*", w) else "")
        if battery and eng == "ml":  # "10000 مللي" for a power bank is mAh: not a size we check
            continue
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
    "air conditioner": ("air conditioner", "air conditioners", "split", "تكييف", "مكيف"),
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
    # beauty and personal care
    "perfume": ("perfume", "perfumes", "parfum", "eau de parfum", "eau de toilette", "edp", "edt", "fragrance",
                "cologne", "برفان", "بارفان", "برفيوم", "بيرفيوم", "عطر", "عطور"),
    "body splash": ("splash", "body splash", "body mist", "fragrance mist", "سبلاش", "بودي سبلاش", "بودي ميست"),
    "deodorant": ("deodorant", "deodorants", "antiperspirant", "deo", "roll on", "roll-on", "مزيل عرق", "ديودرانت"),
    "shampoo": ("shampoo", "شامبو"),
    "hair conditioner": ("conditioner", "hair conditioner", "بلسم", "كونديشنر"),
    "serum": ("serum", "سيروم", "سيرم"),
    "sunscreen": ("sunscreen", "sunblock", "sun block", "sun screen", "sun cream", "واقي شمس", "صن بلوك"),
    "cream": ("cream", "moisturizer", "moisturiser", "lotion", "body butter", "كريم", "مرطب", "لوشن"),
    "cleanser": ("cleanser", "face wash", "facial wash", "cleansing gel", "cleansing foam", "micellar", "غسول"),
    "toner": ("toner", "تونر"),
    "lipstick": ("lipstick", "lipsticks", "lip color", "lip colour", "lip tint", "lip crayon", "روج", "احمر شفاه"),
    "lip gloss": ("gloss", "lip gloss", "جلوس"),
    "lip balm": ("lip balm", "ليب بالم", "مرطب شفاه", "مرطب شفايف"),
    "mascara": ("mascara", "ماسكرا", "ماسكارا", "مسكره"),
    "eyeliner": ("eyeliner", "eye liner", "kohl", "kajal", "ايلاينر", "كحل"),
    "foundation": ("foundation", "bb cream", "cc cream", "فاونديشن", "كريم اساس"),
    "concealer": ("concealer", "كونسيلر"),
    "face powder": ("face powder", "compact powder", "setting powder", "loose powder", "pressed powder", "بودره"),
    "blush": ("blush", "blusher", "بلاشر", "بلشر"),
    "eyeshadow": ("eyeshadow", "eye shadow", "ايشادو"),
    "nail polish": ("nail polish", "nail lacquer", "nail color", "مانيكير", "مناكير", "طلاء اظافر"),
    "hair dryer": ("hair dryer", "hairdryer", "blow dryer", "hair dryers", "استشوار", "سشوار", "مجفف شعر"),
    "hair straightener": ("straightener", "hair straightener", "flat iron", "hair iron", "straightening brush",
                          "مكواه شعر"),
    "hair curler": ("curler", "curling iron", "curling wand", "hair curler", "كيرلي"),
    "epilator": ("epilator", "ipl", "hair removal device", "ابيلاتور"),
    "shaver": ("shaver", "trimmer", "razor", "clipper", "clippers", "grooming kit", "تريمر", "ماكينه حلاقه"),
    "soap": ("soap", "soaps", "صابون", "صابونه"),
    "shower gel": ("shower gel", "body wash", "شاور جل"),
    "hair color": ("hair color", "hair colour", "hair dye", "صبغه"),
}
# The kinds above that are beauty and personal-care products: their titles say "kit", "cover" ("full cover"),
# "skin" or "cleansing" without being an accessory of another product.
BEAUTY_KINDS = {"perfume", "body splash", "deodorant", "shampoo", "hair conditioner", "serum", "sunscreen", "cream",
                "cleanser", "toner", "lipstick", "lip gloss", "lip balm", "mascara", "eyeliner", "foundation",
                "concealer", "face powder", "blush", "eyeshadow", "nail polish", "hair dryer", "hair straightener",
                "hair curler", "epilator", "shaver", "soap", "shower gel", "hair color"}
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


@lru_cache(maxsize=8192)
def _folded(text: str) -> str:
    """Lower case, Arabic spelling variants made one (ة->ه, أ->ا), accents and curly apostrophes plain ("L’Oréal" ->
    "l'oreal")."""
    text = unicodedata.normalize("NFKD", (text or "").replace("’", "'").replace("`", "'"))
    return "".join(c for c in text if not unicodedata.combining(c) or "\u0600" <= c <= "\u06ff").lower().translate(_FOLD)


def _says(title: str, word: str) -> bool:
    """The title has this word or phrase as a whole word (never inside another: "lg" is not in "50LG8070Q", "tv" not
    in "ctv"). Arabic words may carry ال before them."""
    return bool(_word_pattern(word).search(_folded(title)))


@lru_cache(maxsize=8192)
def _word_pattern(word: str) -> re.Pattern:
    # compiled once per word: _says runs for every brand and kind word on every title (the deals page: 300 titles
    # x thousands of words took over 30 s of CPU and froze the server, live 2026-11-28)
    return re.compile(rf"(?<![a-z0-9؀-ۿ])(?:ال)?{re.escape(_folded(word))}(?![a-z0-9؀-ۿ])")


def kinds_in(text: str) -> set[str]:
    """The kinds of products a text names (by KIND_WORDS). A word inside a longer one another kind names doesn't
    count: "conditioner" in "air conditioner" isn't hair conditioner, "iron" in "hair iron" isn't a clothes iron."""
    hits = [(k, _folded(w)) for k, words in KIND_WORDS.items() for w in words if _says(text, w)]
    return {k for k, w in hits
            if not any(k2 != k and len(w2) > len(w) and re.search(rf"(?<![a-z0-9])(?:ال)?{re.escape(w)}(?![a-z0-9])", w2)
                       for k2, w2 in hits)}


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
    return any(_says(title, w) for w in _brand_words(brand))


@lru_cache(maxsize=1024)
def _brand_words(brand: str) -> tuple[str, ...]:
    """The brand, its aliases, and the Arabic words and phrases shoppers write for it."""
    return ((brand,) + tuple(BRAND_ALIASES.get(brand, ())) + tuple(a for a, e in AR_WORDS.items() if e == brand)
            + tuple(a for a, e in AR_PHRASES.items() if e == brand))


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
    "case", "cases", "cover", "covers", "protector", "tempered", "film", "skins", "wrap", "holder", "mount",
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


# Words of ACCESSORY_WORDS that a beauty product's own title uses ("Makeup Kit", "Full Cover Concealer", "Hair
# Dryer with Stand", "Cleansing Gel", a straightener's "heat-resistant bag"): not an accessory there.
BEAUTY_FREE = {"kit", "cover", "covers", "case", "bag", "pouch", "stand", "cleaning", "cleaner", "charger", "cable",
               "adapter", "film", "guard", "sleeve", "skin"}
_BEAUTY_TITLE = r"make\s*-?\s*up|cosmetic|beauty|skin\s*care|skincare|hair\s*care|مكياج|ميكب|تجميل|العنايه"


def is_accessory(title: str) -> bool:
    """The title is an accessory or a part (by ACCESSORY_WORDS), not a product sold with one."""
    t = _folded(title)
    free = BEAUTY_FREE if kinds_in(title) & BEAUTY_KINDS or re.search(_BEAUTY_TITLE, t) else set()
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
        if bare in free:
            continue
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


# A perfume or a cosmetic sold as something else's copy ("Inspired by La Vie Est Belle", "Dupe", "عطر مستوحى من")
# or a tester (no box, often no cap): not the product the shopper named, unless they asked for one.
COPY = r"inspired|inspiration|impression of|dupe|alternative to|similar to|our version of|clone|مستوحي|مستوحى|شبيه|تقليد|كوبي|هاي كوبي"
TESTER = r"(?<![a-z])tester(?![a-z])|تستر"


def copy_or_tester(title: str, eng: str) -> str | None:
    """'copy' when the title says it's another product's copy, 'tester' when it's a tester, and the request (english_
    words output) didn't ask for that; else None."""
    t = _folded(title)
    if re.search(COPY, t) and not re.search(COPY, eng.lower()):
        return "copy"
    if re.search(TESTER, t) and "tester" not in eng.lower().split():
        return "tester"
    return None


# Who a product is for, when the shopper says it ("برفان حريمي", "مزيل عرق رجالي"): a title made only for the other
# one isn't what they want. A title that says neither (most unisex products) is kept.
GENDERS = {
    "women": r"women|woman|womens|female|ladies|lady|for her|pour femme|femme|girls?|حريمي|نسائي|للنساء|للبنات|للستات",
    "men": r"(?<![a-z])(?:men|man|mens|male|for him|pour homme|homme|boys?)(?![a-z])|رجالي|للرجال",
}


def other_gender(title: str, eng: str) -> bool:
    """The request (english_words output) names women or men and the title names only the other."""
    words = set(eng.lower().split())
    t = _folded(title)
    for want, other in (("women", "men"), ("men", "women")):
        if want in words and other not in words:
            return bool(re.search(GENDERS[other], t)) and not re.search(GENDERS[want], t) and "unisex" not in t
    return False


def kind_synonyms(word: str) -> set[str]:
    """Every word store titles use for the kind of product this request word names ("perfume" -> "eau de parfum",
    "edp", "fragrance"...): a title saying any of them has the word."""
    return {w for k in kinds_in(word) for w in KIND_WORDS[k]}


BEAUTY_BRANDS = {"lancome", "loreal", "maybelline", "nivea", "dove", "pantene", "garnier", "cerave", "vichy", "bioderma",
                 "eva", "starville", "neutrogena", "essence", "bourjois", "rimmel", "flormar", "nars", "chanel", "dior",
                 "versace", "armani", "gucci", "burberry", "palmolive", "sunsilk", "tresemme", "vatika", "babyliss",
                 "cetaphil", "olay", "rexona", "lattafa", "kiko", "catrice", "revlon", "clinique", "esteelauder",
                 "ordinary", "sheglam", "eucerin", "isdin", "kerastase", "schwarzkopf", "gliss", "kamena", "himalaya",
                 "vaseline", "johnsons", "head-shoulders", "la-roche-posay", "huda-beauty", "calvin-klein",
                 "carolina-herrera", "ysl", "hugo-boss", "victorias-secret", "mary-kay"}
_BEAUTY_REQUEST = r"makeup|cosmetic|skin|hair|serum|perfume|lipstick|mascara|cream|lotion|nail|beauty"


def is_beauty(eng: str) -> bool:
    """The request (English words) is for a beauty or personal-care product: a kind, a brand or a word of those."""
    words = set(eng.lower().split())
    return bool(kinds_in(eng) & BEAUTY_KINDS or words & BEAUTY_BRANDS or re.search(_BEAUTY_REQUEST, eng.lower()))


# A perfume's other editions sold under the same name ("La Vie Est Belle Intensément", "L'Elixir", "Iris Absolu", a
# refill bottle): another product than the one named, unless the shopper names that edition too.
FLANKERS = r"intense|intensement|elixir|absolu|absolue|extreme|prive|privee|nuit|night|noir|refill|recharge|l'eau|" \
           r"eau fraiche|sport|legere|florale|soleil|edition"


def other_edition(title: str, eng: str) -> bool:
    """The request (english_words output) is for a perfume and the title is another edition of it (FLANKERS)."""
    if "perfume" not in kinds_in(eng) and "body splash" not in kinds_in(eng):
        return False
    asked = _folded(eng)
    return any(m.group(0)[:5] not in asked for m in re.finditer(rf"(?<![a-z])(?:{FLANKERS})(?![a-z])", _folded(title)))
