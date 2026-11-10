import re

from .models import ParsedQuery, Product
from .utils import normalize

# Titles containing these are usually accessories, not the product itself.
ACCESSORY_WORDS = {"case", "cover", "pouch", "sticker", "skin", "holder", "bag", "جراب"}


# Different stores name the same thing differently; map variants to the words users search with.
SYNONYMS = {
    "battery pack": "power bank", "portable charger": "power bank", "powerbank": "power bank",
    "external battery": "power bank", "earphones": "earbuds", "headphones": "earbuds",
}


def relevance(p: Product, q: ParsedQuery) -> float:
    title = normalize(p.title)
    for variant, canonical in SYNONYMS.items():
        if variant in title:
            title += f" {canonical}"
    words = set(re.findall(r"[\w.]+", title))
    for token in q.must_have:
        if token not in title.replace(" ", ""):
            return 0.0
    if words & ACCESSORY_WORDS and not set(q.keywords) & ACCESSORY_WORDS:
        return 0.0
    if not q.keywords:
        return 1.0
    hits = sum(1 for k in q.keywords if k in words or k in title)
    return hits / len(q.keywords)


def rank(products: list[Product], q: ParsedQuery, min_score: float = 0.5, limit: int = 30) -> list[Product]:
    out = []
    for p in products:
        if q.max_price is not None and (p.coupon_price or p.price) > q.max_price:
            continue
        if q.min_price is not None and p.price < q.min_price:
            continue
        p.score = round(relevance(p, q), 2)
        if p.score >= min_score:
            out.append(p)
    out.sort(key=lambda p: (p.coupon_price or p.price, -p.score))
    return out[:limit]
