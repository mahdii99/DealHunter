"""A learned reader for a product page's main price, for stores seen for the first time (no recipe, no clear buy box).

Every element of the page that shows just one price is a candidate. Each candidate is described by plain facts that
hold on any store (no store-specific selector): how close it is to the product's heading and to the buy button, what
its class names say (price / sale / old / related / installment), whether it's crossed out or hidden, whether it sits
in a list of other products, how often the number appears on the page... A small model (logistic regression, its
weights in app/price_model.json) learned from pages whose price we know (the page's own product data, LLM reads and
clear reads) scores each candidate: the chance it's the main product's current price.

It's used only when it's sure: the best candidate's chance is at least MIN_SURE and clearly above the next different
number's. Else the page goes to the LLM, as before. Its number is still checked on the page opened just now.

Training: scripts/train_price_model.py (offline, scikit-learn); the app only needs the weights (no extra packages).
"""
import json
import math
import os
import re
from urllib.parse import urlparse

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import cards

MIN_SURE = 0.95     # the best candidate's chance needed to read the page without the LLM
MARGIN = 0.4        # ... and how far above the best candidate showing another number
MAX_CANDIDATES = 80
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "price_model.json")

_CUR = r"(?:EGP|E\.?G\.?P|L\.?E\.?|ج\.?\s?م\.?|جنيه(?:\s?مصري)?|[A-Z]{3}|[$€£]|ر\.?\s?س|د\.?\s?إ|درهم|ريال)"
_JUST_PRICE = re.compile(r"^\s*(?:" + _CUR + r")?\s*(" + cards._NUMBER + r")\s*(?:" + _CUR + r")?\s*$")
_HAS_CUR = re.compile(_CUR)
_SPLIT = re.compile(r"(\d)\s*([.,])\s*(\d)")
_BUY = re.compile(r"add.to.(?:cart|bag|basket)|buy.now|[أا]ض[فـ]+\s*(?:[اإ]لى|لل)|[أا]ضف للسل|اشتر|شراء|اطلب الآن", re.I)
_WORDS = {
    "price": r"price|amount|money|cost|سعر",
    "current": r"sale|special|final|current|now|offer|actual|selling|deal|discount(?:ed)?-price|to-pay|pay",
    "old": r"old|was|regular|compare|strike|crossed|list-?price|before|original|retail|msrp|rrp",
    "related": r"related|recent|similar|upsell|cross|recommend|also|slider|carousel|swiper|widget|you-may",
    "card": r"card|item|tile|grid|product-?list|listing|thumb|mini",
    "monthly": r"install|month|emi|valu|tabby|tamara|souhoola|per-",
    "code": r"sku|code|model|barcode|ean|isbn|part|serial|phone|tel|whatsapp|count|qty|quantity|rating|review|stock",
    "main": r"main|primary|summary|detail|info|product-?(?:page|single|view)|pdp|buy-?box|purchase",
}
_WORD_RES = {k: re.compile(v, re.I) for k, v in _WORDS.items()}
_SKIP = ("script", "style", "noscript", "svg", "template", "option", "head")

_weights = None


def _load_weights():
    global _weights
    if _weights is None:
        try:
            with open(WEIGHTS_FILE, encoding="utf-8") as f:
                _weights = json.load(f)
        except (OSError, ValueError):
            _weights = {}
    return _weights


def _text(node) -> str:
    return " ".join((node.text(deep=True, separator=" ") or "").split())


def _value(text: str) -> float | None:
    m = _JUST_PRICE.match(_SPLIT.sub(r"\1\2\3", text.translate(cards._ARABIC_NUMS)))
    return cards._num(m.group(1)) if m else None


def _attrs_chain(node, up: int = 4) -> tuple[str, str]:
    """(its own class/id/itemprop words, its parents' up to `up` levels)."""
    def words(n):
        a = n.attributes or {}
        return " ".join(str(a.get(k) or "") for k in ("class", "id", "itemprop", "data-testid", "data-qa"))
    own, up_words, n = words(node), [], node.parent
    for _ in range(up):
        if n is None or n.tag in ("body", "html"):
            break
        up_words.append(words(n))
        n = n.parent
    return own, " ".join(up_words)


def _ancestors(node) -> list:
    out, n = [], node
    while n is not None:
        out.append(n)
        n = n.parent
    return out


def _hops(a_chain: list, b_chain: list) -> int:
    """Steps between two elements through their nearest common parent."""
    ids = {n.mem_id: i for i, n in enumerate(b_chain)}
    for i, n in enumerate(a_chain):
        if n.mem_id in ids:
            return i + ids[n.mem_id]
    return 60


def _hidden(chain) -> bool:
    for n in chain:
        a = n.attributes or {}
        style = (a.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style or "hidden" in a \
                or any(c.endswith("-hidden") for c in (a.get("class") or "").split()):
            return True
    return False


MAX_HTML = 3_000_000  # a bigger page isn't a store's product page (and would take too long)


def _many_children(node, most: int = 12) -> bool:
    """An element holding many others (a section, a table) shows more than one price: not worth reading its text."""
    for i, _ in enumerate(node.iter(include_text=False)):
        if i >= most:
            return True
    return False


def _price_elements(tree):
    """(element, number) for each element showing just one price: from every bit of text with a digit, the element
    around it that shows the whole price ("45.72", not its parts "45" and "72"), the innermost one showing it."""
    for t in tree.body.traverse(include_text=True):
        raw = t.text_content or ""
        if t.tag != "-text" or len(raw) > 200 or not any(c.isdigit() for c in raw):
            continue
        best, n = None, t.parent
        for _ in range(4):  # (a price is drawn in a few nested elements at most)
            if n is None or n.tag in ("body", "html") or n.tag in _SKIP or _many_children(n):
                break
            text = _text(n)
            if len(text) > 40:
                break
            v = _value(text)
            if v is not None and (best is None or v != best[1]):
                best = (n, v)  # a bigger whole replaces its part; the same number higher up keeps the innermost
            n = n.parent
        if best and best[1] >= cards.MIN_PRICE:
            yield best


def candidates(html: str, url: str, with_nodes: bool = False) -> list[tuple]:
    """(the number, its facts) for every element showing just one price, in page order."""
    if "<" not in html[:2000] or len(html) > MAX_HTML:
        return []
    tree = HTMLParser(html)
    tree.strip_tags(list(_SKIP))
    if tree.body is None:
        return []
    site = urlparse(url).netloc.lower().removeprefix("www.")
    h1 = tree.css_first("h1")
    h1_chain = _ancestors(h1) if h1 is not None else []
    buys = [b for b in tree.css("button, input[type=submit], a, [role=button]")
            if (lab := (_text(b) or (b.attributes or {}).get("value") or "")) and len(lab) <= 40 and _BUY.search(lab)]
    buy_chains = [_ancestors(b) for b in buys[:6]]
    found, order, seen = [], 0, set()
    for node, v in _price_elements(tree):
        if node.mem_id in seen:
            continue
        seen.add(node.mem_id)
        text = _text(node)
        own, up = _attrs_chain(node)
        has_cur = bool(_HAS_CUR.search(text))
        # a bare number is a price only where the page names it one ("5" stars, a phone number, a count aren't)
        if (not has_cur and not (_WORD_RES["price"].search(own) or _WORD_RES["price"].search(up))) or v > 5_000_000 \
                or (v < 20 and not has_cur):
            continue
        order += 1
        chain = _ancestors(node)
        f = {"order": math.log1p(order), "first": 1.0 if order == 1 else 0.0,
             "cur": 1.0 if has_cur else 0.0,
             "in_h1": 1.0 if h1 is not None and any(n.mem_id == h1.mem_id for n in chain) else 0.0,
             "h1_hops": math.log1p(_hops(chain, h1_chain)) if h1_chain else math.log1p(60),
             "buy_hops": math.log1p(min((_hops(chain, b) for b in buy_chains), default=60)),
             "struck": 1.0 if any(n.tag in ("del", "s", "strike") for n in chain[:4]) else 0.0,
             "hidden": 1.0 if _hidden(chain) else 0.0,
             "itemprop": 1.0 if (node.attributes or {}).get("itemprop") == "price" else 0.0,
             "layout": 1.0 if any(n.tag in ("nav", "header", "footer", "aside") for n in chain) else 0.0}
        for k, rx in _WORD_RES.items():
            f["own_" + k] = 1.0 if rx.search(own) else 0.0
            f["up_" + k] = 1.0 if rx.search(up) else 0.0
        found.append((v, f, chain))
        if len(found) >= MAX_CANDIDATES:
            break
    if not found:
        return []
    values = [v for v, _, _ in found]
    top = max(values)
    out = []
    for v, f, chain in found:
        f["times"] = math.log1p(values.count(v))
        f["rel_max"] = v / top
        f["is_max"] = 1.0 if v == top else 0.0
        # a list of other products around it (related items): links to other pages of the store's shape
        links = set()
        for n in chain[1:7]:
            for a in n.css("a[href]")[:80]:
                if (link := cards._product_link(a.attributes.get("href"), url, site)) and link.rstrip("/") != url.rstrip("/"):
                    links.add(cards._key(link))
            if len(links) > 3:
                break
        f["others"] = math.log1p(len(links))
        # another price right beside it (an old and a new price side by side): which one of the pair it is
        near = [w for w2, _, ch in found if (w := w2) != v and _hops(chain, ch) <= 4]
        f["near_lower"] = 1.0 if any(w < v for w in near) else 0.0
        f["near_higher"] = 1.0 if any(w > v for w in near) else 0.0
        out.append((v, f, chain) if with_nodes else (v, f))
    return out


def score(f: dict, w: dict) -> float:
    z = w.get("bias", 0.0) + sum(w.get(k, 0.0) * x for k, x in f.items())
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


STOCK_UP = 7  # levels above the chosen price searched for its buy button / sold-out mark (app/buybox.py's test)


def read(html: str, url: str) -> tuple[float, float, bool | None] | None:
    """(the main price, how sure, can it be bought: the buy button or sold-out mark around that price, None when
    neither or both), when the model is sure; None: the LLM reads the page."""
    from . import buybox
    w = _load_weights()
    if not w:
        return None
    cands = candidates(html, url, with_nodes=True)
    if not cands:
        return None
    scored = sorted(((score(f, w), v, chain) for v, f, chain in cands), key=lambda x: -x[0])
    best_p, best_v, chain = scored[0]
    other = max((p for p, v, _ in scored[1:] if abs(v - best_v) >= 1), default=0.0)
    if best_p < MIN_SURE or best_p - other < MARGIN:
        return None
    site = urlparse(url).netloc.lower().removeprefix("www.")
    stock = None
    for n in chain[1:STOCK_UP + 1]:  # the nearest part of the page around the price saying it (not related items')
        if n.tag in ("body", "html") or buybox._other_products(n, site, cards._key(url), url) > buybox.MAX_OTHER_PRODUCTS:
            break
        if (stock := buybox._stock(n)) is not None:
            break
    return best_v, best_p, stock


def remember(url: str, html: str, price: float, how: str) -> None:
    """A page whose main price is known becomes a training example (when that price is one of its candidates)."""
    from . import db, trust
    if "<" not in html[:2000]:
        return
    try:
        cands = candidates(html, url)
        if not any(abs(v - price) < 1 for v, _ in cands):
            return
        db.save_price_example(url[:500], trust.domain(url), float(price), how,
                              json.dumps([[v, {k: round(x, 3) for k, x in f.items()}] for v, f in cands]))
    except Exception:  # (collecting examples never breaks a search)
        return
