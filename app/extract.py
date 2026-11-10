"""Store-agnostic product extraction.

Modern store pages (Next.js, Nuxt, Magento PWA...) ship their product data as JSON inside the HTML.
Instead of writing CSS selectors per store, we decode every JSON blob in the page and pick out
objects that look like products (a name + a price). The LLM then judges which ones match.
"""
import json
import re
from html import unescape
from urllib.parse import urljoin, urlparse

from . import market
from .models import Product
from .utils import parse_price

_DECODER = json.JSONDecoder()
NAME_KEYS = ("name", "title", "product_name", "displayName")
PRICE_KEYS = ("final_without_coupon", "final_price", "special_price", "sale_price", "price", "lowPrice", "current_price", "amount")
OLD_PRICE_KEYS = ("base_price", "old_price", "regular_price", "original_price", "list_price")
URL_KEYS = ("url", "href", "link", "product_url", "canonical_url")
IMAGE_KEYS = ("image", "image_url", "thumbnail_url", "thumbnail", "img", "picture")
CONDITION_WORDS = {
    "used": ("pre-owned", "preowned", "pre owned", "used", "second hand", "مستعمل", "استعمال"),
    "refurbished": ("refurbished", "renewed", "مجدد", "ريفربش"),
    "open box": ("open box", "open-box", "اوبن بوكس"),
}


def _condition(d: dict) -> str | None:
    """Look for used/refurbished hints in the product's own fields (tags, condition, type, name)."""
    texts = []
    for k, v in d.items():
        if k in ("tags", "condition", "itemCondition", "product_type", "type", "badges", "labels", "name", "title",
                 "collections", "category"):
            texts.append(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v)
    blob = " ".join(texts).lower()
    if "newcondition" in blob.replace(" ", ""):
        return None
    for label, words in CONDITION_WORDS.items():
        if any(w in blob for w in words):
            return label
    return None


OOS_WORDS = ("out of stock", "out-of-stock", "outofstock", "sold out", "sold-out", "soldout", "notify me",
             "currently unavailable", "غير متوفر", "غير متاح", "نفذت الكمية", "نفدت الكمية", "نفذ من المخزون",
             "نفد من المخزون", "اعلمني", "أعلمني", "انتهت الكمية")


def _availability(d: dict) -> bool | None:
    """In stock according to the product's own data (schema.org, WooCommerce, Shopify, Magento...)."""
    offers = d.get("offers")
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    for src in (d, offers if isinstance(offers, dict) else {}):
        for k in ("availability", "stock_status", "stockStatus", "inventory_status"):
            v = src.get(k)
            if isinstance(v, str) and v:
                low = v.lower().replace(" ", "").replace("_", "")
                if any(w in low for w in ("outofstock", "soldout", "discontinued", "unavailable")):
                    return False
                if any(w in low for w in ("instock", "available", "limitedavailability", "onlineonly")):
                    return True
        for k in ("is_in_stock", "in_stock", "inStock", "available", "isAvailable"):
            if isinstance(src.get(k), bool):
                return src[k]
    return None


def _visible_text(html: str) -> str:
    text = re.sub(r"<head\b.*?</head>|<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>", " ", html,
                  flags=re.S | re.I)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text)))


def _page_says_sold_out(html: str, visible: str, title: str) -> bool | None:
    """Single product page: what the shopper sees next to the title/price wins over the page's data
    (some stores keep 'InStock' in their data while the button says 'غير متوفر')."""
    meta = re.search(r'<meta[^>]+(?:product:availability|og:availability|twitter:data2)[^>]+content="([^"]+)"', html, re.I)
    if meta and any(w in meta.group(1).lower() for w in ("out of stock", "sold out", "oos", "غير متوفر")):
        return True
    low = visible.lower()
    words = title.lower().split()
    for n in (6, 4, 2):  # find where the title is shown, using fewer words if the page shortens it
        probe = " ".join(words[:n])
        if probe and (i := low.find(probe)) != -1:
            return any(w in low[i:i + 900] for w in OOS_WORDS)
    return None


def _visible_numbers(html: str) -> set[float]:
    """Numbers a shopper can see on the page (scripts and tags removed)."""
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    out = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            out.add(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def _anchors(html: str) -> list[tuple[str, str]]:
    out = []
    for m in re.finditer(r"<a\b[^>]*href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", html, re.S | re.I):
        text = re.sub(r"<[^>]+>|\s+", " ", m.group(2)).strip().lower()
        out.append((m.group(1), text))
    return out


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06ff]+", " ", t.lower()).strip()


def _json_texts(html: str):
    """Yield text chunks that may contain JSON: script bodies, plus decoded Next.js flight data."""
    flight = []
    for m in re.finditer(r"self\.__next_f\.push\(\[\d+,\s*(\"(?:[^\"\\]|\\.)*\")\]\)", html, re.S):
        try:
            flight.append(json.loads(m.group(1)))
        except ValueError:
            pass
    if flight:
        yield "".join(flight)
    for m in re.finditer(r"<script[^>]*>(.*?)</script>", html, re.S):
        body = m.group(1)
        if "__next_f" not in body and ("{" in body):
            yield body


def _objects(text: str):
    """Decode every top-level JSON object/array found anywhere in text."""
    i, n = 0, len(text)
    while i < n:
        j = min((k for k in (text.find("{", i), text.find("[{", i)) if k != -1), default=-1)
        if j == -1:
            return
        try:
            obj, end = _DECODER.raw_decode(text, j)
            yield obj
            i = end
        except ValueError:
            i = j + 1


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    if isinstance(v, str):
        return parse_price(v)
    if isinstance(v, dict):  # {"final_price": 349, ...} or {"amount": "349.00"}
        for k in PRICE_KEYS:
            if k in v and (x := _num(v[k])):
                return x
    return None


def _first(d: dict, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, dict):
            v = v.get("url") or v.get("src") or v.get("href")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _as_product(d: dict, store: str, base_url: str, url_pattern: str | None, image_base: str | None):
    name = _first(d, NAME_KEYS)
    if not name or len(name) < 4 or "$" in name[:1] or name.strip().lower() == "default title":
        return None, []
    price = None
    for k in PRICE_KEYS:
        if k in d and (price := _num(d[k])):
            break
    if price is None and isinstance(d.get("offers"), (dict, list)):
        offers = d["offers"][0] if isinstance(d["offers"], list) and d["offers"] else d["offers"]
        price = _num(offers) if isinstance(offers, dict) else None
    if not price:
        return None, []
    currency = d.get("priceCurrency") or d.get("currency")
    if not currency and isinstance(d.get("offers"), (dict, list)):
        o = d["offers"][0] if isinstance(d["offers"], list) and d["offers"] else d["offers"]
        currency = o.get("priceCurrency") if isinstance(o, dict) else None
    if isinstance(currency, str) and currency.upper() not in {a.upper() for a in market.current().currency_aliases}:
        return None, []  # listing in another country's currency
    pdict = d.get("price") if isinstance(d.get("price"), dict) else d
    old = None
    for k in OLD_PRICE_KEYS:
        if k in pdict and (old := _num(pdict[k])):
            break
    coupon = None
    if pdict is not d and "final_without_coupon" in pdict:  # B.TECH style: coupon taken off at checkout
        after = _num(pdict.get("final_price"))
        coupon = after if after and after < price else None
    url = _first(d, URL_KEYS)
    if not url and url_pattern and d.get("slug"):
        url = url_pattern.format(slug=d["slug"])
    if not url and isinstance(d.get("handle"), str):  # Shopify stores
        url = f"/products/{d['handle']}"
    image = _first(d, IMAGE_KEYS)
    if image and image_base and not image.startswith(("http", "//", "/")):
        image = image_base.rstrip("/") + "/" + image
    return Product(
        store=store, title=name, price=price, url=urljoin(base_url, url or ""),
        image=urljoin(base_url, image) if image else None,
        old_price=old if old and old > price else None,
        coupon_price=coupon,
        condition=_condition(d),
        in_stock=_availability(d),
    ), [str(d[k]) for k in ("slug", "handle", "sku", "id", "product_id") if d.get(k)]


_CURRENCY = re.compile(r"(?:priceCurrency|currencyCode|currency)[\"']?\s*[:=]\s*[\"']?([A-Z]{3})\b")


def foreign_currency(html: str) -> str | None:
    """The page's currency when the store sells in another country's currency, else None."""
    local = {a.upper() for a in market.current().currency_aliases}
    codes = [c for c in _CURRENCY.findall(html) if c.isalpha()]
    if not codes or any(c in local for c in codes):
        return None
    return max(set(codes), key=codes.count)


def extract_products(html: str, store: str, base_url: str, url_pattern: str | None = None,
                     image_base: str | None = None) -> list[Product]:
    found: dict[str, tuple[Product, list[str]]] = {}
    prices: dict[str, set[float]] = {}  # every price seen for the same product (raw + formatted copies)
    page = base_url.rstrip("/")
    if foreign_currency(html):
        return []

    def walk(node, depth=0):
        if depth > 40:
            return
        if isinstance(node, dict):
            p, hints = _as_product(node, store, base_url, url_pattern, image_base)
            if p:
                key = p.url if p.url.rstrip("/") != page else p.title
                found.setdefault(key, (p, hints))
                prices.setdefault(key, set()).add(p.price)
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, depth + 1)

    for text in _json_texts(html):
        for obj in _objects(text):
            walk(obj)
    if not found:
        return []

    visible = _visible_numbers(html)
    # Decimal-formatted numbers anywhere in the page data ("46999.0"): a cents value never has decimals.
    decimals = {float(m) for m in re.findall(r"(?<![\d.])(\d{3,7}\.\d{1,2})(?![\d.])", html)}
    anchors = None
    out = []
    for key, (p, hints) in found.items():
        # Prices stored in cents/piasters (e.g. Shopify 3649900 = 36,499.00): trust what the page shows,
        # or another copy of the same product's price on the page.
        # (only for big numbers: a real 4,399 price must never turn into 43.99)
        if p.price >= 20000 and p.price not in visible and (
                p.price / 100 in visible or p.price / 100 in prices[key] or p.price / 100 in decimals):
            p.price /= 100
            p.old_price = p.old_price / 100 if p.old_price else None
            p.coupon_price = p.coupon_price / 100 if p.coupon_price else None
        # No own link (only the listing page): find the product's link among the page's <a> tags.
        if p.url.rstrip("/") == page:
            anchors = anchors if anchors is not None else _anchors(html)
            title = _norm(p.title)[:40]
            for href, text in anchors:
                if any(h and len(h) > 3 and h.lower() in href.lower() for h in hints) or (title and title in _norm(text)):
                    p.url = urljoin(base_url, href)
                    break
        others = [q.url for q, _ in found.values() if q.url != p.url]
        if p.url.rstrip("/") != page:
            if p.condition is None:
                p.condition = _card_condition(html, p.url, others)
            if _card_has(html, p.url, others, OOS_WORDS):
                p.in_stock = False
        out.append(p)
    # A product page (one main product): trust what the buy box shows.
    main = [p for p in out if p.url.rstrip("/") == page or len(out) == 1]
    if main:
        visible = _visible_text(html)
        for p in main:
            if _page_says_sold_out(html, visible, p.title):
                p.in_stock = False
    return out


OOS_CLASSES = {"outofstock", "out-of-stock", "out_of_stock", "sold-out", "soldout", "is-sold-out", "unavailable"}


def _card_has(html: str, url: str, other_urls: list[str], words) -> bool:
    """Sold out = the card shows the words, or the card element carries a sold-out CSS class
    (only class names count: themes often list 'soldout' in settings attributes of every card)."""
    raw = _card_window(html, url, other_urls, raw=True)
    if not raw:
        return False
    classes = {c for attr in re.findall(r'class="([^"]*)"', raw) for c in attr.split()}
    text = re.sub(r"<[^>]+>", " ", raw)
    return bool(classes & OOS_CLASSES) or any(w in text for w in words)


def _card_condition(html: str, url: str, other_urls: list[str]) -> str | None:
    """Used/refurbished badges often live in the product card's HTML, not in the data."""
    window = _card_window(html, url, other_urls)
    if not window:
        return None
    for label, words in CONDITION_WORDS.items():
        if any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", window) for w in words):
            return label
    return None


def _card_window(html: str, url: str, other_urls: list[str], raw: bool = False) -> str:
    """The product card around the product's link, without crossing into a neighbouring card.
    raw=True keeps tag attributes (stores mark sold-out cards with classes like 'outofstock')."""
    path = urlparse(url).path
    if len(path) < 5 or (pos := html.find(path)) == -1:
        return ""
    lo, hi = max(0, pos - 2500), pos + 800
    for other in other_urls:
        op = urlparse(other).path
        if len(op) < 5 or op.startswith(path) or path.startswith(op):
            continue
        j = html.rfind(op, lo, pos)
        if j != -1:
            lo = j + len(op)
        k = html.find(op, pos + len(path), hi)
        if k != -1:
            hi = k
    window = re.sub(r"<script.*?(</script>|$)|<style.*?(</style>|$)", " ", html[lo:hi], flags=re.S | re.I)
    if raw:
        return unescape(window).lower()
    return unescape(re.sub(r"<[^>]+>", " ", window)).lower()
