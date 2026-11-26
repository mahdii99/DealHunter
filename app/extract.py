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


# ISO codes of currencies a page found from Egypt may really be in (the region + the big ones)
REAL_CURRENCIES = set("""EGP SAR AED KWD QAR BHD OMR JOD IQD LBP SYP YER MAD DZD TND LYD SDG TRY IRR ILS USD EUR GBP
CHF SEK NOK DKK PLN CZK HUF RON RUB UAH INR PKR BDT LKR NPR CNY HKD TWD JPY KRW SGD MYR IDR THB VND PHP AUD NZD CAD
MXN BRL ARS CLP COP PEN ZAR NGN KES GHS ETB TZS UGX XOF XAF""".split())
_CURRENCY = re.compile(r"(?:priceCurrency|currencyCode|currency)[\"']?\s*[:=]\s*[\"']?([A-Z]{3})\b")


def foreign_currency(html: str) -> str | None:
    """The page's currency when the store sells in another country's currency, else None."""
    local = {a.upper() for a in market.current().currency_aliases}
    codes = [c for c in _CURRENCY.findall(html) if c in REAL_CURRENCIES]  # not a template placeholder ("ABC")
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


_VARIANT_ATTR = re.compile(r"""\sdata-[\w-]*variations?[\w-]*=(["'])(\[.*?\])\1""", re.S | re.I)
_OPTION_KEYS = ("attributes", "options", "selectedOptions", "option_values", "variant_options")


def _variant_label(v: dict) -> str:
    """The option names of one variant: "46", "46 / Black" (WooCommerce attributes, Shopify option1-3, a title)."""
    parts: list[str] = []
    for k in _OPTION_KEYS:
        x = v.get(k)
        if isinstance(x, dict):
            parts += [str(y) for y in x.values() if isinstance(y, (str, int, float)) and str(y).strip()]
        elif isinstance(x, list):
            for y in x:
                if isinstance(y, (str, int, float)):
                    parts.append(str(y))
                elif isinstance(y, dict) and isinstance(y.get("value"), (str, int, float)):
                    parts.append(str(y["value"]))
    if not parts:
        parts = [str(v[k]) for k in ("option1", "option2", "option3") if isinstance(v.get(k), (str, int, float))]
    if not parts:  # not a product's name: a list of related products isn't a list of variants
        for k in ("size", "color", "variant_title"):
            if isinstance(v.get(k), (str, int)) and str(v[k]).strip():
                parts = [str(v[k])]
                break
    return " / ".join(p.strip() for p in parts if p.strip())[:40]


def variant_stock(html: str) -> list[tuple[str, bool, float | None]]:
    """The product's variants (sizes, colors...) with their stock, from the page's own data: WooCommerce's
    data-product_variations, Shopify's variants, any list of variant objects that carry a stock flag.
    A size menu often lists every size while only this data says which ones are sold out.
    Returns [(label, in stock, price or None)]."""
    texts = [unescape(m.group(2)) for m in _VARIANT_ATTR.finditer(html)] + list(_json_texts(html))
    out: dict[str, tuple[bool, float | None]] = {}

    def walk(node, depth=0):
        if depth > 30 or len(out) >= 120:
            return
        if isinstance(node, list):
            variants = [x for x in node if isinstance(x, dict)]
            if variants:
                for v in variants:
                    stock = _availability(v)
                    label = _variant_label(v)
                    if stock is not None and label:
                        price = _num(v.get("display_price") or v.get("price"))
                        if price and price >= 20000 and isinstance(v.get("price"), int) and "display_price" not in v:
                            price /= 100  # Shopify keeps cents in its page data
                        out.setdefault(label, (stock, price))
            for x in node:
                walk(x, depth + 1)
        elif isinstance(node, dict):
            for x in node.values():
                walk(x, depth + 1)
    for text in texts:
        for obj in _objects(text):
            walk(obj)
    return [(label, stock, price) for label, (stock, price) in out.items()]


def page_image(html: str, base_url: str) -> str | None:
    """The product page's own picture: its og:image / twitter:image, else the product data's image."""
    for prop in ("og:image", "og:image:secure_url", "twitter:image"):
        m = (re.search(r"<meta[^>]+(?:property|name)=[\"']" + re.escape(prop) + r"[\"'][^>]*content=[\"']([^\"']+)", html, re.I)
             or re.search(r"<meta[^>]+content=[\"']([^\"']+)[\"'][^>]*(?:property|name)=[\"']" + re.escape(prop) + r"[\"']", html, re.I))
        if m and m.group(1).strip():
            return urljoin(base_url, unescape(m.group(1).strip()))
    return None


_PRICE_BY_CURRENCY = re.compile(
    r"(?:(?:EGP|E\.?G\.?P|L\.?E\.?|ج\.?\s?م\.?|جنيه(?:\s?مصري)?)\s*(\d[\d,]*(?:\.\d{1,2})?)"
    r"|(\d[\d,]*(?:\.\d{1,2})?)\s*(?:EGP|E\.?G\.?P|L\.?E\.?|ج\.?\s?م\.?|جنيه))", re.I)
_ARABIC_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,.")
_OLD_PRICE = re.compile(r"<(span|div|p|del)\b[^>]*class=[\"'][^\"']*\b(?:old|was|before|regular|strike|crossed)"
                        r"[\w-]*[^\"']*[\"'][^>]*>[^<]*</\1>", re.I)
MAIN_WINDOW = 1200  # where a product page names its product: its own price is shown there, before related items


def page_title(html: str) -> str:
    """The product page's name for its product: its <h1>, else its og:title ('' when neither names one)."""
    h1 = re.search(r"<h1\b[^>]*>(.*?)</h1>", html, re.S | re.I)
    title = " ".join(unescape(re.sub(r"<[^>]+>", " ", h1.group(1))).split()) if h1 else ""
    if len(title.split()) < 2:
        og = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html, re.I)
        title = " ".join(unescape(og.group(1)).split()) if og else ""
    return title if 2 <= len(title.split()) and len(title) <= 300 else ""


def main_product(html: str) -> tuple[str, float] | None:
    """A single product page read with no LLM call: (its product's name, its price), or None when the page isn't
    clear (the LLM reads it then). Clear means the page names one product (its <h1>, or og:title) and right after
    that name shows exactly one price written with the currency. Two prices there (old and new, installments, a
    size's price) or a size/color to choose with prices of their own: not clear."""
    title = page_title(html)
    if not title:
        return None
    variants = variant_stock(html)
    if len({p for _, _, p in variants if p}) > 1:
        return None
    # a crossed-out old price isn't the price: it's dropped, the price shown beside it is read
    shown = re.sub(r"<(del|s|strike)\b[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    shown = _OLD_PRICE.sub(" ", shown)
    visible = _visible_text(shown)
    i = visible.find(title[:60])
    if i == -1:
        return None
    window = visible[i + len(title[:60]):i + MAIN_WINDOW].translate(_ARABIC_NUMS)
    prices = set()
    for m in _PRICE_BY_CURRENCY.finditer(window):
        try:
            v = float((m.group(1) or m.group(2)).replace(",", ""))
        except ValueError:
            continue
        if v >= 5:
            prices.add(v)
    return (title, prices.pop()) if len(prices) == 1 else None


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TEXT_TITLE = re.compile(r"^(?:Title:\s*|#{1,2}\s+)(.+)$", re.M)
_MONTHLY = re.compile(r"^\s*(?:/|per|a|كل|في)?\s*(?:month|mo\b|شهر|شهريا|شهرياً)", re.I)
_DISCOUNT = re.compile(r"%|خصم|وفر|was\b|save|off\b|بدلا من|بدلاً من|instead of|before", re.I)
_RELATED = re.compile(r"you may also like|related products|similar products|customers also|frequently bought|"
                      r"منتجات (?:ذات صلة|مشابهة)|قد يعجبك|شاهد أيضا", re.I)
_TEXT_BUY = re.compile(r"add to (?:cart|bag|basket)|buy now|[أا]ضف (?:[اإ]لى|لل) ?(?:ال)?سل[ةه]|[أا]ضف للسل[ةه]|اشتر[يى]? الآن|"
                       r"\bin stock\b|متوفر في المخزون", re.I)
TEXT_WINDOW = 1200  # after a product's name in a text copy: its own price is there, before related items


def text_prices_anywhere(text: str) -> bool:
    """A text copy writes at least one price with the currency ("EGP 1,299", "1,299 ج.م")."""
    from .cards import _CURRENCY_AFTER, _CURRENCY_FIRST
    text = (text or "").translate(_ARABIC_NUMS)
    return bool(_CURRENCY_FIRST.search(text) or _CURRENCY_AFTER.search(text))


def main_product_text(text: str, title: str = "") -> tuple[str, float, bool | None] | None:
    return read_text_copy(text, title)[0]


def read_text_copy(text: str, title: str = "") -> tuple[tuple[str, float, bool | None] | None, str]:
    """main_product_text, with why the copy wasn't clear ("" when it was): shown by the live check."""
    got, why = None, "no product name found in it"
    for got, why in _text_reads(text, title):
        if got:
            return got, ""
    return None, why


def _words(t: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", unescape(t).lower().replace("\\", "")).split()


def _swapped(name_words, line_words) -> bool:
    """The line writes another number where the name has one ("256GB" on a line, "128GB" in the name): another
    product, a sibling model or size. A line leaving some of the name's numbers out is fine."""
    nums = lambda ws: {w for w in ws if any(c.isdigit() for c in w)}
    return bool(nums(name_words) - nums(line_words)) and bool(nums(line_words) - nums(name_words))


def _name_line_end(text: str, name: str) -> int:
    """Where the line naming the product ends, when the copy writes its name another way than the title does
    ("Buy ...", other punctuation, "\\-" escapes, its words in another order): the first short line holding nearly
    all the name's words (4 of 5) and no other number in place of one of its own. -1: no such line."""
    want = set(_words(name))
    if len(want) < 3:
        return -1
    pos = 0
    for line in text.splitlines(keepends=True):
        got = _words(line)
        if got and len(got) <= 2 * len(want) + 3 and len(want & set(got)) >= 0.8 * len(want) and not _swapped(want, got):
            return pos + len(line)
        pos += len(line)
    return -1


def _text_reads(text: str, title: str):
    """A product page's text copy (a search service's or a reader's: no tags left) read with no LLM call:
    (its product's name, its price, False when it says sold out, True when it has a buy button, else None), or None when it isn't clear.
    `title` is the search service's title for the page; else the copy's own "Title:" or first heading names it.
    Clear means the name appears in the text and, right after it (before any related items), one price is written
    with the currency, or two where the page marks a discount (the lower one is the price); monthly installments
    don't count. Anything else goes to the LLM."""
    from .cards import _CURRENCY_AFTER, _CURRENCY_FIRST, _num
    text = _MD_LINK.sub(r"\1", _MD_IMAGE.sub(" ", text or ""))
    names = [title] + [m.group(1) for m in _TEXT_TITLE.finditer(text[:3000])]
    for name in names:
        name = " ".join(unescape(name or "").split())
        name = re.split(r"\s[|–—]\s", name)[0].strip()  # " | noon Egypt": the site's name, not the product's
        if len(name.split()) < 2 or len(name) > 300:
            continue
        low, probe = text.lower(), name.lower()[:60]
        i = low.find(probe)
        if i == -1:
            words = probe.split()[:4]
            i = low.find(" ".join(words)) if len(words) >= 3 else -1
            if i != -1 and _swapped(_words(name), _words(text[i:].split("\n", 1)[0])):
                i = -1  # the line found names a sibling model or size
        start = i + len(probe) if i != -1 else _name_line_end(text, name)
        if start == -1:
            yield None, "its name isn't in the text"
            continue
        window = text[start:start + TEXT_WINDOW].translate(_ARABIC_NUMS)
        if m := _RELATED.search(window):
            window = window[:m.start()]
        found = sorted(((m.start(), n) for pattern in (_CURRENCY_FIRST, _CURRENCY_AFTER) for m in pattern.finditer(window)
                        if (n := _num(m.group(1))) and not _MONTHLY.match(window[m.end():m.end() + 15])),
                       key=lambda x: x[0])
        prices = list(dict.fromkeys(n for _, n in found))
        if len(prices) == 1:
            price = prices[0]
        elif len(prices) == 2 and _DISCOUNT.search(window[:found[-1][0] + 40]) and max(prices) <= min(prices) * 3:
            price = min(prices)
        else:
            yield None, f"{len(prices)} prices after its name" + (f" ({', '.join(f'{x:g}' for x in prices[:4])})" if prices else "")
            return
        near = window[:600].lower()
        stock = False if any(w in near for w in OOS_WORDS) else True if _TEXT_BUY.search(near) else None
        yield (name, price, stock), ""
        return
