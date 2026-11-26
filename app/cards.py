"""Product cards of a store's listing page (search results, a category), read with no LLM call.

Most store pages carry their products as data (app/extract.py). Many others only show them as HTML: a card per
product, with the product's own link, its name and its price. Every store theme lays its cards out differently,
but they all share the same shape, and that shape is what is read here (no store-specific selectors):

- a card is the biggest element around a product link that holds no link to another product;
- its price is the one current price the card shows: a number written with the currency, or a price element
  (class "price", itemprop="price", data-price-amount), with crossed-out old prices removed. A card with two
  different current prices (installments, "from ... to ...") is not clear and is left out;
- its name is the link's text, or the card's heading, or the picture's alt text.

A page counts as a listing only when MIN_CARDS cards or more read this way: one priced link on a page is more
likely a banner than a product list. Whatever is read here is a candidate, like the LLM's reading: its price is
checked on the product's own page before a shopper sees it.
"""
import re
from html import unescape
from urllib.parse import unquote, urljoin, urlparse

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from .extract import CONDITION_WORDS, OOS_WORDS
from .models import Product

MIN_CARDS = 3          # priced cards that make a page a product list
MAX_CARD_DEPTH = 25    # levels above a product link searched for its card
MIN_PRICE = 5
_ARABIC_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,.")
_CURRENCY = r"(?:EGP|E\.?G\.?P|L\.?E\.?|ج\.?\s?م\.?|جنيه(?:\s?مصري)?|جم)"
_PRICE_TEXT = re.compile(r"(?:" + _CURRENCY + r"\s*(\d[\d,]*(?:\.\d{1,2})?)|(\d[\d,]*(?:\.\d{1,2})?)\s*" + _CURRENCY
                         + r")(?![\w؀-ۿ])", re.I)
_NUMBER = r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"  # 1,299 / 1.299,00 / 1299.5
_CURRENCY_FIRST = re.compile(r"(?<![A-Za-z\u0600-\u06ff])" + _CURRENCY + r"\s*(" + _NUMBER + r")(?![\d.,]*\d)", re.I)
_CURRENCY_AFTER = re.compile(r"(?<![\d.,])(" + _NUMBER + r")\s*" + _CURRENCY + r"(?![\w\u0600-\u06ff])", re.I)
_OLD_WORD = re.compile(r"old|was|before|regular|strike|crossed|compare|original|list|line-?through|del", re.I)


def _is_old_class(classes: str) -> bool:
    """A class naming a crossed-out price: "old-price", "price--compare", "regular-price", "strike"... A word like
    "washing" in a category class ("product_cat-washing-machines") is not one."""
    for c in classes.lower().split():
        if ":" in c or "[" in c:
            continue  # a utility class styling its children ("[&_.old-price]:hidden"), not this element
        if c in ("strike", "strikethrough", "line-through", "crossed", "was", "old"):
            return True
        if ("price" in c or "amount" in c) and _OLD_WORD.search(c.replace("price", " ")):
            return True
    return False


_PRICE_CLASS = re.compile(r"(?:^|[\s_-])(?:price|amount|money|final-price|special-price|sale-price)(?:$|[\s_-])", re.I)
_NOT_PRODUCT = re.compile(r"/(?:cart|checkout|account|login|register|wishlist|compare|search|tag|tags|blog|"
                          r"page|pages|policy|policies|privacy|terms|returns?|refund|shipping|delivery|contact|about|faq|brands?|customer|"
                          r"my-account)(?:/|$)|سياس[ةه]|الاسترجاع|الشحن|اتصل", re.I)
_OOS_CLASSES = {"outofstock", "out-of-stock", "out_of_stock", "sold-out", "soldout", "is-sold-out", "unavailable"}
_JUNK_NAME = re.compile(r"^(?:add to (?:cart|bag|basket)|buy now|shop now|quick ?view|view (?:full )?details|view product|"
                        r"more details|learn more|read more|select options|choose options|compare|"
                        r"[اأ]ضف (?:الى|إلى|لل)\s?السل[ةه]|اشتري? الآن|شراء|عرض سريع|معاينة سريعة|التفاصيل|"
                        r"عرض التفاصيل|اقرأ المزيد|قراءة المزيد)$", re.I)


_BADGES = re.compile(r"^\s*\d+\s*/\s*\d+\b|(?:خصم|off|sale|save|وفر)?\s*-?\d+(?:\.\d+)?\s*%\s*(?:off|خصم)?|"
                     r"لفترة محدودة|نفذ(?:ت)? الكمية|out of stock|sold out|new arrival|\bnew\b|\bsale\b|جديد", re.I)


def _num(text: str) -> float | None:
    text = text.translate(_ARABIC_NUMS).strip()
    # the last mark before 1-2 digits is the decimal point ("1.299,00", "800,00", "1,299.50"); every other mark
    # groups thousands
    if m := re.fullmatch(r"([\d.,]*?)[.,](\d{1,2})", text):
        text = re.sub(r"[.,]", "", m.group(1)) + "." + m.group(2)
    else:
        text = re.sub(r"[.,]", "", text)
    try:
        v = float(text)
    except ValueError:
        return None
    return v if v >= MIN_PRICE else None


def _key(url: str) -> str:
    p = urlparse(url)
    return (p.netloc.lower().removeprefix("www.") + p.path.rstrip("/").lower()) if p.path else ""


def _product_link(href: str, base: str, site: str) -> str | None:
    """A link that can be a product page of this store: same site, a path long enough to name a product."""
    href = (href or "").strip()
    if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
        return None
    url = urljoin(base, href)
    p = urlparse(url)
    if p.scheme not in ("http", "https") or p.netloc.lower().removeprefix("www.") != site:
        return None
    path = p.path.rstrip("/")
    if len(path) < 9 or _NOT_PRODUCT.search(unquote(path)) or re.search(r"\.(?:jpe?g|png|webp|gif|svg|pdf|css|js)$", path, re.I):
        return None
    return url.split("#")[0]


def _prices(card) -> set[float]:
    """The current prices one card shows (old crossed-out prices removed)."""
    found: set[float] = set()
    for node in card.css("[itemprop=price], [data-price-amount]"):
        v = node.attributes.get("content") or node.attributes.get("data-price-amount")
        if v and (n := _num(v)):
            if not _old(node, card):
                found.add(n)
    if found:
        return found
    def number(node) -> float | None:
        text = " ".join(node.text(deep=True, separator=" ").split()).translate(_ARABIC_NUMS)
        if (got := _text_prices(text)) and len(got) == 1:  # the number written with the currency
            return got.pop()
        nums = [n for x in re.findall(_NUMBER, text) if (n := _num(x))]
        return nums[0] if len(nums) == 1 else None
    priced = [n for n in card.css("*") if _PRICE_CLASS.search(n.attributes.get("class") or "")]
    for node in priced:
        if _old(node, card):
            continue
        # the innermost price elements that show a number (a price box holds the old and new prices as children)
        inner = [c for c in node.css("*") if c.mem_id != node.mem_id and _PRICE_CLASS.search(c.attributes.get("class") or "")
                 and not _old(c, card) and number(c)]
        if not inner and (n := number(node)):
            found.add(n)
    if found:
        return found
    return _text_prices(_shown_text(card))


def _text_prices(text: str) -> set[float]:
    """Prices written with the currency. A card writes them one way ("EGP 2,792" or "2,792 ج.م"): read either
    way, "5 من 5 EGP 2,792" (a rating before the price) would also give 5, so the way that reads the most
    numbers next to a currency, each only once, wins; a tie reads both."""
    text = text.translate(_ARABIC_NUMS)
    first = {n for m in _CURRENCY_FIRST.finditer(text) if (n := _num(m.group(1)))}
    after = {n for m in _CURRENCY_AFTER.finditer(text) if (n := _num(m.group(1)))}
    if first and after and first != after:
        # the same price read both ways is the price; otherwise the bigger numbers are the prices (a rating or a
        # count beside the currency is small)
        both = first & after
        if both:
            return both
        return first if min(first) >= min(after) else after
    return first or after


def _old(node, card) -> bool:
    """A crossed-out / "was" price: the element or a parent inside the card says so."""
    n = node
    while n is not None:
        if n.mem_id == card.mem_id:
            return False
        if n.tag in ("del", "s", "strike") or _is_old_class(n.attributes.get("class") or ""):
            return True
        n = n.parent
    return False


def _shown_text(card) -> str:
    parts = []
    for node in card.traverse(include_text=True):
        if node.tag == "-text":
            n, old = node.parent, False
            while n is not None and n.mem_id != card.mem_id:
                if n.tag in ("del", "s", "strike") or _is_old_class(n.attributes.get("class") or ""):
                    old = True
                    break
                n = n.parent
            if not old:
                parts.append(node.text_content or "")
    return " ".join(" ".join(parts).split())


def _name(card, anchors) -> str:
    """The product's name: the text (or title) of its own link, else the card's heading, else its picture's alt
    text. Buttons ("View Full Details"), prices and slide counters ("1 / 6") aren't names."""
    def fine(n: str) -> str:
        n = _BADGES.sub(" ", " ".join(unescape(n or "").split()))
        n = " ".join(n.strip(" ·|-–—").split())
        words = [w for w in n.split() if re.search(r"[^\W\d_]{2,}", w)]
        return n if (len(words) >= 2 and 8 <= len(n) <= 220 and not _JUNK_NAME.match(n)
                     and not _text_prices(n)) else ""
    groups = [[t for a in anchors for t in (" ".join(a.text(deep=True, separator=" ").split()), a.attributes.get("title"))],
              [" ".join(h.text(deep=True, separator=" ").split()) for h in card.css("h1, h2, h3, h4, h5, h6")],
              [img.attributes.get("alt") for img in card.css("img[alt]")]]
    for group in groups:
        if ok := [n for x in group if (n := fine(x))]:
            return max(ok, key=len)  # (the title link's text is longer than the picture link's alt or a badge)
    return ""


def _image(card, base: str) -> str | None:
    for img in card.css("img"):
        for k in ("data-src", "data-lazy-src", "data-original", "src"):
            v = (img.attributes.get(k) or "").strip()
            if v and not v.startswith("data:"):
                return urljoin(base, v)
    return None


def listing_cards(html: str, store: str, base_url: str) -> list[Product]:
    """The products of a listing page's cards, [] when the page isn't clearly a list of priced products."""
    site = urlparse(base_url).netloc.lower().removeprefix("www.")
    if not site:
        return []
    tree = HTMLParser(html)
    tree.strip_tags(["script", "style", "noscript", "svg", "template"])
    if tree.body is None:
        return []
    page = _key(base_url)
    by_key: dict[str, list] = {}
    under: dict[int, set[str]] = {}  # element -> the product links inside it (filled once, walking up from each link)
    for a in tree.body.css("a[href]"):
        url = _product_link(a.attributes.get("href"), base_url, site)
        if not url or (k := _key(url)) == page:
            continue
        by_key.setdefault(k, []).append((url, a))
        n = a.parent
        while n is not None:
            got = under.setdefault(n.mem_id, set())
            if k in got and len(got) > 1:
                break  # (everything above already has it, and another link too)
            got.add(k)
            n = n.parent
    if len(by_key) < MIN_CARDS:
        return []
    out, seen = [], set()
    for k, links in by_key.items():
        url, a = links[0]
        card, node = None, a
        for _ in range(MAX_CARD_DEPTH):
            up = node.parent
            if up is None or up.tag in ("body", "html") or under.get(up.mem_id, set()) - {k}:
                break
            node = up
            card = up
        if card is None or k in seen:
            continue
        prices = _prices(card)
        if len(prices) != 1:
            continue
        name = _name(card, [x for _, x in links])
        if not name:
            continue
        seen.add(k)
        raw = (card.html or "").lower()
        classes = {c for attr in re.findall(r'class="([^"]*)"', raw) for c in attr.split()}
        text = _shown_text(card).lower()
        condition = next((label for label, words in CONDITION_WORDS.items()
                          if any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text) for w in words)), None)
        old = [v for node in card.css("*") if (node.tag in ("del", "s", "strike")
                                               or _is_old_class(node.attributes.get("class") or ""))
               and (m := re.search(_NUMBER, node.text(deep=True).translate(_ARABIC_NUMS)))
               and (v := _num(m.group(0)))]
        price = prices.pop()
        out.append(Product(store=store, title=name[:200], price=price, url=url, image=_image(card, base_url),
                           old_price=max(old) if old and price < max(old) <= price * 10 else None,
                           condition=condition,
                           in_stock=False if (classes & _OOS_CLASSES or any(w in text for w in OOS_WORDS)) else None))
    return out if len(out) >= MIN_CARDS else []
