"""A product page's own product, read with no LLM call: its name, its current price and whether it can be bought.

A store's product page shows its product the same way on every theme: a heading (<h1>) with the product's name and,
next to it, the buy box with the price and the buy button. Related products, other sellers and recently viewed items
come lower down, each with its own link. So (no store-specific selectors):

- the buy box is the smallest element around the <h1> that shows a price, as long as it holds no list of other
  products (MAX_OTHER_PRODUCTS links to other product pages or more means we climbed into the page's related items);
- its price is read like a listing card's (app/cards.py): itemprop="price", else the innermost price elements, else
  the numbers written with the currency, with crossed-out old prices removed. It must be exactly one price: two
  (installments, a "from ... to ...", a bundle) is not clear, and the page goes to the LLM;
- it can be bought when the buy box (or the part of the page right around it) has a buy button, and can't when it
  says sold out. Neither: not known, the page goes to the LLM.

What is read here is checked like everything else: the number must be shown on the page opened just now.
"""
import re
from html import unescape
from urllib.parse import urlparse

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import cards
from .extract import OOS_WORDS

MAX_UP = 14              # levels above the <h1> searched for the buy box
MAX_OTHER_PRODUCTS = 2   # links to other products a buy box may hold (a bundle, "see also"); more is a list
STOCK_UP = 3             # levels above the buy box searched for the buy button / sold-out mark
_BUY = re.compile(r"add.to.(?:cart|bag|basket)|buy.now|add.to.order|[أا]ض[فـ]+\s*(?:[اإ]لى|لل)\s?(?:ال)?(?:سل[ةه]|عرب[ةه])|"
                  r"[أا]ضف للسل[ةه]|اشتر[يى]? الآن|اشتر[يى]?\s*$|شراء الآن|اطلب الآن|^شراء$", re.I)
_MONTHLY = re.compile(r"/\s*(?:month|mo)\b|per month|monthly|شهري|شهرياً|/\s*شهر|بالشهر|في الشهر|installment|قسط|تقسيط", re.I)
_SAVED = re.compile(r"save|saving|you save|discount|off|توفير|التوفير|وفر|خصم|تخفيض|يبدأ من|يبدا من|starting|from\s*$", re.I)
_NOTIFY = {"notify me", "اعلمني", "أعلمني"}
_OOS_CLASSES = {"outofstock", "out-of-stock", "out_of_stock", "sold-out", "soldout", "is-sold-out"}


def _text(node) -> str:
    return " ".join((node.text(deep=True, separator=" ") or "").split())


def _heading(tree) -> tuple[object, str] | None:
    """The page's one <h1> naming a product (two words or more)."""
    named = [(h, " ".join(unescape(_text(h)).split())) for h in tree.css("h1")]
    named = [(h, t) for h, t in named if len(t.split()) >= 2 and len(t) <= 300]
    if len({t for _, t in named}) > 1:
        # several headings ("About this item", "Product information"): the product's is the one the page's own
        # title names too
        # (the title may word it a little differently: the heading with most of its words in the title, clearly)
        head = tree.css_first("title")
        title = _norm(head.text() if head else "")
        words = set(title.split())

        def share(t: str) -> float:
            own = _norm(t).split()
            return 1.0 if _norm(t)[:80] in title else sum(w in words for w in own) / max(1, len(own))
        scored = sorted({t: share(t) for _, t in named}.items(), key=lambda x: -x[1])
        if scored:  # a heading repeating the name ("Product Summary: <name>") is the same name, not another one
            top = _norm(scored[0][0])
            scored = scored[:1] + [x for x in scored[1:] if top not in _norm(x[0])]
        if not scored or scored[0][1] < 0.6 or (len(scored) > 1 and scored[1][1] >= scored[0][1] - 0.1):
            return None
        named = [(h, t) for h, t in named if t == scored[0][0]]
    if not named or len({t for _, t in named}) > 1:  # no name, or the page names several things
        return None
    return named[0]


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", unescape(t or "").lower()).split())


def _shape(url: str) -> tuple:
    """A store's product pages share a link shape: /product/<name>, /products/<name>, /<name>/p123..."""
    parts = [x for x in urlparse(url).path.lower().split("/") if x]
    return (len(parts), parts[0] if len(parts) >= 2 else "")


def _other_products(node, site: str, page: str, url: str) -> int:
    """Links to other products of the store (a category, a tag or a brand page isn't one)."""
    keys, shape = set(), _shape(url)
    for a in node.css("a[href]"):
        link = cards._product_link(a.attributes.get("href"), url, site)
        if link and (k := cards._key(link)) != page and _shape(link) == shape:
            keys.add(k)
            if len(keys) > MAX_OTHER_PRODUCTS:
                break
    return len(keys)


_ONLY_PRICE = re.compile(r"^\s*(?:" + cards._CURRENCY + r"|[A-Z]{3})?\s*(" + cards._NUMBER + r")\s*(?:"
                         + cards._CURRENCY + r"|[A-Z]{3})?\s*$", re.I)


def _inside(node, other) -> bool:
    n = node
    while n is not None:
        if n.mem_id == other.mem_id:
            return True
        n = n.parent
    return False


def _hidden(node, box) -> bool:
    """Not shown on the page: inside an element hidden by its style, its hidden attribute or a "...-hidden" class
    (a used offer, another option's price kept for a script). Its price isn't the one the shopper sees."""
    n = node
    while n is not None and n.mem_id != box.mem_id:
        style = (n.attributes.get("style") or "").replace(" ", "").lower()
        if ("display:none" in style or "visibility:hidden" in style or "hidden" in n.attributes
                or any(c.endswith("-hidden") for c in (n.attributes.get("class") or "").split())):
            # (a bare "hidden" / "d-none" class is often undone at wider screens, "hidden md:block": not counted)
            return True
        n = n.parent
    return False


_SPLIT = re.compile(r"(\d)\s*([.,])\s*(\d)")


def _prices(box, h1) -> set[float] | None:
    """The current prices the buy box shows (the heading's own numbers, "340 جم" in a name, aren't prices):
    itemprop="price", else the innermost price elements that show just a price, else the numbers written with the
    currency. Crossed-out old prices are removed. None: the box shows prices that can't be told apart (stop there)."""
    found = set()
    for node in box.css("[itemprop=price], [data-price-amount]"):
        v = node.attributes.get("content") or node.attributes.get("data-price-amount")
        if v and (n := _attr_number(v)) and not cards._old(node, box) and not _inside(node, h1) and not _hidden(node, box):
            found.add(n)
    if found:
        return found

    def just_price(node) -> float | None:
        # (a price drawn in parts, "45 . 72", is one number)
        m = _ONLY_PRICE.match(_SPLIT.sub(r"\1\2\3", cards._shown_text(node).translate(cards._ARABIC_NUMS)))
        return cards._num(m.group(1)) if m else None

    def priced(node) -> bool:
        return bool(cards._PRICE_CLASS.search(node.attributes.get("class") or "")) and bool(just_price(node))

    def part_of_price(node) -> bool:
        """Inside a price element that is itself one whole price ("45" and "72" of "AED 45.72")."""
        own, n = just_price(node), node.parent
        while n is not None and n.mem_id != box.mem_id:
            if priced(n) and just_price(n) != own:
                return True
            n = n.parent
        return False
    marked = False  # the box shows price elements (current or crossed out)
    for node in box.css("*"):
        if not cards._PRICE_CLASS.search(node.attributes.get("class") or "") or _inside(node, h1) or _hidden(node, box):
            continue
        marked = marked or bool(just_price(node))
        if cards._old(node, box):
            continue
        inner = [c for c in node.css("*") if c.mem_id != node.mem_id
                 and cards._PRICE_CLASS.search(c.attributes.get("class") or "") and not cards._old(c, box) and just_price(c)]
        whole = just_price(node)
        if whole and inner and all(part_of_price(c) for c in inner) and not any(
                just_price(c) == whole for c in inner):
            inner = []  # its inner price elements are only parts of this one price
        if not inner and (n := whole) and not part_of_price(node):
            found.add(n)
    if found:
        return found
    if marked:  # every price element it shows reads as an old price: its names can't be told apart, not clear
        return None
    parts = [t.text_content or "" for t in box.traverse(include_text=True)
             if t.tag == "-text" and not _inside(t, h1) and not cards._old(t.parent, box) and not _hidden(t.parent, box)]
    text = " ".join(" ".join(parts).split()).translate(cards._ARABIC_NUMS)
    # (unlike a card, either way of writing the price counts: a second number beside the currency is a second price)
    # ("جم" after a number is grams as often as pounds: "340 جم" is a weight)
    return ({n for m in cards._CURRENCY_FIRST.finditer(text) if not m.group(0).startswith("جم") and (n := cards._num(m.group(1)))}
            | {n for m in cards._CURRENCY_AFTER.finditer(text) if not m.group(0).endswith("جم") and (n := cards._num(m.group(1)))})


def _attr_number(v: str) -> float | None:
    """A price written for machines ("1698.999", "4240.00"): a plain decimal number."""
    try:
        n = float(v.strip().replace(",", ""))
    except ValueError:
        return cards._num(v)
    return n if n >= cards.MIN_PRICE else None


def _without_monthly(node, prices: set[float]) -> set[float]:
    """Prices that are an installment ("EGP 500 / month", "قسط يبدأ من 500") or a saving ("وفر 1,500") aren't the
    price. Only the words between this number and the numbers beside it count."""
    text = cards._shown_text(node).translate(cards._ARABIC_NUMS)
    out = set()
    for v in prices:
        other = False
        for m in re.finditer(cards._NUMBER, text):
            if cards._num(m.group(0)) != v:
                continue
            after = re.split(r"\d", text[m.end():m.end() + 30], maxsplit=1)[0]
            before = re.split(r"\d", text[max(0, m.start() - 25):m.start()][::-1], maxsplit=1)[0][::-1]
            if _SAVED.search(before) or _MONTHLY.search(after) or _MONTHLY.search(before):
                other = True
        if not other:
            out.add(v)
    return out

def _stock(node) -> bool | None:
    raw = (node.html or "").lower()
    text = cards._shown_text(node).lower()
    classes = {c for attr in re.findall(r'class="([^"]*)"', raw) for c in attr.split()}
    # ("notify me" alone can be a price-drop alert: not a sold-out mark here)
    sold_out = bool(classes & _OOS_CLASSES or any(w in text for w in OOS_WORDS if w not in _NOTIFY))
    buy = any((label := _text(b) or b.attributes.get("value") or b.attributes.get("aria-label") or "")
              and len(label) <= 40 and _BUY.search(label) and "disabled" not in b.attributes
              for b in node.css("button, input[type=submit], input[type=button], a"))
    # both: a theme's hidden "sold out" label beside a working buy button, or a sold-out option; not clear
    return None if sold_out == buy else buy


def read(html: str, url: str) -> tuple[str, float, bool | None] | None:
    """(the product's name, its one current price, can it be bought: True / False / None not known), or None when the
    page isn't clear this way."""
    if "<" not in html[:2000]:
        return None
    site = urlparse(url).netloc.lower().removeprefix("www.")
    if not site:
        return None
    tree = HTMLParser(html)
    tree.strip_tags(["script", "style", "noscript", "svg", "template"])
    if tree.body is None or not (got := _heading(tree)):
        return None
    h1, name = got
    page = cards._key(url)
    node, box = h1, None
    for _ in range(MAX_UP):
        node = node.parent
        if node is None or node.tag in ("body", "html"):
            return None
        if _other_products(node, site, page, url) > MAX_OTHER_PRODUCTS:
            return None  # climbed past the product into a list of others before any price
        prices = _prices(node, h1)
        if prices is None:
            return None
        if prices:
            # a number of the product's own name ("340 جم", "128GB") shown by a size picker isn't its price
            named = {cards._num(x) for x in re.findall(cards._NUMBER, name.translate(cards._ARABIC_NUMS))}
            prices = _without_monthly(node, prices) - named
            if len(prices) != 1:
                return None
            box = node
            break
    if box is None:
        return None
    price = prices.pop()
    stock, region = _stock(box), box
    for _ in range(STOCK_UP):
        if stock is not None:
            break
        region = region.parent
        if region is None or region.tag in ("body", "html") or _other_products(region, site, page, url) > MAX_OTHER_PRODUCTS:
            break
        stock = _stock(region)
    return name, price, stock
