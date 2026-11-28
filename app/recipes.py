"""How to read one store's product page without the LLM, learned by the LLM once per store.

Why: checking a result's price on its own page is the step that costs most LLM calls (one per page without clear
product data). Most of those pages come from a few stores, and every page of a store is built the same way.

How (no code per store):
  1. A page's main price is known: the LLM read it (agent._read_main_price), or the page was clear without it (its buy
     box, app/buybox.py). Clear pages teach too, at no LLM cost (after_clear_read): every store we open learns.
  2. We list the page's elements that show that exact number, and the buttons / marks near the buy area, each with a
     CSS selector. When only one element shows it (and one buy button), that's the recipe (learn_free). Else the LLM picks the one that is the main product's price, the buy button and the sold-out mark
     (LEARN_PROMPT). It's checked on that page right away: the selector must give the same price.
  3. The next pages of that store: while the recipe is new, the LLM still reads them and the recipe must agree
     (TRUST_AFTER pages in a row). Then it replaces the LLM read, and every SPOT_CHECK-th page is still read by the
     LLM too. A page where it finds nothing, or a price the LLM disagrees with, counts as a failure; after
     RELEARN_AFTER failures in a row the LLM learns the store again from the next page it reads.

Accuracy: a recipe only ever reads the page opened just now, the number must be shown on that page, and a size or
color the shopper wants always goes to the LLM (a recipe can't tell which option a price is for).
"""
import asyncio
import json
import logging
import re

from selectolax.lexbor import LexborHTMLParser as HTMLParser, LexborNode as Node

from . import db, llm, pricemodel, trust

log = logging.getLogger("uvicorn.error")

TRUST_AFTER = 3     # pages in a row where the recipe gave the LLM's price before it is used alone
SPOT_CHECK = 10     # a trusted recipe: every n-th page is still read by the LLM, so a changed page is noticed
RELEARN_AFTER = 2   # failures in a row before the store is learned again
MAX_CANDIDATES = 14
ENABLED = True      # (tests turn it off unless they check it)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,.")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_BUY_WORDS = re.compile(r"add.to.(?:cart|bag|basket)|buy now|[أا]ضف|اشتر|out.of.stock|sold.out|unavailable|"
                        r"غير متوفر|نفذ|نفدت|خلصان|in.stock|متوفر|notify me", re.I)
_CLASS_OK = re.compile(r"^[A-Za-z_][\w-]{0,40}$")
_HASHY = re.compile(r"\d{3,}|[a-z0-9]{6,}_[a-z0-9]+|__[a-zA-Z0-9]{5,}$")  # css-module hashes change with every build

LEARN_PROMPT = """This is a product page of the online store {domain}. Its main product is "{title}", and its price on
the page is {price}. Below are elements of the page, each with a CSS selector, how many elements that selector matches
on the page, and its text.

PRICE CANDIDATES (elements showing {price}):
{prices}

BUY AREA CANDIDATES (buttons and stock marks):
{buys}

Pick the selectors that will work on EVERY product page of this store: the main product's own current price (not a
related item, an old/crossed-out price, an installment or a per-month price), the button to buy it, and an element
that appears only when it is sold out. Prefer selectors that match 1 element. Reply JSON:
{{"price": "<one PRICE selector, copied exactly>" or null if none is the main price,
  "buy": "<one BUY selector, copied exactly>" or null, "sold_out": "<one BUY selector>" or null}}"""


def _number(text: str) -> float | None:
    m = _NUM.search((text or "").translate(_ARABIC_DIGITS))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _part(node: Node) -> str:
    """One step of a selector: the tag, its id or its steady classes."""
    attrs = node.attributes or {}
    if (i := attrs.get("id")) and _CLASS_OK.match(i) and not _HASHY.search(i):
        return f"{node.tag}#{i}"
    classes = [c for c in (attrs.get("class") or "").split() if _CLASS_OK.match(c) and not _HASHY.search(c)][:3]
    for k in ("itemprop", "data-testid", "data-test", "data-qa"):
        if (v := attrs.get(k)) and re.match(r"^[\w-]{1,40}$", v):
            return f'{node.tag}[{k}="{v}"]'
    return node.tag + "".join(f".{c}" for c in classes)


def _selector(node: Node) -> str:
    """The element's selector: its own step and up to two parents with an id or classes."""
    parts, cur = [_part(node)], node.parent
    while cur is not None and cur.tag not in ("body", "html", "-undef") and len(parts) < 3:
        step = _part(cur)
        if step != cur.tag:
            parts.insert(0, step)
            if "#" in step:
                break
        cur = cur.parent
    return " ".join(parts)


def _own_text(node: Node) -> str:
    return " ".join((node.text(deep=True, separator=" ") or "").split())


def _candidates(tree: HTMLParser, price: float) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    """(price candidates, buy-area candidates): (selector, elements it matches, text)."""
    want = {price, float(int(price))}
    prices: dict[str, tuple[int, str]] = {}
    buys: dict[str, tuple[int, str]] = {}
    for node in tree.css("body *"):
        if node.tag in ("script", "style", "noscript", "svg", "path", "option"):
            continue
        text = _own_text(node)
        if not text or len(text) > 60:
            continue
        child_same = any(_own_text(c) == text for c in node.iter() if c.tag != "-text")
        if child_same:  # the innermost element with this text is the one to read
            continue
        if _number(text) in want:
            sel = _selector(node)
            if sel not in prices:
                prices[sel] = (len(tree.css(sel)), text)
        elif node.tag in ("button", "a", "span", "div", "p", "input", "link") and _BUY_WORDS.search(text):
            sel = _selector(node)
            if sel not in buys:
                buys[sel] = (len(tree.css(sel)), text)
    return ([(s, n, t) for s, (n, t) in prices.items()][:MAX_CANDIDATES],
            [(s, n, t) for s, (n, t) in buys.items()][:MAX_CANDIDATES])


def apply(recipe: dict, html: str) -> tuple[float | None, bool | None]:
    """(the main product's price, in stock) as the recipe reads them from this page. (None, None) = it found nothing."""
    try:
        tree = HTMLParser(html)
        node = tree.css_first(recipe["price"]) if recipe.get("price") else None
        if node is None:
            return None, None
        price = _number(_own_text(node))
        stock = None
        if recipe.get("sold_out") and tree.css_first(recipe["sold_out"]) is not None:
            stock = False
        elif recipe.get("buy") and tree.css_first(recipe["buy"]) is not None:
            stock = True
        return price, stock
    except Exception as e:  # a selector this parser can't read: the recipe is no use
        log.info("recipe: can't apply %s: %s", recipe, e)
        return None, None


def _load(domain: str) -> dict | None:
    try:
        row = db.get_recipe(domain)
        if row:
            row["recipe"] = json.loads(row["recipe"])
        return row
    except Exception as e:  # the database is down: the LLM reads the page
        log.warning("recipe: database: %s", e)
        return None


def read(url: str, html: str) -> tuple[float, bool | None] | None:
    """The page read with its store's trusted recipe, instead of the LLM: (price, in stock), or None when the LLM has
    to read it (no trusted recipe, it found nothing, or this page is a spot check)."""
    if not ENABLED or "<" not in html[:2000]:
        return None
    domain = trust.domain(url)
    row = _load(domain)
    if not row or row["ok"] < TRUST_AFTER or row["fails"]:
        return None
    if (row["uses"] + 1) % SPOT_CHECK == 0:  # now and then the LLM reads it too, and the recipe must agree
        _safe(db.recipe_result, domain, None)
        return None
    price, stock = apply(row["recipe"], html)
    if price is None or price < 5:
        _safe(db.recipe_result, domain, False)
        log.info("recipe: %s found no price on %s", domain, url[:100])
        return None
    if stock is None:  # no buy button and no sold-out mark: can't tell it's for sale, the LLM reads the page
        return None
    _safe(db.recipe_result, domain, None)
    return price, stock


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        log.warning("recipe: database: %s", e)


async def after_llm_read(url: str, html: str, title: str, price: float) -> None:
    """The LLM just read this page's main price: check the store's recipe against it, or learn one."""
    if not ENABLED or "<" not in html[:2000]:  # text from the reader service: no page elements to learn from
        return
    await asyncio.to_thread(pricemodel.remember, url, html, price, "llm")  # an example for the learned reader
    if await asyncio.to_thread(_check, url, html, price):  # the database and page parsing: off the event loop
        return
    if await asyncio.to_thread(learn_free, url, html, price) is None:
        await learn(url, html, title, price)


def after_clear_read(url: str, html: str, price: float) -> None:
    """A page read without the LLM and clear (its buy box, its product data): its price is known just as surely as
    after an LLM read, so the store's recipe is checked against it, or learned from it, at no LLM cost. Every page of
    every store we open teaches the reader for the store's next pages (the ones that aren't clear by themselves)."""
    if not ENABLED or "<" not in html[:2000]:
        return
    try:
        pricemodel.remember(url, html, price, "clear")
        if not _check(url, html, price):
            learn_free(url, html, price)
    except Exception as e:  # (learning never breaks a search)
        log.info("recipe: learning from a clear read failed on %s: %s", url[:100], e)


def _check(url: str, html: str, price: float) -> bool:
    """The store's recipe checked against a page whose price is known. False: there's no recipe to check (or it
    failed too often and is learned again)."""
    domain = trust.domain(url)
    row = _load(domain)
    if not row or row["fails"] >= RELEARN_AFTER:
        return False
    got, _ = apply(row["recipe"], html)
    right = got is not None and abs(got - price) < 1
    _safe(db.recipe_result, domain, right)
    if not right:
        log.info("recipe: %s read %s, the page's price is %s on %s", domain, got, price, url[:100])
    return True


_SOLD_OUT = re.compile(r"out.of.stock|sold.out|unavailable|غير متوفر|نفذ|نفدت|خلصان", re.I)
_BUY_ONLY = re.compile(r"add.to.(?:cart|bag|basket)|buy now|[أا]ضف|اشتر", re.I)


def learn_free(url: str, html: str, price: float) -> dict | None:
    """A recipe learned with no LLM call from a page whose price is known: the element showing that price with the
    most exact selector, and the buy button. Like every recipe it's used alone only after TRUST_AFTER more pages of
    the store agree with their own known price, so a wrong pick is never used."""
    tree = HTMLParser(html)
    prices, buys = _candidates(tree, price)

    def first_shows(sel: str) -> bool:  # (a recipe reads the first element its selector matches)
        node = tree.css_first(sel)
        return node is not None and (n := _number(_own_text(node))) is not None and abs(n - price) < 1
    # the first element on the page (the buy box comes before related items) that shows the price by a selector
    # whose first match is it; fewest matches first. A wrong pick doesn't agree on the store's next pages and is
    # never trusted
    own = sorted((n, i, s) for i, (s, n, _) in enumerate(prices)
                 if not re.search(r"old|was|before|strike|regular|compare|list|related|recent|similar", s, re.I)
                 and first_shows(s))
    buy = sorted((n, i, s) for i, (s, n, t) in enumerate(buys) if _BUY_ONLY.search(t) and not _SOLD_OUT.search(t))
    if not own or not buy:
        return None
    sold = [s for s, n, t in buys if n == 1 and _SOLD_OUT.search(t)]
    recipe = {"price": own[0][2], "buy": buy[0][2], "sold_out": sold[0] if len(sold) == 1 else None}
    check, stock = apply(recipe, html)
    if check is None or abs(check - price) >= 1 or stock is False:
        return None
    _safe(db.save_recipe, trust.domain(url), json.dumps(recipe))
    log.info("recipe: learned %s without the LLM: %s", trust.domain(url), recipe)
    return recipe


async def learn(url: str, html: str, title: str, price: float) -> dict | None:
    """One LLM call: which of this page's elements is the main price, the buy button, the sold-out mark."""
    llm.background()  # (a task of its own: its context ends with it) shoppers' searches go first
    domain = trust.domain(url)
    prices, buys = await asyncio.to_thread(lambda: _candidates(HTMLParser(html), price))  # CPU work: off the loop
    if not prices:
        return None
    lines = lambda xs: "\n".join(f"- {s} | matches {n} | {t}" for s, n, t in xs) or "(none)"  # noqa: E731
    try:
        got = await llm.ask_json(LEARN_PROMPT.format(domain=domain, title=title[:150], price=f"{price:g}",
                                                     prices=lines(prices), buys=lines(buys)),
                                 timeout=15, prefer="bulk", step="learn_page")
    except llm.LLMError as e:
        log.info("recipe: learning %s failed: %s", domain, e.details or e)
        return None
    allowed_p, allowed_b = {s for s, _, _ in prices}, {s for s, _, _ in buys}
    recipe = {"price": got.get("price") if got.get("price") in allowed_p else None,
              "buy": got.get("buy") if got.get("buy") in allowed_b else None,
              "sold_out": got.get("sold_out") if got.get("sold_out") in allowed_b else None}
    if not recipe["price"] or not recipe["buy"]:  # without the buy button a recipe can't tell it's for sale
        return None
    check, stock = apply(recipe, html)
    if check is None or abs(check - price) >= 1 or stock is False:  # wrong on the very page it was learned from
        log.info("recipe: %s: the picked selector reads %s, not %s", domain, check, price)
        return None
    await asyncio.to_thread(_safe, db.save_recipe, domain, json.dumps(recipe))
    log.info("recipe: learned %s: %s", domain, recipe)
    return recipe
