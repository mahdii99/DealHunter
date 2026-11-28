"""Today's best deals: the products of our price list (app/catalog.py) that really got cheaper, checked on their own
store page just before they're listed.

A shopper trusts a deal only when it's real. Stores often show a high "old price" that the product never sold at, so
what counts most here is our own history: a price below what the same store page asked for it in the last 30 days.
A store's own discount is listed after those, and says it's the store's number, not ours; one from a "before" price
the page never asked in our 30 days of history is left out (app/pricechart.verdict says "fake").

Prices on the page are never older than their check: every deal was opened on its store page (the same check as a
search result, agent.checked_only) when the list was made, and the page shows when that was. The list is made
after each price list run (.github/workflows/scrape-prices.yml) and stops being shown after MAX_AGE_HOURS.
"""
import asyncio
import json
import logging
import re
import time

from . import agent, catalog, db, market, pricechart, query
from .models import Product

log = logging.getLogger("uvicorn.error")

KEY = "__deals__"           # the list is kept in `searches` under this request
MAX_AGE_HOURS = 14          # an older list isn't shown (prices move)
HISTORY_DAYS = 30
MIN_DROP = 5                # % below the lowest price seen before today, to count as a real drop
MIN_STORE_DISCOUNT = 15     # % off the store's own old price, for the store-discount part
CHECK = 300                 # candidates opened per run (each one is a store page, read with no LLM call)
BATCH = 25                  # opened together (agent.checked_only checks at most VERIFY_MAX pages per call)
PER_CATEGORY = 40
PER_STORE = 200


def candidates(now: float | None = None) -> list[dict]:
    """Price list products seen in the last day, with how their price moved, best deals first."""
    now = now or time.time()
    mk = market.current().code
    rows = db.catalog_since(mk, now - 26 * 3600)
    if not rows:
        return []
    rows = _accessories_apart(rows)
    hist = db.histories([r["url"] for r in rows], now - HISTORY_DAYS * 86400)
    out = []
    for r in rows:
        points = hist.get(r["url"]) or []
        before = [p["price"] for p in points if now - HISTORY_DAYS * 86400 <= p["seen_at"] < now - 20 * 3600]
        low = min(before) if before else None
        drop = round((low - r["price"]) / low * 100) if low and low > r["price"] else 0
        off = (round((r["old_price"] - r["price"]) / r["old_price"] * 100)
               if r.get("old_price") and r["old_price"] > r["price"] else 0)
        judged = pricechart.verdict(points, r["price"], r.get("old_price"), now)
        trail = [[round(p["seen_at"]), p["price"]] for p in points]  # for the card's small chart
        if drop >= MIN_DROP and drop <= 70:
            out.append({**r, "kind": "drop", "was": low, "pct": drop, "hist": trail})
        elif MIN_STORE_DISCOUNT <= off <= 70 and not (judged and judged["kind"] == "fake"):
            # a "before" price this page never asked in 30 days is no deal (stores inflate it)
            out.append({**r, "kind": "store", "was": r["old_price"], "pct": off, "hist": trail,
                        "real": bool(judged and judged["kind"] == "real")})
    out.sort(key=lambda d: (d["kind"] != "drop", -d["pct"], -(d["was"] - d["price"])))
    picked, per_cat, per_store = [], {}, {}
    for d in out:
        if per_cat.get(d["category"], 0) >= PER_CATEGORY or per_store.get(d["store"], 0) >= PER_STORE:
            continue
        per_cat[d["category"]] = per_cat.get(d["category"], 0) + 1
        per_store[d["store"]] = per_store.get(d["store"], 0) + 1
        picked.append(d)
    return picked


def fair_share(found: list[dict]) -> list[dict]:
    """The best deals of every store in turn (each store's own order kept), so the checks of a run aren't all spent
    on the stores with the most products: a smaller store's real deals reach the page too."""
    by_store: dict[str, list[dict]] = {}
    for d in found:
        by_store.setdefault(d["store"], []).append(d)
    out, queues = [], list(by_store.values())
    while queues:
        out += [q.pop(0) for q in queues]
        queues = [q for q in queues if q]
    return out


ACCESSORY = "accessories"
ACCESSORY_SHARE = 0.15      # far under the usual price of its category: a case, a bag, a stand, not the product


def _accessories_apart(rows: list[dict]) -> list[dict]:
    """A store's search for "washing machine" also lists washer bags and covers. Shoppers browsing washers don't want
    those, so anything priced far under its category's middle price goes under accessories instead."""
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        conf = catalog.CATEGORIES.get(r["category"], {})
        if not conf.get("broad") and not conf.get("mixed"):  # one kind of product at one kind of price
            by_cat.setdefault(r["category"], []).append(r["price"])
    mid = {c: sorted(v)[len(v) // 2] for c, v in by_cat.items() if len(v) >= 10}
    return [{**r, "category": ACCESSORY} if r["category"] in mid and r["price"] < mid[r["category"]] * ACCESSORY_SHARE
            else r for r in rows]


async def build() -> dict:
    """Opens the best candidates on their store pages and keeps those still at (or under) that price."""
    found = fair_share(await asyncio.to_thread(candidates))[:CHECK]  # the database: off the event loop
    # no picture given: the product page opened for the check gives its own (a list page's can be broken or old)
    picks = [Product(store=d["store"], title=d["title"], price=d["price"], url=d["url"], image=None,
                     old_price=d.get("old_price"), trusted=True, trust_score=100) for d in found]
    by_url = {d["url"]: d for d in found}
    shown, unchecked, gone = [], [], 0
    for i in range(0, len(picks), BATCH):  # a few at a time: each batch gets the agent's full time to check its pages
        batch = picks[i:i + BATCH]
        got = await agent.checked_only(batch, patient=True, use_llm=False)
        shown += got[0]
        unchecked += got[1]
        gone += got[2]
    deals = []
    for p in shown:
        d = by_url[p.url]
        if p.price > d["price"] + 1:  # dearer than when we listed it: not a deal any more
            continue
        was = d["was"]
        about = _attrs(d)
        deals.append({"title": p.title, "store": p.store, "url": p.url, "image": p.image or d["image"] or None,
                      "price": p.price, "was": was, "pct": round((was - p.price) / was * 100), "kind": d["kind"],
                      "category": d["category"], "checked_at": p.checked_at, "hist": d.get("hist") or [],
                      "real": d["kind"] == "drop" or bool(d.get("real")),
                      "brand": about.get("brand"), "kind_ar": about.get("kind_ar")})
    deals.sort(key=lambda d: (d["kind"] != "drop", -d["pct"]))
    payload = {"checked_at": time.time(), "deals": deals, "candidates": len(found), "sold_out": gone,
               "unchecked": len(unchecked)}
    await asyncio.to_thread(db.save_search, db.cache_key(KEY, market.current().code), market.current().code, payload)
    log.info("deals: %d of %d candidates checked on their page (%d sold out)", len(deals), len(found), gone)
    return payload


def current() -> dict:
    """The latest checked list, if it's recent enough to show."""
    got = db.cached_search(db.cache_key(KEY, market.current().code), MAX_AGE_HOURS * 3600)
    return got or {"checked_at": None, "deals": []}


def category_name(key: str) -> str:
    if key == ACCESSORY:
        return "إكسسوارات"
    return catalog.CATEGORIES.get(key, {}).get("ar", key)


def _attrs(d: dict) -> dict:
    """What the price list's title reading (catalog.understand, done once per product) knows about it: brand, kind."""
    try:
        got = json.loads(d.get("attrs") or "{}")
    except ValueError:
        return {}
    return got if isinstance(got, dict) else {}


# The filters inside a category (brand, kind of product) and the search box of the deals page: all from what we
# already know about each product (its title, and the brand/kind read once when it entered the price list), with no
# LLM call when the page is shown.
_NOT_BRANDS = {"the", "new", "original", "set", "pack", "for", "with", "and", "men", "women", "kids", "baby", "mini",
               "pro", "max", "plus", "smart", "portable", "wireless", "digital", "electric", "classic", "premium"}
_KIND_AR = {k: next((w for w in words if re.search(r"[\u0600-\u06ff]", w)), k) for k, words in query.KIND_WORDS.items()}
_KIND_AR.update({"tv": "شاشات", "refrigerator": "تلاجات", "washing machine": "غسالات", "phone": "موبايلات",
                 "laptop": "لابتوبات", "earbuds": "سماعات", "watch": "ساعات", "perfume": "برفانات",
                 "body splash": "بودي سبلاش", "cream": "كريمات ولوشن", "air conditioner": "تكييفات"})

_LINES = {"iphone": "Apple", "redmi": "Xiaomi", "poco": "Xiaomi", "soundcore": "Anker", "lg": "LG", "hp": "HP",
          "tcl": "TCL", "jbl": "JBL", "msi": "MSI", "ysl": "YSL", "loreal": "L'Oreal", "oneplus": "OnePlus"}


def _first_word(title: str) -> str:
    w = (title or "").split()[:1]
    w = re.sub(r"[^\w&'-]", "", w[0]).strip("-'") if w else ""
    return w if len(w) >= 2 and re.fullmatch(r"[A-Za-z][A-Za-z&'-]*", w) and w.lower() not in _NOT_BRANDS else ""


def facets(rows: list[dict]) -> list[dict]:
    """Each deal's brand and kind of product for the page's filters: [{"brand", "brand_name", "kind", "kind_name"}].
    The brand: the one read with the title, else a brand we know named in the title, else the title's first word
    when other deals start with it too ("Fabyo Men Winter Jacket", "Fabyo Kids Jacket")."""
    firsts: dict[str, int] = {}
    for d in rows:
        if w := _first_word(d.get("title", "")).lower():
            firsts[w] = firsts.get(w, 0) + 1
    out = []
    for d in rows:
        title = d.get("title") or ""
        brand = str(d.get("brand") or "").strip()
        if not brand:
            named = [b for b in query.BRANDS if query.names_brand(title, b)]
            low = query._folded(title)
            named.sort(key=lambda b: (low.find(b.replace("-", " ")) % 10_000, b))  # the one written first
            brand = _LINES.get(named[0], named[0].replace("-", " ").title()) if named else ""
        if not brand and firsts.get((w := _first_word(title)).lower(), 0) >= 2:
            brand = w if not w.isupper() or len(w) <= 4 else w.title()
        kinds = sorted(query.kinds_in(title))
        kind_name = _KIND_AR.get(kinds[0], "") if len(kinds) == 1 else ""
        kind_name = kind_name or str(d.get("kind_ar") or "").strip()
        out.append({"brand": _key(brand), "brand_name": brand[:30], "kind": _key(kind_name), "kind_name": kind_name[:30]})
    return out


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", query._folded(text)).strip()


_SYNONYMS: dict[str, set[str]] = {}
for _ar, _en in list(query.AR_WORDS.items()) + list(query.AR_PHRASES.items()):
    if not re.search(r"[\u0600-\u06ff]", _en):  # an Arabic word -> the English word store titles use
        _SYNONYMS.setdefault(query._folded(_en).replace("-", " "), set()).add(query._folded(_ar))
for _en, _aliases in query.BRAND_ALIASES.items():
    for _a in _aliases:
        _SYNONYMS.setdefault(query._folded(_a), set()).update(_SYNONYMS.get(query._folded(_en).replace("-", " "), ()))

# Arabic kind words that are really a brand's product line ("ايفون" is a phone, but only Apple's)
_NAMES = {query._folded(ar) for ar, en in query.AR_WORDS.items() if en in query.BRANDS | {"iphone", "ipad", "airpods"}}


def search_text(d: dict, facet: dict, category: str) -> str:
    """The words the search box matches a deal by, folded like the box folds what's typed (أ->ا, ة->ه): its title,
    brand, kind and category, and the Arabic words shoppers write for the English words of its title ("سامسونج" finds
    "Samsung Galaxy A56", "غسالة" finds "Washing Machine")."""
    title = query._folded(d.get("title") or "")
    words = re.findall(r"[a-z0-9\u0600-\u06ff]+", title.replace("'", ""))
    grams = set(words) | {" ".join(words[i:i + 2]) for i in range(len(words) - 1)}
    extra = {ar for g in grams for ar in _SYNONYMS.get(g, ())}
    extra |= {query._folded(w) for k in query.kinds_in(title) for w in query.KIND_WORDS[k]
              if re.search(r"[\u0600-\u06ff]", w) and query._folded(w) not in _NAMES}
    parts = [title, facet["brand"], facet["kind"], query._folded(category), " ".join(sorted(extra))]
    return " ".join(p for p in parts if p)[:600]
