"""Today's best deals: the products of our price list (app/catalog.py) that really got cheaper, checked on their own
store page just before they're listed.

A shopper trusts a deal only when it's real. Stores often show a high "old price" that the product never sold at, so
what counts most here is our own history: a price below what the same store page asked for it in the last 30 days.
A store's own discount is listed after those, and says it's the store's number, not ours.

Prices on the page are never older than their check: every deal was opened on its store page (the same check as a
search result, agent.checked_only) when the list was made, and the page shows when that was. The list is made
after each price list run (.github/workflows/scrape-prices.yml) and stops being shown after MAX_AGE_HOURS.
"""
import logging
import time

from . import agent, catalog, db, market
from .models import Product

log = logging.getLogger("uvicorn.error")

KEY = "__deals__"           # the list is kept in `searches` under this request
MAX_AGE_HOURS = 14          # an older list isn't shown (prices move)
HISTORY_DAYS = 30
MIN_DROP = 5                # % below the lowest price seen before today, to count as a real drop
MIN_STORE_DISCOUNT = 15     # % off the store's own old price, for the store-discount part
CHECK = 120                 # candidates opened per run (each one is a store page, some an LLM read)
BATCH = 20                  # opened together (agent.checked_only checks at most VERIFY_MAX pages per call)
PER_CATEGORY = 30
PER_STORE = 70


def candidates(now: float | None = None) -> list[dict]:
    """Price list products seen in the last day, with how their price moved, best deals first."""
    now = now or time.time()
    mk = market.current().code
    rows = db.catalog_since(mk, now - 26 * 3600)
    if not rows:
        return []
    rows = _accessories_apart(rows)
    past = db.prices_between([r["url"] for r in rows], now - HISTORY_DAYS * 86400, now - 20 * 3600)
    out = []
    for r in rows:
        before = past.get(r["url"]) or []
        low = min(before) if before else None
        drop = round((low - r["price"]) / low * 100) if low and low > r["price"] else 0
        off = (round((r["old_price"] - r["price"]) / r["old_price"] * 100)
               if r.get("old_price") and r["old_price"] > r["price"] else 0)
        if drop >= MIN_DROP and drop <= 70:
            out.append({**r, "kind": "drop", "was": low, "pct": drop})
        elif MIN_STORE_DISCOUNT <= off <= 70:
            out.append({**r, "kind": "store", "was": r["old_price"], "pct": off})
    out.sort(key=lambda d: (d["kind"] != "drop", -d["pct"], -(d["was"] - d["price"])))
    picked, per_cat, per_store = [], {}, {}
    for d in out:
        if per_cat.get(d["category"], 0) >= PER_CATEGORY or per_store.get(d["store"], 0) >= PER_STORE:
            continue
        per_cat[d["category"]] = per_cat.get(d["category"], 0) + 1
        per_store[d["store"]] = per_store.get(d["store"], 0) + 1
        picked.append(d)
    return picked


ACCESSORY = "accessories"
ACCESSORY_SHARE = 0.15      # far under the usual price of its category: a case, a bag, a stand, not the product


def _accessories_apart(rows: list[dict]) -> list[dict]:
    """A store's search for "washing machine" also lists washer bags and covers. Shoppers browsing washers don't want
    those, so anything priced far under its category's middle price goes under accessories instead."""
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["price"])
    mid = {c: sorted(v)[len(v) // 2] for c, v in by_cat.items() if len(v) >= 10}
    return [{**r, "category": ACCESSORY} if r["category"] in mid and r["price"] < mid[r["category"]] * ACCESSORY_SHARE
            else r for r in rows]


async def build() -> dict:
    """Opens the best candidates on their store pages and keeps those still at (or under) that price."""
    found = candidates()[:CHECK]
    picks = [Product(store=d["store"], title=d["title"], price=d["price"], url=d["url"], image=d["image"] or None,
                     old_price=d.get("old_price"), trusted=True, trust_score=100) for d in found]
    by_url = {d["url"]: d for d in found}
    shown, unchecked, gone = [], [], 0
    for i in range(0, len(picks), BATCH):  # a few at a time: each batch gets the agent's full time to check its pages
        got = await agent.checked_only(picks[i:i + BATCH])
        shown += got[0]
        unchecked += got[1]
        gone += got[2]
    deals = []
    for p in shown:
        d = by_url[p.url]
        if p.price > d["price"] + 1:  # dearer than when we listed it: not a deal any more
            continue
        was = d["was"]
        deals.append({"title": p.title, "store": p.store, "url": p.url, "image": p.image or d["image"] or None,
                      "price": p.price, "was": was, "pct": round((was - p.price) / was * 100), "kind": d["kind"],
                      "category": d["category"]})
    deals.sort(key=lambda d: (d["kind"] != "drop", -d["pct"]))
    payload = {"checked_at": time.time(), "deals": deals, "candidates": len(found), "sold_out": gone,
               "unchecked": len(unchecked)}
    db.save_search(db.cache_key(KEY, market.current().code), market.current().code, payload)
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
