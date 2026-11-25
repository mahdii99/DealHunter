"""One page per product ("سعر Galaxy A56 256GB في مصر"), for shoppers who start on Google.

A product here is what the LLM read from the store titles in our price list (catalog.understand): the same brand,
line and size, whichever store sells it. Its page lists the offers of every store we read, cheapest first.

Prices follow the same rule as everywhere on the site: an offer is listed with a price only after its own store page
was opened and showed that price (agent.checked_only), and the page says when that was. The pages are made after each
price list run (POST /api/cron/pages from .github/workflows/scrape-prices.yml). An offer whose check is older than
SHOW_HOURS is no longer shown; a page left with no checked offer stays reachable (its address doesn't break) but
tells search engines not to list it until a new check, and it's left out of the sitemap.
"""
import json
import logging
import re
import time
from collections import Counter

from . import agent, catalog, db, market
from .models import Product

log = logging.getLogger("uvicorn.error")

SHOW_HOURS = 26          # an offer checked longer ago than this isn't shown (the list is made twice a day)
MAX_PAGES = 50           # products checked per run (each offer is one store page, read with no LLM call)
OFFERS_PER_PAGE = 6      # offers opened per product, cheapest first, at most PER_STORE per store
PER_STORE = 2
BATCH = 25               # pages opened together (agent.checked_only's batch, as in deals.build)
OUTLIER = 0.5            # an offer under half the product's middle price is something else (a part, a case...)
CATEGORY_ORDER = ["phones", "tvs", "washers", "fridges", "laptops", "acs", "tablets", "headphones", "smartwatches"]


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _size(text: str) -> str:
    """'256 GB' / '256gb' -> '256gb', '55 Inch' -> '55 inch': the same size however a store writes it."""
    t = _norm(text)
    return re.sub(r"(\d)\s+(gb|tb|mb|kg|l|w|hp)\b", r"\1\2", t)


def product_key(attrs: dict) -> tuple[str, str] | None:
    """(address, name) of the product a store title is, from what the LLM read in it; None when the title doesn't say
    enough to tell one product from another (no brand or line), or it's an accessory."""
    if not isinstance(attrs, dict) or attrs.get("accessory"):
        return None
    brand, line, size = (str(attrs.get(k) or "").strip() for k in ("brand", "line", "size"))
    if not brand or not line:
        return None
    if _norm(line).startswith(_norm(brand) + " "):  # "Samsung Galaxy A56" given as the line too
        line = line[len(brand):].strip()
    parts = [_norm(brand), _norm(line)] + ([_size(size)] if size else [])
    slug = re.sub(r"[^a-z0-9]+", "-", " ".join(parts)).strip("-")
    if len(slug) < 3 or not re.search(r"[a-z]", slug):  # an Arabic-only name: no clean address for it
        return None
    name = " ".join(x for x in (brand, line, size) if x)
    return slug[:90], name[:120]


def groups(rows: list[dict]) -> list[dict]:
    """The price list's products grouped by what they are, the ones most stores sell first."""
    by: dict[str, dict] = {}
    for r in rows:
        try:
            attrs = json.loads(r.get("attrs") or "{}")
        except ValueError:
            continue
        key = product_key(attrs)
        if not key:
            continue
        g = by.setdefault(key[0], {"slug": key[0], "names": Counter(), "category": Counter(), "rows": [],
                                   "brand": attrs.get("brand")})
        g["names"][key[1]] += 1
        g["category"][r.get("category") or ""] += 1
        g["rows"].append(r)
    out = []
    for g in by.values():
        prices = sorted(r["price"] for r in g["rows"])
        mid = prices[len(prices) // 2]
        rows = sorted((r for r in g["rows"] if r["price"] >= mid * OUTLIER), key=lambda r: r["price"])
        per_store: Counter = Counter()
        picked = []
        for r in rows:
            if per_store[r["store"]] < PER_STORE and len(picked) < OFFERS_PER_PAGE:
                per_store[r["store"]] += 1
                picked.append(r)
        cat = g["category"].most_common(1)[0][0]
        out.append({"slug": g["slug"], "name": g["names"].most_common(1)[0][0], "brand": g["brand"], "category": cat,
                    "rows": picked, "stores": len(per_store)})
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    out.sort(key=lambda g: (-min(g["stores"], 3), rank.get(g["category"], len(rank)), -len(g["rows"]), g["slug"]))
    return out


async def build(limit: int = MAX_PAGES) -> dict:
    """Opens the offers of the top products on their store pages and saves each product's page with the offers whose
    page showed a price just now."""
    mk = market.current().code
    rows = db.catalog_since(mk, time.time() - SHOW_HOURS * 3600)
    todo = groups(rows)[:limit]
    picks, owner = [], {}
    for g in todo:
        for r in g["rows"]:
            owner[r["url"]] = g["slug"]
            picks.append(Product(store=r["store"], title=r["title"], price=r["price"], url=r["url"],
                                 image=r["image"] or None, old_price=r.get("old_price"), trusted=True, trust_score=100))
    shown, gone = [], 0
    for i in range(0, len(picks), BATCH):  # a few at a time: each batch gets the agent's full time to check its pages
        got = await agent.checked_only(picks[i:i + BATCH], patient=True, use_llm=False)
        shown += got[0]
        gone += got[2]
    now = time.time()
    checked: dict[str, list[dict]] = {}
    for p in shown:
        checked.setdefault(owner[p.url], []).append(
            {"title": p.title, "store": p.store, "url": p.url, "price": p.price, "old_price": p.old_price,
             "image": p.image or "", "checked_at": now})
    made = 0
    for g in todo:
        offers = sorted(checked.get(g["slug"], []), key=lambda o: o["price"])
        db.save_product_page(g["slug"], mk, g["name"], g["category"], g.get("brand") or "",
                             [r["url"] for r in g["rows"]], offers, now)
        made += bool(offers)
    log.info("product pages: %d of %d products with checked offers (%d offers, %d sold out)",
             made, len(todo), len(shown), gone)
    return {"products": len(todo), "with_offers": made, "offers": len(shown), "sold_out": gone, "candidates": len(picks)}


def fresh_offers(page: dict, now: float | None = None) -> list[dict]:
    """The page's offers whose check is recent enough to show their price."""
    since = (now or time.time()) - SHOW_HOURS * 3600
    return [o for o in page.get("offers") or [] if (o.get("checked_at") or 0) >= since]


def get(slug: str) -> dict | None:
    return db.product_page(slug, market.current().code)


def listed() -> list[dict]:
    """Every product page with an offer checked recently enough to show (the sitemap and the /p list)."""
    since = time.time() - SHOW_HOURS * 3600
    return [p for p in db.product_pages(market.current().code, since) if fresh_offers(p)]


def category_name(key: str) -> str:
    return catalog.CATEGORIES.get(key, {}).get("ar", key)
