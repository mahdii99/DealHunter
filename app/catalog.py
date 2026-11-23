"""Our own price list: the big stores' category pages, read on a schedule (once or twice a day).

Why: most shoppers ask for the same popular products (phones first, then TVs and big appliances, then laptops).
Reading those categories ahead of time means a search already knows the offers of the biggest stores before it
spends any of the free web-search quota, and every price we read builds the price history.

How it reads a page: the same store-agnostic extraction the agent uses everywhere (product JSON / JSON-LD in the
page, app/extract.py), plus the product cards of pages without such data (Amazon's, app/stores/amazon.py, and
Jumia's). Whichever reads the most products wins, so a store that changes its page doesn't need new code here.

Rules we keep:
  - robots.txt is read first and obeyed, for every page.
  - an honest crawler name with the site's address (Jumia's robots.txt asks for exactly that), one page at a time
    per store with a pause between pages: far below what any store allows.
  - a store that blocks us is reported in the run's answer and skipped; we never try to get around it.
  - a price from here is never shown as checked: the agent uses these products as candidates, and every result
    is opened again on its own page before it's shown (agent._verify_stock).
"""
import asyncio
import logging
import os
import re
import time
from urllib.parse import urlparse

import httpx

from . import db, fetchers, market, trust
from .extract import extract_products
from .models import Product
from .stores.amazon import parse_amazon_cards
from .stores.jumia import Jumia

log = logging.getLogger("uvicorn.error")

SITE = os.getenv("SITE_URL", "https://dealhunter-opk9.onrender.com").rstrip("/")
BOT_HEADERS = {
    "User-Agent": f"DealHunterBot/1.0 (+{SITE}/bot)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}
PAUSE = float(os.getenv("CATALOG_PAUSE", "4"))  # seconds between two pages of the same store
PAGE_TIMEOUT = 40
FRESH_HOURS = 36  # a catalog price older than this isn't used in searches

# The category pages read in each store, best sellers / most popular first. {n} = page number (1, 2, ...).
# Only addresses: how a page is read is the same for every store (see read_listing).
CATEGORIES: dict[str, dict] = {
    "phones": {
        "ar": "موبايلات", "pages": 5,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/mobile-phones/?page={n}",
            "noon.com": "https://www.noon.com/egypt-en/electronics-and-mobiles/mobiles-and-accessories/"
                        "mobiles-20905/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?k=smartphone&i=electronics&language=en_AE&page={n}",
        },
    },
}
STORES = sorted({d for c in CATEGORIES.values() for d in c["stores"]})


class NothingRead(Exception):
    """The page opened but no product was read from it (the last page of a category, or a changed page)."""


class Blocked(Exception):
    """The store refused the page (robots.txt, a 403/503 or a robot check): skip this store for this run."""


def read_listing(html: str, url: str) -> list[Product]:
    """Every product a category page lists, with the reader that finds the most of them."""
    store = trust.store_name(url)
    tries = [extract_products(html, store, url), parse_amazon_cards(html, store, url)]
    jumia = Jumia()
    jumia.name, jumia.base_url = store, f"https://{urlparse(url).netloc}"
    tries.append(jumia.parse(html))
    best = max(tries, key=len)
    host = trust.domain(url)
    out = []
    for p in best:  # only this store's own product pages, new, for sale, with a real price
        if trust.domain(p.url) != host or urlparse(p.url).path.rstrip("/") == urlparse(url).path.rstrip("/"):
            continue
        if p.in_stock is False or p.condition or p.price < 5:
            continue
        p.store = store
        out.append(p)
    return out


def _robot_check(html: str) -> bool:
    low = html[:20000].lower()
    return any(w in low for w in ("captcha", "robot check", "are you a human", "access denied", "verify you are human"))


async def _page(client: httpx.AsyncClient, url: str) -> list[Product]:
    if not await fetchers.robots_allows(client, url):
        raise Blocked(f"robots.txt: {fetchers.robots_reason(url)}")
    r = await client.get(url, headers=BOT_HEADERS, follow_redirects=True, timeout=PAGE_TIMEOUT)
    if r.status_code in (401, 403, 429, 503):
        raise Blocked(f"HTTP {r.status_code}")
    r.raise_for_status()
    products = read_listing(r.text, str(r.url))
    if not products and _robot_check(r.text):
        raise Blocked("robot check page")
    if not products:  # say what the page was, so a changed or refused page can be told from an empty one
        m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
        raise NothingRead(f"no products read ({len(r.text)} chars, title: {' '.join((m.group(1) if m else '-').split())[:80]})")
    return products


async def scrape(stores: list[str] | None = None, categories: list[str] | None = None,
                 pages: int | None = None, client: httpx.AsyncClient | None = None) -> dict:
    """Reads the category pages and saves what they list. The stores run side by side; each store's pages one
    after another with a pause. Returns what happened, per store and category (for the run's log)."""
    mk = market.current().code
    report: dict[str, dict] = {}
    own = client is None
    client = client or httpx.AsyncClient()

    async def one_store(d: str):
        for cat, conf in CATEGORIES.items():
            if (categories and cat not in categories) or d not in conf["stores"]:
                continue
            t, got, err, seen = time.monotonic(), 0, None, set()
            for n in range(1, (pages or conf["pages"]) + 1):
                if n > 1:
                    await asyncio.sleep(PAUSE)
                url = conf["stores"][d].format(n=n)
                try:
                    products = await _page(client, url)
                except Blocked as e:
                    err = str(e)
                    break
                except NothingRead as e:
                    if n == 1:  # past the last page is normal; an empty first page is worth a look
                        err = str(e)
                    break
                except Exception as e:  # a slow page: the next run tries again
                    err = f"{type(e).__name__}: {e}"[:200]
                    break
                new = [p for p in products if p.url not in seen]
                if not new:  # past the last page (some stores show page 1 again)
                    break
                seen.update(p.url for p in new)
                rows = [{"url": p.url, "store": p.store, "title": p.title, "price": p.price,
                         "old_price": p.old_price, "image": p.image, "in_stock": True} for p in new]
                got += await asyncio.to_thread(db.save_catalog, rows, mk, cat)
            report[f"{d} {cat}"] = {"products": got, "seconds": round(time.monotonic() - t, 1),
                                    **({"error": err} if err else {})}
            log.info("catalog: %s %s: %d products%s", d, cat, got, f" ({err})" if err else "")

    try:
        await asyncio.gather(*(one_store(d) for d in (stores or STORES) if d in STORES))
    finally:
        if own:
            await client.aclose()
    return report


def offers(words: list[str], max_age_hours: float = FRESH_HOURS) -> list[Product]:
    """Catalog products whose title has any of these words, seen in the last hours: candidates for a search
    (still checked on their own page before they're shown)."""
    rows = db.catalog_like(words, market.current().code, time.time() - max_age_hours * 3600)
    out = []
    for r in rows:
        p = Product(store=r["store"], title=r["title"], price=r["price"], url=r["url"], image=r["image"] or None,
                    old_price=r["old_price"])
        p.trusted, p.trust_score, p.trust_signals = True, 100, ["متجر مشهور"]
        out.append(p)
    return out
