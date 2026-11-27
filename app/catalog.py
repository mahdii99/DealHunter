"""Our own price list: the big stores' category pages, read on a schedule (twice a day).

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
import json
import logging
import os
import re
import time
from urllib.parse import urlparse

import httpx

from . import db, fetchers, llm, market, trust
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
MIN_TRUST = 35  # a shop a search found goes into the index only with at least this trust score (agent shows from 30)
FRESH_HOURS = 36  # a catalog price older than this isn't used in searches

# The category pages read in each store, best sellers / most popular first. {n} = page number (1, 2, ...).
# Only addresses: how a page is read is the same for every store (see read_listing).
CATEGORIES: dict[str, dict] = {
    "phones": {
        "ar": "موبايلات", "pages": 8,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/mobile-phones/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?k=smartphone&i=electronics&language=en_AE&page={n}",
        },
    },
    "tvs": {
        "ar": "شاشات", "pages": 6,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/televisions/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?k=smart+tv&i=electronics&language=en_AE&page={n}",
        },
    },
    "washers": {
        "ar": "غسالات", "pages": 6,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/appliances-washers-dryers/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?k=washing+machine&i=appliances&language=en_AE&page={n}",
        },
    },
    "fridges": {
        "ar": "تلاجات", "pages": 6,
        "stores": {
            "amazon.eg": "https://www.amazon.eg/s?k=refrigerator&i=appliances&language=en_AE&page={n}",
        },
    },
    "acs": {
        "ar": "تكييفات", "pages": 6,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/air-conditioning/?page={n}",
        },
    },
    "laptops": {
        "ar": "لابتوبات", "pages": 6,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/laptops/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?k=laptop&i=computers&language=en_AE&page={n}",
        },
    },
    # more kinds shoppers often ask for (Amazon's own search pages; robots.txt is read before each page)
    "tablets": {"ar": "تابلت", "pages": 4, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=tablet&i=electronics&language=en_AE&page={n}"}},
    "headphones": {"ar": "سماعات", "pages": 4, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=wireless+earbuds+headphones&i=electronics&language=en_AE&page={n}"}},
    "smartwatches": {"ar": "ساعات ذكية", "pages": 3, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=smart+watch&i=electronics&language=en_AE&page={n}"}},
    "microwaves": {"ar": "ميكروويف", "pages": 3, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=microwave+oven&i=appliances&language=en_AE&page={n}"}},
    "heaters": {"ar": "سخانات", "pages": 3, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=water+heater&i=appliances&language=en_AE&page={n}"}},
    "cookers": {"ar": "بوتاجازات", "pages": 3, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=gas+cooker&i=appliances&language=en_AE&page={n}"}},
    "vacuums": {"ar": "مكانس", "pages": 3, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=vacuum+cleaner&i=appliances&language=en_AE&page={n}"}},
    "airfryers": {"ar": "قلايات هوائية", "pages": 3, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=air+fryer&i=kitchen&language=en_AE&page={n}"}},
    # beauty and personal care (many shoppers, women most of all, buy these online): one kind of product each, so a
    # search finds them in our own list first. "mixed": very different prices in one kind (a lipstick and a palette),
    # so a cheap one is not set apart as an accessory on the deals page.
    "perfumes": {"ar": "برفانات", "pages": 5, "mixed": True, "stores": {
        "jumia.com.eg": "https://www.jumia.com.eg/fragrances-allgenders/?page={n}",
        "amazon.eg": "https://www.amazon.eg/s?k=perfume&i=beauty&language=en_AE&page={n}"}},
    "makeup": {"ar": "ميكب", "pages": 5, "mixed": True, "stores": {
        "jumia.com.eg": "https://www.jumia.com.eg/womens-make-up/?page={n}",
        "amazon.eg": "https://www.amazon.eg/s?k=makeup&i=beauty&language=en_AE&page={n}"}},
    "skincare": {"ar": "عناية بالبشرة", "pages": 4, "mixed": True, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=skin+care&i=beauty&language=en_AE&page={n}"}},
    "haircare": {"ar": "عناية بالشعر", "pages": 3, "mixed": True, "stores": {
        "jumia.com.eg": "https://www.jumia.com.eg/hair-conditioner-d/?page={n}",
        "amazon.eg": "https://www.amazon.eg/s?k=shampoo&i=beauty&language=en_AE&page={n}"}},
    "hair_tools": {"ar": "أجهزة الشعر", "pages": 3, "mixed": True, "stores": {
        "amazon.eg": "https://www.amazon.eg/s?k=hair+dryer+straightener&i=beauty&language=en_AE&page={n}"}},
    "deodorants": {"ar": "مزيلات عرق", "pages": 2, "mixed": True, "stores": {
        "jumia.com.eg": "https://www.jumia.com.eg/styling-products-antiperspirants-deodorants/?page={n}"}},
    # broad sections for the deals page (shoppers browse those like a store's offers tab). Amazon's own "30% off and
    # more" filter (rh=p_8) gives discounted products only; Jumia's robots.txt forbids its filters and sorting, so its
    # plain section pages are read, plus its flash sales page.
    "home": {
        "ar": "البيت والمطبخ", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/home-office/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=kitchen&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "beauty": {
        "ar": "جمال وعناية", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/health-beauty/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=beauty&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "fashion": {
        "ar": "موضة", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/category-fashion-by-jumia/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=fashion&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "electronics": {
        "ar": "إلكترونيات", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/electronics/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=electronics&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "computing": {
        "ar": "كمبيوتر وإكسسواراته", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/computing/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=computers&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "mobile_acc": {
        "ar": "إكسسوارات موبايل", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/mobile-accessories/?page={n}",
        },
    },
    "toys": {
        "ar": "ألعاب أطفال", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/toys-games/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=toys&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "baby": {
        "ar": "مستلزمات البيبي", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/baby-products/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=baby&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "sports": {
        "ar": "رياضة", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/sporting-goods/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=sporting&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "grocery": {
        "ar": "سوبر ماركت", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/groceries/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=grocery&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "gaming": {
        "ar": "ألعاب فيديو", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/video-games/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=videogames&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "auto": {
        "ar": "مستلزمات العربيات", "pages": 3, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/automobile/?page={n}",
            "amazon.eg": "https://www.amazon.eg/s?i=automotive&rh=p_8%3A30-&language=en_AE&page={n}",
        },
    },
    "flash": {
        "ar": "عروض لفترة محدودة", "pages": 5, "broad": True,
        "stores": {
            "jumia.com.eg": "https://www.jumia.com.eg/flash-sales/?page={n}",
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


FOUND_DAYS = 7  # an offer a search found (and checked) stays a candidate this long: it is checked again before shown


def offers(words: list[str], max_age_hours: float = FRESH_HOURS) -> list[Product]:
    """Catalog products whose title (or the words the LLM gave them) has any of these words, seen in the last hours:
    candidates for a search (still checked on their own page before they're shown). Besides the big stores' category
    pages, the index holds every offer a search found on the web and checked on its page (smaller shops too)."""
    now = time.time()
    rows = db.catalog_like(words, market.current().code, now - max_age_hours * 3600, found_since=now - FOUND_DAYS * 86400)
    out = []
    for r in rows:
        p = Product(store=r["store"], title=r["title"], price=r["price"], url=r["url"], image=r["image"] or None,
                    old_price=r["old_price"])
        if r.get("trust") is None and r.get("category") != db.FOUND:  # read from a big store's own category pages
            p.trusted, p.trust_score, p.trust_signals = True, 100, ["متجر مشهور"]
        else:
            p.trusted, p.trust_score = trust.is_known(r["url"]), r.get("trust")
        out.append(p)
    return out


def remember_found(results: list[dict]) -> int:
    """Offers a search just checked on their own page go into the index, so the next shopper asking for them gets
    them from the index first. Only offers whose price was confirmed there, from shops trusted enough to be shown."""
    rows = [{"url": r["url"], "store": r.get("store") or trust.store_name(r["url"]), "title": r.get("title") or "",
             "price": r.get("price"), "old_price": r.get("old_price"), "image": r.get("image") or "",
             "trust": 100 if r.get("trusted") else r.get("trust_score")}
            for r in results if r.get("url") and r.get("verified") and not r.get("condition") and not r.get("option")
            and isinstance(r.get("price"), (int, float)) and (r.get("trusted") or (r.get("trust_score") or 0) >= MIN_TRUST)]
    return db.save_catalog(rows, market.current().code, db.FOUND, history=False) if rows else 0


UNDERSTAND_BATCH = 15     # titles per LLM call (40 was too long an answer for the small models: most came back cut)
UNDERSTAND_MAX = 450      # titles per run (30 light calls): the free quota is shared with shoppers' searches
UNDERSTAND_PROMPT = """These are product titles from online stores in Egypt. For each one, say what the product is, so
shoppers find it whatever words they use (Arabic or English, the brand or just the kind of product).
{items}
Reply JSON: {{"items": [{{"i": <number>, "brand": "brand or null", "line": "product line/name without the brand, e.g.
'Galaxy A56', or null", "model": "model / part number exactly as written, or null", "kind": "what it is in plain
English, e.g. 'washing machine', 'smartphone', 'phone case', 'perfume', 'lipstick'", "kind_ar": "the same in the
Arabic Egyptian shoppers use, e.g. 'غسالة', 'موبايل', 'جراب موبايل', 'برفان', 'روج'", "size": "its main
size/capacity/storage, e.g. '8 kg', '256GB', '55 inch', '100 ml', '50 g', or null", "specs": ["up to 5 short English
specs that matter when choosing, e.g. 'top load', 'automatic', '5G', 'inverter', 'eau de parfum', 'matte', 'oily skin',
'for women', a shade name or number"], "accessory": true if it is an accessory/part for another product (case, cover, cable, filter)}}]}}
One entry per title, same numbers. Use only what the title says or what the model number surely means."""


def _words(attrs: dict) -> str:
    """The words a search finds a product by: everything the LLM said about it, lower case, once each."""
    bits = [attrs.get(k) for k in ("brand", "line", "model", "kind", "kind_ar", "size")] + list(attrs.get("specs") or [])
    seen: dict[str, None] = {}
    for b in bits:
        if isinstance(b, str):
            for w in b.lower().split():
                seen.setdefault(w, None)
    return " ".join(seen)[:1000]


async def understand(limit: int = UNDERSTAND_MAX) -> dict:
    """The LLM reads the titles of new index products (light models first, at quiet times: a cron job) and saves
    brand, line, model, kind (English and Arabic), size and specs, so a search can find a product by any of them."""
    mk = market.current().code
    todo = await asyncio.to_thread(db.catalog_to_understand, mk, limit)
    done, failed = 0, 0
    for start in range(0, len(todo), UNDERSTAND_BATCH):
        part = todo[start:start + UNDERSTAND_BATCH]
        items = "\n".join(f"{i}. {r['title'][:160]}" for i, r in enumerate(part))
        try:
            got = await llm.ask_json(UNDERSTAND_PROMPT.format(items=items), timeout=30, budget=60, prefer="bulk",
                                     spread=start // UNDERSTAND_BATCH, step="understand")
        except llm.LLMError as e:  # the free quota is out for now: the next run goes on from here
            log.warning("catalog: understanding titles stopped: %s", e.details or e)
            failed += len(part)
            break
        rows = []
        for a in got.get("items") or []:
            i = a.get("i") if isinstance(a, dict) else None
            if not isinstance(i, int) or not 0 <= i < len(part):
                continue
            attrs = {k: a.get(k) for k in ("brand", "line", "model", "kind", "kind_ar", "size")
                     if isinstance(a.get(k), str) and a.get(k).strip()}
            attrs["specs"] = [s for s in a.get("specs") or [] if isinstance(s, str)][:5]
            attrs["accessory"] = a.get("accessory") is True
            rows.append((part[i]["url"], json.dumps(attrs, ensure_ascii=False), _words(attrs)))
        missed = len(part) - len(rows)
        got_urls = {u for u, _, _ in rows}
        if len(rows) * 2 >= len(part):
            # a title the LLM skipped in an answer that read the others isn't asked again and again (it is still
            # found by its own words); a cut-off answer leaves the rest for the next run
            rows += [(r["url"], "{}", "") for r in part if r["url"] not in got_urls]
        await asyncio.to_thread(db.save_understood, rows)
        done += len(rows) - missed
        failed += missed
    return {"understood": done, "not_understood": failed, "asked": len(todo)}
