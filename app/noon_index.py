"""noon's own public product sitemaps: the names and addresses of every product noon sells in Egypt.

Why: noon's category pages answer crawlers with a bot check (Akamai), so our price list (app/catalog.py) can't read
them, and we never try to get around such a check. But noon publishes sitemaps for search engines (listed in its
robots.txt), and its robots.txt allows product pages for every crawler. A product page carries the price as
structured data (JSON-LD Product / Offer).

A probe from our server (2026-11-25) showed noon's own site never answers our crawler, product pages included, while
the sitemap files on noon's public storage open fine. So we can't read a noon price, and a price we didn't read on
the store's own page is never shown. What we can do: keep the names and addresses from the sitemaps (once a week),
and when a shopper searches, show the few noon products that fit as a plain "see the price on noon" link, without
any price. Clicks on those links are counted (/go), which is the traffic noon's affiliate program asks for.

probe() is kept to check again later whether noon answers our crawler.
"""
import asyncio
import json
import logging
import re
import os
import time
from urllib.parse import quote

import httpx

from . import db, fetchers, market
from .catalog import BOT_HEADERS
from .extract import extract_products

log = logging.getLogger("uvicorn.error")

SITEMAP_INDEX = "https://noon-sitemap.storage.googleapis.com/sitemap-index.xml"
SITEMAP_HOST = "https://noon-sitemap.storage.googleapis.com/"  # the files the index lists on www.noon.com/sitemap/
EG_PDP = re.compile(r"sitemap-eg-pdp-\d+\.xml")
EN_PRODUCT = re.compile(r"<loc>(https://www\.noon\.com/egypt-en/([^/<]*)/([A-Z0-9]+)/p/)</loc>")


def _file_url(loc: str) -> str:
    """The copy of a sitemap file on noon's public storage (the index names them under www.noon.com/sitemap/)."""
    return SITEMAP_HOST + loc.rsplit("/", 1)[-1]


async def _timed(client: httpx.AsyncClient, url: str, timeout: float, limit: int | None = None) -> dict:
    t = time.monotonic()
    try:
        if limit:  # only the start of a big file
            async with client.stream("GET", url, headers=BOT_HEADERS, follow_redirects=True, timeout=timeout) as r:
                body = b""
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) >= limit:
                        break
                text = body.decode("utf-8", "ignore")
                status, final = r.status_code, str(r.url)
        else:
            r = await client.get(url, headers=BOT_HEADERS, follow_redirects=True, timeout=timeout)
            text, status, final = r.text, r.status_code, str(r.url)
    except Exception as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"[:200], "seconds": round(time.monotonic() - t, 1)}
    return {"url": url, "status": status, "final_url": final, "chars": len(text),
            "seconds": round(time.monotonic() - t, 1), "_text": text}


def _page_summary(got: dict, url: str) -> dict:
    text = got.pop("_text", "")
    if not text:
        return got
    low = text[:30000].lower()
    got["bot_check"] = any(w in low for w in ("akamai", "captcha", "access denied", "are you a human", "_abck"))
    got["json_ld"] = text.count("application/ld+json")
    try:
        products = extract_products(text, "noon", url)
    except Exception as e:
        products, got["read_error"] = [], str(e)[:120]
    main = next((p for p in products if p.url.rstrip("/").endswith(url.rstrip("/").rsplit("/p", 1)[0].rsplit("/", 1)[-1])
                 or p.url.rstrip("/") == url.rstrip("/")), products[0] if products else None)
    if main:
        got.update({"title": main.title[:90], "price": main.price, "old_price": main.old_price,
                    "in_stock": main.in_stock, "products_read": len(products)})
    m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
    got["page_title"] = " ".join((m.group(1) if m else "-").split())[:90]
    return got


async def probe(pages: int = 20, pause: float = 3.0) -> dict:
    """Opens the sitemap index, one Egypt product sitemap and a sample of its product pages with our crawler name,
    one page at a time, and says what each answered. Nothing is saved."""
    out: dict = {"user_agent": BOT_HEADERS["User-Agent"]}
    async with httpx.AsyncClient() as client:
        idx = await _timed(client, SITEMAP_INDEX, 60)
        text = idx.pop("_text", "")
        files = [m.group(0) for m in EG_PDP.finditer(text)]
        out["index"] = idx | {"eg_pdp_files": len(set(files))}
        if not files:
            return out
        # the same file from noon's own site, as the index lists it, and from the public storage copy
        own = await _timed(client, "https://www.noon.com/sitemap/sitemap-eg-pdp-20.xml", 40, limit=300_000)
        own.pop("_text", None)
        out["sitemap_on_noon_site"] = own
        sm = await _timed(client, SITEMAP_HOST + "sitemap-eg-pdp-20.xml", 60, limit=2_000_000)
        sm_text = sm.pop("_text", "")
        found = [(m.group(1), m.group(2)) for m in EN_PRODUCT.finditer(sm_text)]
        out["sitemap_file"] = sm | {"products_in_first_2mb": len(found)}
        named = [u for u, slug in found if slug and slug != "None"]
        step = max(1, len(named) // max(1, pages))
        sample = named[::step][:pages]
        out["robots_allows_product"] = bool(sample) and await fetchers.robots_allows(client, sample[0])
        if sample and not out["robots_allows_product"]:
            out["robots_reason"] = fetchers.robots_reason(sample[0])
            return out
        results = []
        for i, url in enumerate(sample):
            if i:
                await asyncio.sleep(pause)
            results.append(_page_summary(await _timed(client, url, 25), url))
        out["pages"] = results
        out["summary"] = {
            "opened": sum(1 for r in results if r.get("status") == 200),
            "with_price": sum(1 for r in results if r.get("price")),
            "bot_check": sum(1 for r in results if r.get("bot_check")),
            "errors": sum(1 for r in results if r.get("error")),
            "statuses": sorted({str(r.get("status") or r.get("error", "")[:30]) for r in results}),
        }
    log.info("noon probe: %s", json.dumps(out.get("summary") or out.get("index"), ensure_ascii=False))
    return out


# ---------- the index: names and addresses from the sitemaps, rebuilt once a week ----------

MARKETS = {"eg": "egypt"}  # our market -> the word noon uses in its sitemap files and addresses
PDP_FILES = 80             # sitemap-eg-pdp-0 ... -79 today; the index lists how many there are
DB_MAX_MB = float(os.getenv("NOON_DB_MAX_MB", "430"))  # stop filling the index before the free database is full
URL_BLOCK = re.compile(r"<url>(.*?)</url>", re.S)
IMG = re.compile(r"https://f\.nooncdn\.com/p/v(\d+)/([A-Z0-9]+)_1\.jpg")
SLUG_OK = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_SLUG = 110


def read_sitemap(text: str, country: str = "egypt") -> list[tuple[str, str, str, str]]:
    """(head, sku, slug, img) of every product in one sitemap file whose address has a name (some have none)."""
    loc = re.compile(rf"<loc>https://www\.noon\.com/{country}-en/([^/<]*)/([A-Z0-9]+)/p/</loc>")
    out = []
    for block in URL_BLOCK.finditer(text):
        b = block.group(1)
        m = loc.search(b)
        if not m:
            continue
        slug, sku = m.group(1)[:MAX_SLUG].strip("-"), m.group(2)
        if not slug or not SLUG_OK.match(slug):
            continue
        im = IMG.search(b)
        out.append((slug.split("-", 1)[0], sku, slug, im.group(1) if im and im.group(2) == sku else ""))
    return out


async def _sitemap_files(client: httpx.AsyncClient, country: str) -> list[int]:
    r = await client.get(SITEMAP_INDEX, headers=BOT_HEADERS, timeout=60)
    r.raise_for_status()
    return sorted({int(n) for n in re.findall(rf"sitemap-{country[:2]}-pdp-(\d+)\.xml", r.text)})


async def build(first: int = 0, last: int = PDP_FILES - 1, mk: str = "eg", client: httpx.AsyncClient | None = None) -> dict:
    """Reads sitemap files first..last (one at a time) into the index, each in place of what it had before."""
    mk = mk.lower()  # our market codes are "EG", "SA", ...
    country = MARKETS.get(mk)
    if not country:
        return {"error": f"no noon sitemaps for market {mk}"}
    own = client is None
    client = client or httpx.AsyncClient()
    report: dict = {"files": {}, "products": 0}
    t = time.monotonic()
    try:
        files = [n for n in await _sitemap_files(client, country) if first <= n <= last]
        for n in files:
            size = await asyncio.to_thread(db.size_mb)
            if size is not None and size > DB_MAX_MB:
                report["stopped"] = f"database is {size} MB (limit {DB_MAX_MB:.0f} MB)"
                break
            try:
                r = await client.get(f"{SITEMAP_HOST}sitemap-{country[:2]}-pdp-{n}.xml", headers=BOT_HEADERS, timeout=120)
                r.raise_for_status()
                rows = read_sitemap(r.text, country)
                del r
            except Exception as e:  # the next run reads it again; what the file had stays meanwhile
                report["files"][n] = f"{type(e).__name__}: {e}"[:120]
                continue
            await asyncio.to_thread(db.noon_replace_file, mk, n, rows)
            report["files"][n] = len(rows)
            report["products"] += len(rows)
        report["db_mb"] = await asyncio.to_thread(db.size_mb)
    finally:
        if own:
            await client.aclose()
    report["seconds"] = round(time.monotonic() - t, 1)
    log.info("noon index: %d products from %d files in %.0fs", report["products"], len(report["files"]), report["seconds"])
    return report


# ---------- at search time ----------

LINKS_MAX = 3
# words that make a product an accessory/part for another one: those aren't shown for a search of the product itself
ACCESSORY = {"case", "cases", "cover", "covers", "protector", "tempered", "film", "skin", "strap", "straps", "holder",
             "mount", "cable", "charger", "adapter", "replacement", "compatible", "sticker", "stickers", "pouch",
             "sleeve", "spare", "parts", "bracket", "tpu", "bumper"}
CONDITION = {"refurbished", "renewed", "used"}  # like everywhere on the site: new products only


def _tokens(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if w]


def _size_key(text: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", (text or "").lower().replace("inches", "inch").replace("gigabyte", "gb"))


def title(slug: str) -> str:
    """A readable name from noon's address words: model numbers in capitals, the rest with a capital first letter."""
    words = slug.split("-")[:14]
    return " ".join(w.upper() if any(c.isdigit() for c in w) and len(w) <= 8 else w.capitalize() for w in words)


def url(country: str, slug: str, sku: str) -> str:
    return f"https://www.noon.com/{country}-ar/{slug}/{sku}/p/"


def links(product: dict | None, codes: list[str] | None = None, mk: str | None = None) -> list[dict]:
    """The few noon products that fit what the shopper asked for (the planner's brand / model / size / kind), as
    links without a price. noon's names start with the brand ("samsung-galaxy-a56-...") or with the product line
    ("galaxy-a56-...", "iphone-16-..."), so both are looked up; every word of the model must be in the name, and the
    size when the shopper gave one. Parts, accessories and used products are left out unless the shopper wants an
    accessory. [] when the request names no brand (too vague to point anywhere)."""
    mk = (mk or market.current().code).lower()
    country = MARKETS.get(mk)
    p = product if isinstance(product, dict) else {}
    brand = _tokens(str(p.get("brand") or ""))
    if not country or not brand:
        return []
    kind = _tokens(str(p.get("kind") or ""))
    model = _tokens(str(p.get("model") or "")) or [w for c in codes or [] for w in _tokens(c)][:3]
    model = [w for w in model if w not in brand]
    size = _size_key(str(p.get("size") or ""))
    rows: dict[str, dict] = {}
    lookups = [(brand[0], brand[1:] + model)] + ([(model[0], model[1:])] if model else [])
    for head, need in lookups:
        for r in db.noon_find(mk, head, list(dict.fromkeys(need))[:5]):
            rows.setdefault(r["slug"], r)  # the same name twice (two sellers' listings): shown once
    want_accessory = bool(set(kind) & ACCESSORY)
    picked = []
    for r in rows.values():
        words = set(r["slug"].split("-"))
        if words & CONDITION or (not want_accessory and words & ACCESSORY):
            continue
        if size and size not in _size_key(r["slug"]):
            continue
        n = len(r["slug"].split("-"))
        # the names that say what the product is (its kind) first, then noon's own catalogue products (their numbers
        # start with N; the sellers' own listings with Z, often with a vague name like "samsung-galaxy-a56"), then
        # the fuller names
        picked.append((-sum(1 for w in kind if w in words), not r["sku"].startswith("N"), -min(n, 12), r))
    picked.sort(key=lambda x: x[:3])
    out = []
    for *_, r in picked[:LINKS_MAX]:
        u = url(country, r["slug"], r["sku"])
        out.append({"title": title(r["slug"]), "url": u, "store": "noon",
                    "image": f"https://f.nooncdn.com/p/v{r['img']}/{r['sku']}_1.jpg" if r["img"] else None,
                    "go": "/go?src=noon&u=" + quote(u, safe="")})
    return out


NOON_URL = re.compile(r"^https://www\.noon\.com/([a-z]+)-(?:ar|en)/([a-z0-9-]+)/([A-Z0-9]+)/p/?$")


def known(u: str) -> dict | None:
    """A noon product address from our index (the /go link only opens addresses we showed)."""
    m = NOON_URL.match(u or "")
    if not m:
        return None
    mk = next((k for k, c in MARKETS.items() if c == m.group(1)), None)
    slug, sku = m.group(2), m.group(3)
    if not mk or not db.noon_known(mk, slug.split("-", 1)[0], sku, slug):
        return None
    return {"store": "noon.com", "title": title(slug), "price": None}
