"""noon's own public product sitemaps: the names and addresses of every product noon sells in Egypt.

Why: noon's category pages answer crawlers with a bot check (Akamai), so our price list (app/catalog.py) can't read
them, and we never try to get around such a check. But noon publishes sitemaps for search engines (listed in its
robots.txt), and its robots.txt allows product pages for every crawler. A product page carries the price as
structured data (JSON-LD Product / Offer).

This module starts with a probe: before building anything on it, the server checks that the sitemap files and a
sample of product pages open for our honest crawler name.
"""
import asyncio
import json
import logging
import re
import time

import httpx

from . import fetchers
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
