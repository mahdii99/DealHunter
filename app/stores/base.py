import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ..models import Product
from ..utils import parse_price

HEADERS = {
    # Honest, browser-like headers. We fetch only public search pages, one request per user search,
    # with caching, and we stop using any store whose robots.txt / terms disallow it.
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}


DEBUG_DIR = Path(__file__).resolve().parents[2] / "debug"


class Store:
    name = "store"
    base_url = ""
    min_interval = 2.0  # seconds between requests to the same store (politeness)

    def __init__(self):
        self._last = 0.0
        self._lock = asyncio.Lock()

    def search_url(self, text: str) -> str:
        raise NotImplementedError

    def parse(self, html: str) -> list[Product]:
        raise NotImplementedError

    async def search(self, client: httpx.AsyncClient, text: str) -> list[Product]:
        async with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
        r = await client.get(self.search_url(text), headers=HEADERS, follow_redirects=True, timeout=15)
        r.raise_for_status()
        products = self.parse(r.text)
        if not products:  # keep the page so we can see why the parser found nothing
            DEBUG_DIR.mkdir(exist_ok=True)
            (DEBUG_DIR / f"{self.name.lower().replace('.', '')}.html").write_text(r.text, encoding="utf-8")
        return products

    def abs_url(self, href: str | None) -> str:
        return urljoin(self.base_url, href or "")


def parse_jsonld_products(html: str, store: str, base_url: str) -> list[Product]:
    """Generic fallback: many stores embed schema.org Product / ItemList JSON-LD."""
    out = []
    tree = HTMLParser(html)
    for node in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.text())
        except ValueError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop()
            if not isinstance(item, dict):
                continue
            for key in ("itemListElement", "@graph"):
                if isinstance(item.get(key), list):
                    stack.extend(item[key])
            if isinstance(item.get("item"), dict):
                stack.append(item["item"])
            if item.get("@type") == "Product":
                offers = item.get("offers") or {}
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                price = parse_price(str(offers.get("price") or offers.get("lowPrice") or ""))
                if price and item.get("name"):
                    img = item.get("image")
                    out.append(Product(
                        store=store, title=item["name"].strip(), price=price,
                        url=urljoin(base_url, item.get("url") or offers.get("url") or ""),
                        image=img[0] if isinstance(img, list) else img,
                    ))
    return out
