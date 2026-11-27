"""Checks how candidate stores answer our crawler, before they're added to the price list (app/catalog.py).

python -m scripts.store_probe https://store.example/ https://store.example/deals/ ...

For each address: robots.txt allows it?, the HTTP answer to our honest crawler name, how many products the catalog's
reader reads from it (and how many carry the store's own "before" price), a few samples, and for a home page the
links that look like deals / category pages. Nothing is saved.
"""
import asyncio
import re
import sys
from urllib.parse import urljoin, urlparse

import httpx

from app import catalog, fetchers

WORDS = re.compile(r"deal|offer|sale|discount|promo|clearance|mobile|phone|tv|television|laptop|appliance|"
                   r"%D8%B9%D8%B1%D9%88%D8%B6|عروض|خصم", re.I)


async def one(client: httpx.AsyncClient, url: str) -> None:
    print(f"\n=== {url}")
    try:
        ok = await fetchers.robots_allows(client, url)
    except Exception as e:
        ok, _ = False, print(f"robots error {e!r}")
    print(f"robots: {'allowed' if ok else 'NO'} ({fetchers.robots_reason(url)})")
    try:
        r = await client.get(url, headers=catalog.BOT_HEADERS, follow_redirects=True, timeout=30)
    except Exception as e:
        print(f"fetch: {type(e).__name__}: {e}"[:300])
        return
    title = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
    print(f"HTTP {r.status_code} final={r.url} {len(r.text)} chars title={' '.join((title.group(1) if title else '-').split())[:90]!r}"
          f" robotcheck={catalog._robot_check(r.text)} ld={r.text.count('application/ld+json')}"
          f" next={'__NEXT_DATA__' in r.text} server={r.headers.get('server')}")
    if r.status_code != 200:
        return
    got = catalog.read_listing(r.text, str(r.url))
    old = sum(1 for p in got if p.old_price and p.old_price > p.price)
    print(f"products read: {len(got)} (with store's before-price: {old})")
    for p in got[:4]:
        print(f"   {p.price} (was {p.old_price}) {p.title[:70]!r} {p.url[:110]}")
    if urlparse(url).path in ("", "/", "/en", "/en/", "/ar", "/ar/", "/egypt-en/"):
        host = urlparse(str(r.url)).netloc
        links = []
        for h in re.findall(r'href="([^"#]+)"', r.text):
            u = urljoin(str(r.url), h)
            if urlparse(u).netloc == host and WORDS.search(u) and u not in links:
                links.append(u)
        print(f"links ({len(links)}):")
        for u in links[:40]:
            print("   ", u)


async def main(urls: list[str]) -> None:
    async with httpx.AsyncClient() as client:
        for u in urls:
            await one(client, u)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
