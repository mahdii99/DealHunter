"""Shops built on Shopify (many Egyptian beauty shops: Source Beauty, Bloom Pharmacy...) all answer the same public
search address, /search/suggest.json?q=..., with their products as data: name, brand, price, picture and whether
it's for sale. Read with no LLM call. The price is still checked on the product's own page before it's shown."""
import json
from urllib.parse import urljoin, urlparse

from ..models import Product
from ..utils import parse_price

SUGGEST = "/search/suggest.json?q={q}&resources[type]=product&resources[limit]=10"


def parse_suggest(text: str, url: str, store: str) -> list[Product]:
    """The products of a Shopify search answer; [] when the text isn't one."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    items = (((data or {}).get("resources") or {}).get("results") or {}).get("products") if isinstance(data, dict) else None
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    out = []
    for it in items or []:
        if not isinstance(it, dict) or not it.get("title") or not it.get("url"):
            continue
        price = parse_price(str(it.get("price") or ""))
        if not price or price < 5:
            continue
        title = " ".join(str(it["title"]).split())
        brand = " ".join(str(it.get("vendor") or "").split())
        if brand and brand.lower() not in title.lower():
            title = f"{brand} {title}"  # "Glow Deep Serum" by "Beauty of Joseon": shoppers search the brand too
        old = parse_price(str(it.get("compare_at_price_max") or ""))
        link = urljoin(base, str(it["url"]).split("?")[0])
        image = it.get("image") if isinstance(it.get("image"), str) else None
        if not image and isinstance(it.get("featured_image"), dict):
            image = it["featured_image"].get("url")
        out.append(Product(store=store, title=title[:200], price=price, url=link, image=image or None,
                           old_price=old if old and old > price else None,
                           in_stock=it.get("available") if isinstance(it.get("available"), bool) else None))
    return out
