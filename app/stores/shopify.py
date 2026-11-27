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


def parse_collection(text: str, url: str, store: str) -> list[Product]:
    """A Shopify collection's public product list, /collections/<name>/products.json: each product with its options
    (variants), their prices and the store's own "before" price (compare_at_price). The option the product page
    opens on is the first one for sale, so that's the one listed, with its own address (?variant=), so the price
    checked later on the page is the same option's. [] when the text isn't one."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []
    items = data.get("products") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    out = []
    for it in items:
        if not isinstance(it, dict) or not it.get("title") or not it.get("handle"):
            continue
        variants = [v for v in it.get("variants") or [] if isinstance(v, dict)]
        sale = next((v for v in variants if v.get("available") is True), None)
        if sale is None:  # nothing for sale (or the store hides stock): not a deal we can stand behind
            continue
        price = parse_price(str(sale.get("price") or ""))
        if not price or price < 5:
            continue
        title = " ".join(str(it["title"]).split())
        brand = " ".join(str(it.get("vendor") or "").split())
        if brand and brand.lower() not in title.lower() and brand.lower() not in base.lower():
            title = f"{brand} {title}"
        option = " ".join(str(sale.get("title") or "").split())
        if option and option.lower() != "default title" and option.lower() not in title.lower():
            title = f"{title} - {option}"
        old = parse_price(str(sale.get("compare_at_price") or ""))
        link = urljoin(base, f"/products/{it['handle']}")
        if len(variants) > 1 and sale.get("id"):
            link += f"?variant={sale['id']}"
        imgs = [i for i in it.get("images") or [] if isinstance(i, dict) and i.get("src")]
        image = next((i["src"] for i in imgs if sale.get("id") in (i.get("variant_ids") or [])), None)
        image = image or (imgs[0]["src"] if imgs else None)
        out.append(Product(store=store, title=title[:200], price=price, url=link, image=image,
                           old_price=old if old and old > price else None, in_stock=True))
    return out
