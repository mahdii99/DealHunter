import re
from urllib.parse import urljoin

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ..models import Product
from ..utils import parse_price


def _price(node) -> float | None:
    """An Amazon price box. Its hidden copy can lose the decimal point ("EGP85,02999" for 85,029.99), so the
    pounds and piasters it shows are read apart when they're there."""
    if node is None:
        return None
    whole = node.css_first(".a-price-whole")
    if whole is not None:
        pounds = re.sub(r"\D", "", whole.text())
        frac = node.css_first(".a-price-fraction")
        cents = re.sub(r"\D", "", frac.text()) if frac is not None else ""
        if pounds:
            return float(pounds) + (float(cents[:2]) / 10 ** len(cents[:2]) if cents else 0)
    hidden = node.css_first(".a-offscreen")
    return parse_price(hidden.text()) if hidden is not None else None


def parse_amazon_cards(html: str, store: str, base_url: str) -> list[Product]:
    """Amazon's search, category and best-seller pages carry no product data as JSON: each product is a card
    (an element with a data-asin product id) with its own link, name, price and picture. Works the same on every
    Amazon country site. A card without a price shown (sold out, or "see options") is skipped."""
    out, seen = [], set()
    for card in HTMLParser(html).css("[data-asin]"):
        asin = (card.attributes.get("data-asin") or "").strip()
        if len(asin) != 10 or asin in seen or len(card.css("[data-asin]")) > 1:  # (css() counts the card itself: >1 = a wrapper of other cards)
            continue
        if "AdHolder" in (card.attributes.get("class") or "") or card.css_first(".puis-sponsored-label-text"):
            continue  # a paid ad: it repeats a product from elsewhere in the list
        price = _price(card.css_first(".a-price:not(.a-text-price)"))
        if not price:
            for node in card.css("._cDEzb_p13n-sc-price_3mJ9Z, .p13n-sc-price"):
                if price := parse_price(node.text()):
                    break
        if not price:
            continue
        name = None
        for sel in ("h2 span", "h2", "[class*='p13n-sc-truncate']", "[class*='line-clamp']", "img[alt]"):
            node = card.css_first(sel)
            text = (node.attributes.get("alt") if sel == "img[alt]" else node.text(strip=True)) if node else ""
            if text and len(text) > 8:
                name = " ".join(text.split())
                break
        if not name:
            continue
        old = _price(card.css_first(".a-price.a-text-price"))
        img = card.css_first("img")
        seen.add(asin)
        out.append(Product(
            store=store, title=name, price=price, url=urljoin(base_url, f"/dp/{asin}"),
            image=img.attributes.get("src") if img else None,
            old_price=old if old and old > price else None,
        ))
    return out
