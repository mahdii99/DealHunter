"""Product cards of the shop platforms many Egyptian stores are built on, read the same way for every store on them:

  - Magento 2 (2B and many electronics shops): each card is a .product-item-info with its link (.product-item-link)
    and its prices as numbers in data-price-amount (the final price, and the old one when there is one).
  - PrestaShop (Cairo Sales and others): 1.6 cards (.ajax_block_product: .product-name, .price, .old-price) and
    1.7+ cards (.product-miniature: .product-title, .price, .regular-price).

Only what the card itself shows is read; a card with no price or marked out of stock is skipped. The price is still
checked on the product's own page before it's shown anywhere (agent.checked_only).
"""
from urllib.parse import urljoin

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ..models import Product
from ..utils import parse_price

_SOLD_OUT = ("out of stock", "sold out", "غير متوفر", "نفذت", "نفدت", "unavailable")


def _amount(card, kind: str) -> float | None:
    node = card.css_first(f'[data-price-type="{kind}"][data-price-amount]')
    return parse_price(node.attributes.get("data-price-amount")) if node is not None else None


def _text(node) -> str:
    return " ".join(node.text().split()) if node is not None else ""


def _image(card) -> str | None:
    img = card.css_first("img")
    if img is None:
        return None
    src = img.attributes.get("data-src") or img.attributes.get("data-original") or img.attributes.get("src") or ""
    return src if src.startswith("http") else None


def _sold_out(card) -> bool:
    stock = card.css_first(".stock.unavailable, .out-of-stock, .label-danger, .product-unavailable")
    return stock is not None or any(w in _text(card.css_first(".availability, .stock")).lower() for w in _SOLD_OUT)


def parse_platform_cards(html: str, store: str, url: str) -> list[Product]:
    tree = HTMLParser(html)
    out, seen = [], set()

    def add(title: str, link: str | None, price: float | None, old: float | None, card):
        if not (title and link and price) or _sold_out(card):
            return
        link = urljoin(url, link)
        if link in seen:
            return
        seen.add(link)
        out.append(Product(store=store, title=title[:200], price=price, url=link, image=_image(card),
                           old_price=old if old and old > price else None))

    for card in tree.css(".product-item-info"):  # Magento 2
        a = card.css_first("a.product-item-link")
        add(_text(a), a.attributes.get("href") if a else None, _amount(card, "finalPrice"), _amount(card, "oldPrice"), card)
    for card in tree.css("li.ajax_block_product, .product-miniature"):  # PrestaShop 1.6 / 1.7+
        a = card.css_first("a.product-name, .product-title a")
        price = card.css_first(".right-block .price, .product-price-and-shipping .price, .price")
        old = card.css_first(".right-block .old-price, .regular-price, .old-price")
        title = (a.attributes.get("title") or _text(a)) if a else ""
        add(" ".join(title.split()), a.attributes.get("href") if a else None,
            parse_price(_text(price)), parse_price(_text(old)) if old else None, card)
    return out
