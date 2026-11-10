from urllib.parse import quote_plus

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ..models import Product
from ..utils import parse_price
from .base import Store, parse_jsonld_products


class Jumia(Store):
    name = "Jumia"
    base_url = "https://www.jumia.com.eg"

    def search_url(self, text: str) -> str:
        return f"{self.base_url}/catalog/?q={quote_plus(text)}"

    def parse(self, html: str) -> list[Product]:
        out = []
        for card in HTMLParser(html).css("article.prd"):
            link = card.css_first("a.core") or card.css_first("a")
            name = card.css_first(".name")
            price = parse_price(card.css_first(".prc").text() if card.css_first(".prc") else None)
            if not (link and name and price):
                continue
            img = card.css_first("img")
            old = card.css_first(".old")
            stars = card.css_first(".stars._s")
            out.append(Product(
                store=self.name,
                title=name.text(strip=True),
                price=price,
                url=self.abs_url(link.attributes.get("href")),
                image=(img.attributes.get("data-src") or img.attributes.get("src")) if img else None,
                old_price=parse_price(old.text()) if old else None,
                rating=parse_price(stars.text()) if stars else None,
            ))
        return out or parse_jsonld_products(html, self.name, self.base_url)
