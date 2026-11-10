from urllib.parse import quote_plus

from ..extract import extract_products
from ..models import Product
from .base import Store


class BTech(Store):
    """B.TECH (btech.com), a Next.js site: product data is embedded as JSON in the search page."""
    name = "B.TECH"
    base_url = "https://btech.com"
    product_url = "https://btech.com/en/p/{slug}"
    image_base = "https://media.btech.com/catalogs"

    def search_url(self, text: str) -> str:
        return f"{self.base_url}/en/s?q={quote_plus(text)}"

    def parse(self, html: str) -> list[Product]:
        return extract_products(html, self.name, self.base_url, self.product_url, self.image_base)
