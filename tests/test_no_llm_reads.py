"""Pages read with no LLM call: a clear product page, and the big stores' own search pages."""
import asyncio

from app import agent, extract, market


def test_a_clear_product_page_is_read_without_the_llm():
    page = ("<html><body><nav>Home EGP 99</nav><h1>Realme 12 Pro Plus 5G 512GB</h1>"
            "<div class='price'>EGP 21,999</div><p>Free delivery</p></body></html>")
    assert extract.main_product(page) == ("Realme 12 Pro Plus 5G 512GB", 21999.0)
    arabic = "<html><h1>غسالة توشيبا 10 كيلو</h1><span>١٨٬٥٠٠ جنيه</span></html>"
    assert extract.main_product(arabic) == ("غسالة توشيبا 10 كيلو", 18500.0)
    old = "<html><h1>Realme 12 Pro Plus</h1><del>EGP 24,999</del> <span class='old-price'>EGP 23,000</span> EGP 21,999</html>"
    assert extract.main_product(old) == ("Realme 12 Pro Plus", 21999.0)  # crossed-out prices aren't the price


def test_a_page_that_is_not_clear_goes_to_the_llm():
    two = "<html><h1>Realme 12 Pro Plus</h1>EGP 21,999 or EGP 1,833 a month</html>"  # two prices
    none = "<html><h1>Realme 12 Pro Plus</h1><p>Call us for the price</p></html>"
    untitled = "<html><p>EGP 21,999</p></html>"
    assert extract.main_product(two) is None and extract.main_product(none) is None
    assert extract.main_product(untitled) is None


def test_the_big_stores_own_search_pages_are_read_without_the_llm(monkeypatch):
    monkeypatch.setattr(market, "current", lambda: market.MARKETS["EG"])
    asked = []

    async def fake_fetch(client, sem, url):
        asked.append(url)
        if "btech" in url:
            raise TimeoutError("slow")
        card = ('<div data-asin="B0CXYZ1234"><h2><a href="/dp/B0CXYZ1234"><span>Realme 12 Pro+ 512GB</span></a></h2>'
                '<span class="a-price"><span class="a-price-whole">21,999</span></span></div>')
        return [], f"<html>{card}</html>"
    monkeypatch.setattr(agent, "_fetch_page", fake_fetch)
    got, errors = asyncio.run(agent._store_search(["realme 12 pro plus", "ريلمي 12 برو"]))
    assert len(asked) == 2 and "amazon.eg/s?k=realme+12+pro+plus" in asked[0]  # Arabic words: no store search
    assert [(p.price, p.trusted) for p in got] == [(21999.0, True)] and "btech.com search" in errors
