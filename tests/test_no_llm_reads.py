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
    monkeypatch.setattr(agent, "STORE_SEARCHES", {"EG": ["https://www.amazon.eg/s?k={q}", "https://btech.com/en/s?q={q}"]})
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


def test_the_request_in_english_words_with_no_model():
    from app.query import english_words
    assert english_words("ريلمي 12 برو بلس 512 جيجا") == "realme 12 pro plus 512 gb"
    assert english_words("شاشة سامسونج 55 بوصة كريستال") == "tv samsung 55 inch crystal"
    assert english_words("سماعة انكر r50i") == "earbuds anker r50i"
    assert english_words("عايز حاجة حلوة") == ""


def test_store_search_results_must_be_what_the_shopper_wrote():
    P = agent.Product
    found = [P("Amazon.eg", "Anker Soundcore R50i True Wireless Earbuds", 848, "https://a.eg/dp/1"),
             P("Amazon.eg", "Anker Soundcore R50 NC Earbuds", 700, "https://a.eg/dp/2"),  # another model
             P("Amazon.eg", "Earbuds case for Anker R50i", 120, "https://a.eg/dp/3"),  # accessory: far cheaper
             P("B.TECH", "ايربودز انكر R50i اسود", 1240, "https://b.com/p/4"),
             P("Amazon.eg", "Anker R50i earbuds black", 990, "https://a.eg/dp/5")]
    got = agent._fitting(found, "earbuds anker r50i")
    assert [p.url for p in got] == ["https://a.eg/dp/1", "https://a.eg/dp/5", "https://b.com/p/4"]


def test_the_quick_answer_also_searches_the_big_stores(monkeypatch):
    shelf_calls = []
    monkeypatch.setattr(agent, "_shelf_offers", lambda raw, req: shelf_calls.append(req) or [])
    hit = agent.Product("Amazon.eg", "Samsung 55 Inch Crystal UHD 4K Smart TV", 20499, "https://a.eg/dp/tv")

    async def store_search(queries):
        assert queries == ["tv samsung 55 inch crystal"]
        return [hit], {}

    async def checked(picks, options="", patient=False, quick=False, use_llm=True):
        for p in picks:
            p.verified = True
        return picks, [], 0
    monkeypatch.setattr(agent, "_store_search", store_search)
    monkeypatch.setattr(agent, "checked_only", checked)
    out = asyncio.run(agent._outage_answer("شاشة سامسونج 55 بوصة كريستال", "شاشة سامسونج 55 بوصة كريستال", None, ""))
    assert [r["price"] for r in out["results"]] == [20499] and shelf_calls


def test_a_chip_or_size_alone_does_not_name_the_product():
    P = agent.Product
    found = [P("Amazon.eg", "HP 15 Laptop Core i5 8GB", 30000, "https://a.eg/dp/hp"),
             P("Amazon.eg", "Lenovo IdeaPad Slim 3 15IRH10 Core i5-13420H", 34444, "https://a.eg/dp/len")]
    assert [p.url for p in agent._fitting(found, "laptop lenovo ideapad slim 3 i5")] == ["https://a.eg/dp/len"]
