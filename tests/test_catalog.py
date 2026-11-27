"""Our own price list (app/catalog.py): reading category pages, saving them, and searches using them."""
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app import agent, catalog, db, llm, websearch
from app.models import Product

@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


AMAZON = """<html><body><div class="s-main-slot">
<div data-asin="B0AAAAAAA1" data-component-type="s-search-result"><div class="puis-card" data-asin="B0AAAAAAA1">
  <img class="s-image" src="https://m.media-amazon.com/a1.jpg" alt="x">
  <h2><a href="/Samsung-Galaxy-A56/dp/B0AAAAAAA1/ref=sr_1_1"><span>Samsung Galaxy A56 5G, 256GB, 8GB RAM, Graphite</span></a></h2>
  <span class="a-price"><span class="a-offscreen">EGP 19,999.00</span><span aria-hidden="true">19,999</span></span>
  <span class="a-price a-text-price"><span class="a-offscreen">EGP 23,499.00</span></span>
</div></div>
<div data-asin="B0AAAAAAA4"><div id="B0AAAAAAA4" class="p13n-sc-uncoverable-faceout"><a href="/-/en/x/dp/B0AAAAAAA4/ref=zg">
  <img alt="Apple MacBook Air 13" src="https://img/m.jpg"></a><div class="_cDEzb_p13n-sc-css-line-clamp-3_g3dy1">Apple 2025 MacBook Air 13-inch M4 512GB</div>
  <span class="a-price"><span class="a-offscreen">EGP85,02999</span><span aria-hidden="true"><span class="a-price-symbol">EGP</span>
  <span class="a-price-whole">85,029<span class="a-price-decimal"></span></span><span class="a-price-fraction">99</span></span></span></div></div>
<div data-asin="B0AAAAAAA2" class="AdHolder"><h2><span>Sponsored phone with a long name</span></h2>
  <span class="a-price"><span class="a-offscreen">EGP 9,999.00</span></span></div>
<div data-asin="B0AAAAAAA3"><h2><span>Xiaomi Redmi Note 14 without a price shown</span></h2></div>
<div data-asin=""><span>header</span></div>
</div></body></html>"""

JUMIA = """<html><body>
<article class="prd _fb col c-prd"><a class="core" href="/honor-x5c-plus-128gb-black-133885513.html">
<div class="img-c"><img data-src="https://eg.jumia.is/x.jpg"></div><div class="info"><h3 class="name">Honor X5c Plus Dual SIM 4G 128GB/4GB - Black</h3>
<div class="prc">EGP 8,250</div><div class="s-prc-w"><div class="old">EGP 9,900</div></div></div></a></article>
<article class="prd _fb col c-prd"><a class="core" href="https://www.jumia.com.eg/oppo-a6-128gb-gold-134895106.html">
<div class="info"><h3 class="name">OPPO A6 Dual SIM 4G 128GB/4GB - Aurora Gold</h3><div class="prc">EGP 12,899</div></div></a></article>
</body></html>"""


def test_amazon_cards_are_read_without_ads_or_cards_without_a_price():
    got = catalog.read_listing(AMAZON, "https://www.amazon.eg/s?k=smartphone")
    # (a best-seller card: its hidden price "EGP85,02999" lost the decimal point, the shown parts didn't)
    assert [(p.title[:20], p.price, p.old_price) for p in got] == [("Samsung Galaxy A56 5", 19999, 23499),
                                                                   ("Apple 2025 MacBook A", 85029.99, None)]
    assert got[0].url == "https://www.amazon.eg/dp/B0AAAAAAA1" and got[0].store == "Amazon.eg"
    assert got[0].image == "https://m.media-amazon.com/a1.jpg"


def test_jumia_cards_are_read_with_the_stores_own_links():
    got = catalog.read_listing(JUMIA, "https://www.jumia.com.eg/mobile-phones/?page=1")
    assert [(p.price, p.old_price, p.store) for p in got] == [(8250, 9900, "Jumia"), (12899, None, "Jumia")]
    assert got[0].url == "https://www.jumia.com.eg/honor-x5c-plus-128gb-black-133885513.html"


def test_saved_catalog_keeps_the_latest_price_and_the_history():
    rows = [{"url": "https://www.jumia.com.eg/oppo-a6.html", "store": "Jumia", "title": "OPPO A6 128GB", "price": 12899}]
    assert db.save_catalog(rows, "EG", "phones") == 1
    db.save_catalog([{**rows[0], "price": 11999}], "EG", "phones")
    got = catalog.offers(["oppo"])
    assert [(p.price, p.trusted, p.trust_score) for p in got] == [(11999, True, 100)]
    assert [h["price"] for h in db.history(rows[0]["url"])] == [12899, 11999]
    assert catalog.offers(["oppo"], max_age_hours=-1) == []  # old prices are never used


def test_scrape_obeys_robots_and_reports_a_store_that_blocks(monkeypatch):
    async def allowed(client, url):
        return "jumia" in url
    monkeypatch.setattr(catalog.fetchers, "robots_allows", allowed)
    monkeypatch.setattr(catalog, "PAUSE", 0)
    agents = []

    def handler(request: httpx.Request):
        agents.append(request.headers["user-agent"])
        page = int(request.url.params.get("page", "1"))
        return httpx.Response(200, text=JUMIA if page <= 2 else JUMIA)  # page 3 repeats page 1: the end
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    report = asyncio.run(catalog.scrape(["jumia.com.eg", "amazon.eg"], ["phones"], client=client))
    assert report["jumia.com.eg phones"]["products"] == 2 and "error" not in report["jumia.com.eg phones"]
    assert report["amazon.eg phones"]["products"] == 0 and "robots.txt" in report["amazon.eg phones"]["error"]
    assert len(agents) == 2 and all(a.startswith("DealHunterBot/") for a in agents)  # one honest name, stops at a repeat


def test_scrape_stops_at_a_403(monkeypatch):
    async def allowed(client, url):
        return True
    monkeypatch.setattr(catalog.fetchers, "robots_allows", allowed)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403, text="Just a moment...")))
    report = asyncio.run(catalog.scrape(["jumia.com.eg"], ["phones"], client=client))
    assert report["jumia.com.eg phones"] == {"products": 0, "seconds": report["jumia.com.eg phones"]["seconds"],
                                             "error": "HTTP 403"}


def test_scrape_endpoint_needs_the_cron_secret(monkeypatch):
    from app import main
    monkeypatch.setenv("CRON_SECRET", "s3")
    c = TestClient(main.app)
    assert c.post("/api/cron/scrape").status_code == 403
    assert c.post("/api/cron/scrape?store=x.com", headers={"X-Cron-Secret": "s3"}).status_code == 400

    async def fake(stores, categories, pages):
        return {"jumia.com.eg phones": {"products": 7, "seconds": 1.0}}
    monkeypatch.setattr(catalog, "scrape", fake)
    r = c.post("/api/cron/scrape?store=jumia.com.eg", headers={"X-Cron-Secret": "s3"})
    assert r.json()["total"] == 7
    assert "DealHunterBot" in c.get("/bot").text


def _stock(n_jumia=4, n_amazon=3):
    rows = [{"url": f"https://www.jumia.com.eg/galaxy-a56-{i}.html", "store": "Jumia",
             "title": f"Samsung Galaxy A56 5G 256GB color {i}", "price": 19000 + i} for i in range(n_jumia)]
    rows += [{"url": f"https://www.amazon.eg/dp/B0A56A56A{i}", "store": "Amazon.eg",
              "title": f"Samsung Galaxy A56 5G 8GB 256GB v{i}", "price": 19500 + i} for i in range(n_amazon)]
    rows.append({"url": "https://www.jumia.com.eg/oppo-a6.html", "store": "Jumia", "title": "OPPO A6 128GB",
                 "price": 9000})
    db.save_catalog(rows, "EG", "phones")


def test_search_uses_the_price_list_and_spends_fewer_web_searches(monkeypatch):
    _stock()
    calls = []

    async def fake_find(queries, seed=None, famous=None, want="", named=None):
        calls.append((list(queries), list(famous or [])))
        return [Product("2B", "Samsung Galaxy A56 5G 256GB", 18900, "https://2b.com.eg/p/galaxy-a56")], {}

    async def fake_ask(prompt, **kw):
        assert "OPPO" not in prompt  # only the price list's products that fit the request reach the judge
        return {"matches": [{"id": i, "sure": True} for i in range(prompt.count("| Samsung"))], "summary": "s"}
    monkeypatch.setattr(websearch, "available", lambda: True)
    monkeypatch.setattr(agent, "_web_find", fake_find)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    plan = {"action": "search", "request": "سامسونج A56", "simple": True,
            "web_queries": ["سعر سامسونج جالاكسي A56", "Samsung Galaxy A56 5G 256GB price Egypt"]}
    out = asyncio.run(agent.run("سامسونج A56", plan=plan))
    queries, famous = calls[0]
    assert queries == ["سعر سامسونج جالاكسي A56"]  # one web search round instead of two
    assert "amazon.eg" not in famous and "jumia.com.eg" not in famous and "noon.com" in famous
    assert out["from_catalog"] == 8 - 1 and {r["store"] for r in out["results"]} == {"Jumia", "Amazon.eg", "2B"}


def test_few_price_list_offers_keep_every_web_search(monkeypatch):
    _stock(n_jumia=2, n_amazon=0)
    calls = []

    async def fake_find(queries, seed=None, famous=None, want="", named=None):
        calls.append(list(queries))
        return [], {}
    monkeypatch.setattr(websearch, "available", lambda: True)
    monkeypatch.setattr(agent, "_web_find", fake_find)
    monkeypatch.setattr(llm, "available", lambda: True)

    async def fake_ask(prompt, **kw):
        return {"matches": [], "summary": ""}
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    plan = {"action": "search", "request": "A56", "simple": True,
            "web_queries": ["سعر سامسونج A56", "Samsung Galaxy A56 price"]}
    asyncio.run(agent.run("A56", plan=plan))
    assert calls == [["سعر سامسونج A56", "Samsung Galaxy A56 price"]]


def test_a_page_that_refuses_a_browser_is_asked_again_under_our_crawler_name(monkeypatch):
    async def allowed(client, url):
        return True
    monkeypatch.setattr(agent.fetchers, "robots_allows", allowed)
    seen = []

    def handler(request: httpx.Request):
        ua = request.headers["user-agent"]
        seen.append(ua.split("/")[0])
        return httpx.Response(200, text=JUMIA) if ua.startswith("DealHunterBot/") else httpx.Response(403, text="no")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await agent._fetch_page(c, asyncio.Semaphore(1), "https://www.jumia.com.eg/mobile-phones/")
    products, html = asyncio.run(go())
    assert seen == ["Mozilla", "DealHunterBot"] and "Honor" in html


def test_an_empty_first_page_is_reported_with_its_title(monkeypatch):
    async def allowed(client, url):
        return True
    monkeypatch.setattr(catalog.fetchers, "robots_allows", allowed)
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text="<html><title>Something else</title><body>hi</body></html>")))
    report = asyncio.run(catalog.scrape(["amazon.eg"], ["phones"], client=client))
    assert "title: Something else" in report["amazon.eg phones"]["error"]


def _web_and_llm(monkeypatch, judge_prompts, find=None):
    async def fake_find(queries, seed=None, famous=None, want="", named=None):
        return (find or [])[:], {}

    async def fake_ask(prompt, step="other", **kw):
        if step == "catalog_words":
            return {"queries": ["Samsung Galaxy A56 256GB"]}
        judge_prompts.append(prompt)
        lines = [ln for ln in prompt.split("\n") if ln[:1].isdigit() and ". [" in ln[:6]]
        return {"matches": [{"id": i, "sure": True} for i, ln in enumerate(lines) if "Samsung" in ln or "A56" in ln],
                "summary": "s"}
    monkeypatch.setattr(websearch, "available", lambda: True)
    monkeypatch.setattr(agent, "_web_find", fake_find)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)


def test_an_arabic_only_request_still_finds_the_price_list(monkeypatch):
    """Live 23 Nov: with the planner busy there was no English search, and the price list (English titles) was
    skipped, so the big stores were missing. A light model now writes the English words for it."""
    _stock()
    prompts = []
    _web_and_llm(monkeypatch, prompts)
    plan = {"action": "search", "request": "سامسونج ايه 56", "simple": True, "web_queries": ["سامسونج ايه 56"]}
    out = asyncio.run(agent.run("سامسونج ايه 56", plan=plan))
    assert out["from_catalog"] == 7 and len(out["results"]) == 7
    assert out["llm_usage"] is not None


def test_price_list_offers_come_first_and_are_not_judged_twice(monkeypatch):
    """The big stores' offers from our price list are judged and shown first (a 'partial' answer), while the web
    search goes on; the final judge only sees the new shops' offers."""
    _stock()
    prompts, partial = [], []
    _web_and_llm(monkeypatch, prompts, find=[Product("2B", "Samsung Galaxy A56 5G 256GB", 18900,
                                                     "https://2b.com.eg/p/galaxy-a56")])

    async def go():
        token = agent.PARTIAL.set(partial.append)
        try:
            return await agent.run("سامسونج A56", plan={"action": "search", "request": "سامسونج A56", "simple": True,
                                                         "web_queries": ["Samsung Galaxy A56 price"]})
        finally:
            agent.PARTIAL.reset(token)
    out = asyncio.run(go())
    assert len(partial) == 1 and {r["store"] for r in partial[0]["results"]} == {"Jumia", "Amazon.eg"}
    assert len(prompts) == 2 and "2B" not in prompts[0]
    judged_last = [ln for ln in prompts[1].split("\n") if ln[:1].isdigit() and ". [" in ln[:6]]
    assert judged_last == [ln for ln in judged_last if "[2B]" in ln] and len(judged_last) == 1
    assert "already checked" in prompts[1]  # the first offers are given for comparison
    assert len(out["results"]) == 8 and out["first_results"] == 7
    # the same request again (after the saved answer expired): every verdict is reused, no judge call at all
    prompts.clear()
    out = asyncio.run(go())
    assert prompts == [] and len(out["results"]) == 8


def test_a_verdict_is_not_reused_when_the_price_moved_a_lot(monkeypatch):
    _stock(n_jumia=1, n_amazon=0)
    prompts = []
    _web_and_llm(monkeypatch, prompts)
    plan = {"action": "search", "request": "A56", "simple": True, "web_queries": ["Samsung Galaxy A56"]}
    asyncio.run(agent.run("A56", plan=plan))
    assert len(prompts) == 1
    db.save_catalog([{"url": "https://www.jumia.com.eg/galaxy-a56-0.html", "store": "Jumia",
                      "title": "Samsung Galaxy A56 5G 256GB color 0", "price": 9000}], "EG", "phones")
    prompts.clear()
    asyncio.run(agent.run("A56", plan=plan))
    assert len(prompts) == 1  # judged again: half the price can be another item or a mistake


SHOPIFY = """{"products": [
 {"id": 1, "title": "Samsung Galaxy A17 Dual SIM 256GB", "handle": "samsung-galaxy-a17", "vendor": "Samsung",
  "variants": [{"id": 11, "title": "Black", "price": "16999.00", "compare_at_price": "18999.00", "available": false},
               {"id": 12, "title": "Light Blue", "price": "16999.00", "compare_at_price": "18999.00", "available": true}],
  "images": [{"src": "https://cdn.shopify.com/a.jpg", "variant_ids": []}, {"src": "https://cdn.shopify.com/b.jpg", "variant_ids": [12]}]},
 {"id": 2, "title": "Anker Charger 20W", "handle": "anker-20w", "vendor": "Anker",
  "variants": [{"id": 21, "title": "Default Title", "price": "650.00", "compare_at_price": null, "available": true}], "images": []},
 {"id": 3, "title": "Sold out phone", "handle": "gone", "variants": [{"id": 31, "price": "9000", "available": false}]}
]}"""


def test_shopify_collection_lists_the_option_the_page_opens_on():
    got = catalog.read_listing(SHOPIFY, "https://kimostore.net/collections/mobile-phone/products.json?page=1")
    assert [(p.title, p.price, p.old_price) for p in got] == [
        ("Samsung Galaxy A17 Dual SIM 256GB - Light Blue", 16999, 18999), ("Anker Charger 20W", 650, None)]
    assert got[0].url == "https://kimostore.net/products/samsung-galaxy-a17"
    assert got[0].image == "https://cdn.shopify.com/b.jpg" and got[1].url == "https://kimostore.net/products/anker-20w"


MAGENTO = """<ol><li class="product-item"><div class="product-item-info">
<a href="https://2b.com.eg/en/galaxy-a37.html"><img class="product-image-photo" src="https://2b.com.eg/media/a37.jpg"></a>
<strong><a class="product-item-link" href="https://2b.com.eg/en/galaxy-a37.html"> Samsung Galaxy A37 5G - 256GB </a></strong>
<div class="price-box"><span class="old-price"><span data-price-amount="48449" data-price-type="oldPrice">EGP 48,449</span></span>
<span data-price-amount="36349" data-price-type="finalPrice">EGP 36,349</span></div></div></li>
<li class="product-item"><div class="product-item-info"><a class="product-item-link" href="/en/sold.html">Sold phone</a>
<span data-price-amount="9000" data-price-type="finalPrice"></span><div class="stock unavailable">Out of stock</div></div></li></ol>"""

PRESTA = """<ul><li class="ajax_block_product"><div class="left-block"><a class="product_img_link" href="https://cairosales.com/en/tvs/samsung-55.html">
<img src="https://cairosales.com/1-home/samsung-55.jpg"></a></div><div class="right-block">
<h5><a class="product-name" href="https://cairosales.com/en/tvs/samsung-55.html" title="SAMSUNG 55-inch Crystal UHD 4K Smart TV">SAMSUNG 55-inch...</a></h5>
<div class="content_price"><span class="price product-price"> EGP 19,999 </span><span class="old-price product-price"> EGP 22,726 </span>
<span class="availability"><span class="available-now">In Stock.</span></span></div></div></li></ul>"""


def test_magento_and_prestashop_cards_are_read_with_their_before_price():
    got = catalog.read_listing(MAGENTO, "https://2b.com.eg/en/mobile-and-tablet/mobiles.html")
    assert [(p.title, p.price, p.old_price, p.store) for p in got] == [("Samsung Galaxy A37 5G - 256GB", 36349, 48449, "2B")]
    assert got[0].image == "https://2b.com.eg/media/a37.jpg"
    got = catalog.read_listing(PRESTA, "https://cairosales.com/en/tvs-screens/")
    assert [(p.title, p.price, p.old_price, p.store) for p in got] == [
        ("SAMSUNG 55-inch Crystal UHD 4K Smart TV", 19999, 22726, "Cairo Sales")]


def test_a_page_read_whole_is_not_asked_for_again(monkeypatch):
    monkeypatch.setattr(catalog, "PAUSE", 0)
    asked = []

    async def page(client, url):
        asked.append(url)
        return [Product(store="2B", title="Galaxy A37", price=36349, url=f"https://2b.com.eg/en/a{len(asked)}.html")]
    monkeypatch.setattr(catalog, "_page", page)
    asyncio.run(catalog.scrape(["2b.com.eg"], ["phones"]))
    assert asked == ["https://2b.com.eg/en/mobile-and-tablet/mobiles.html"]
