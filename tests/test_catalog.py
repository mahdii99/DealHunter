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
<div data-asin="B0AAAAAAA1" data-component-type="s-search-result"><div class="puis-card">
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
    report = asyncio.run(catalog.scrape(["jumia.com.eg", "amazon.eg"], client=client))
    assert report["jumia.com.eg phones"]["products"] == 2 and "error" not in report["jumia.com.eg phones"]
    assert report["amazon.eg phones"]["products"] == 0 and "robots.txt" in report["amazon.eg phones"]["error"]
    assert len(agents) == 2 and all(a.startswith("DealHunterBot/") for a in agents)  # one honest name, stops at a repeat


def test_scrape_stops_at_a_403(monkeypatch):
    async def allowed(client, url):
        return True
    monkeypatch.setattr(catalog.fetchers, "robots_allows", allowed)
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(403, text="Just a moment...")))
    report = asyncio.run(catalog.scrape(["noon.com"], client=client))
    assert report["noon.com phones"] == {"products": 0, "seconds": report["noon.com phones"]["seconds"],
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
