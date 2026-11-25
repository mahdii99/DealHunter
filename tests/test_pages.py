"""Product pages (app/pages.py): one page per product, only offers checked on their store page, in the sitemap."""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import agent, db, llm, main, pages


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def _row(url, title, price, store, attrs, category="phones"):
    db.save_catalog([{"url": url, "store": store, "title": title, "price": price, "image": "https://img.eg/a.jpg"}],
                    "EG", category)
    db.save_understood([(url, json.dumps(attrs), "")])


A56 = {"brand": "Samsung", "line": "Galaxy A56", "size": "256 GB", "kind": "smartphone"}


def test_same_product_in_different_stores_is_one_page():
    assert pages.product_key(A56) == ("samsung-galaxy-a56-256gb", "Samsung Galaxy A56 256 GB")
    assert pages.product_key({**A56, "line": "Samsung Galaxy A56", "size": "256GB"})[0] == "samsung-galaxy-a56-256gb"
    assert pages.product_key({**A56, "accessory": True}) is None
    assert pages.product_key({"brand": "Samsung", "kind": "phone"}) is None
    assert pages.product_key({"brand": "سامسونج", "line": "جالاكسي"}) is None


def test_build_keeps_only_offers_checked_on_their_page(monkeypatch):
    _row("https://www.jumia.com.eg/a56", "Samsung Galaxy A56 256GB", 23000, "Jumia", A56)
    _row("https://www.amazon.eg/a56", "Samsung Galaxy A56 5G 256 GB", 22500, "Amazon.eg", {**A56, "size": "256GB"})
    _row("https://www.amazon.eg/a56-case", "Case for Galaxy A56", 200, "Amazon.eg", A56)  # far under: not this product
    _row("https://www.jumia.com.eg/other", "Phone case", 150, "Jumia", {**A56, "accessory": True})

    async def checked(picks, options="", patient=False, use_llm=True):
        assert {p.url for p in picks} == {"https://www.jumia.com.eg/a56", "https://www.amazon.eg/a56"}
        out = []
        for p in picks:
            if "amazon" in p.url:
                continue  # its page didn't show the price: never shown
            p.price, p.verified = 23100, True  # the page's price, not the listing's
            out.append(p)
        return out, [], 0
    monkeypatch.setattr(agent, "checked_only", checked)
    got = asyncio.run(pages.build())
    assert got["with_offers"] == 1 and got["offers"] == 1
    page = pages.get("samsung-galaxy-a56-256gb")
    assert [(o["store"], o["price"]) for o in pages.fresh_offers(page)] == [("Jumia", 23100)]
    assert len(page["urls"]) == 2

    # a later run that confirms nothing keeps the last checked offers until they're too old
    async def none(picks, options="", patient=False, use_llm=True):
        return [], [], 0
    monkeypatch.setattr(agent, "checked_only", none)
    asyncio.run(pages.build())
    assert pages.fresh_offers(pages.get("samsung-galaxy-a56-256gb"))
    assert not pages.fresh_offers(pages.get("samsung-galaxy-a56-256gb"), now=time.time() + 27 * 3600)


def _page(offers):
    db.save_product_page("samsung-galaxy-a56-256gb", "EG", "Samsung Galaxy A56 256GB", "phones", "Samsung",
                         ["https://www.jumia.com.eg/a56"], offers, time.time())


def test_product_page_shows_checked_prices_json_ld_and_is_in_the_sitemap():
    now = time.time()
    _page([{"title": "Samsung Galaxy A56 256GB", "store": "Jumia", "url": "https://www.jumia.com.eg/a56",
            "price": 23100, "old_price": 26000, "image": "https://img.eg/a.jpg", "checked_at": now},
           {"title": "Galaxy A56 256", "store": "Amazon.eg", "url": "https://www.amazon.eg/a56",
            "price": 23900, "old_price": None, "image": "", "checked_at": now}])
    c = TestClient(main.app)
    html = c.get("/p/samsung-galaxy-a56-256gb").text
    assert "سعر Samsung Galaxy A56 256GB في مصر" in html
    assert "23,100 ج.م" in html and "تابع السعر" in html and "noindex" not in html
    ld = json.loads(html.split('<script type="application/ld+json">')[1].split("</script>")[0])
    assert ld["@type"] == "Product" and ld["offers"]["lowPrice"] == 23100 and ld["offers"]["offerCount"] == 2
    assert "/p/samsung-galaxy-a56-256gb" in c.get("/sitemap.xml").text
    assert "/p/samsung-galaxy-a56-256gb" in c.get("/p").text
    assert c.get("/p/nothing-here").status_code == 404
    assert c.get("/p/Bad Slug").status_code in (404, 422)


def test_a_page_without_a_recent_check_shows_no_price_and_is_not_listed():
    old = time.time() - 30 * 3600
    _page([{"title": "Samsung Galaxy A56 256GB", "store": "Jumia", "url": "https://www.jumia.com.eg/a56",
            "price": 23100, "old_price": None, "image": "", "checked_at": old}])
    c = TestClient(main.app)
    html = c.get("/p/samsung-galaxy-a56-256gb").text
    assert "23,100" not in html and "noindex" in html and "application/ld+json" not in html
    assert "/p/samsung-galaxy-a56-256gb" not in c.get("/sitemap.xml").text


def test_pages_cron_needs_the_secret(monkeypatch):
    c = TestClient(main.app)
    assert c.post("/api/cron/pages").status_code == 403
    monkeypatch.setenv("CRON_SECRET", "s3")

    async def none(picks, options="", patient=False, use_llm=True):
        return [], [], 0
    monkeypatch.setattr(agent, "checked_only", none)
    assert c.post("/api/cron/pages", headers={"X-Cron-Secret": "s3"}).json()["products"] == 0
