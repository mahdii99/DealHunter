"""Today's deals (app/deals.py): real drops against our own history first, each one checked on its page."""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app import agent, db, deals, llm


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def _seen(url, title, price, old=None, days_ago=0.0, store="Jumia"):
    rows = [{"url": url, "store": store, "title": title, "price": price, "old_price": old}]
    db.save_catalog(rows, "EG", "phones")
    if days_ago:  # move this reading back in time
        at = time.time() - days_ago * 86400
        db._db.run("UPDATE prices SET seen_at = ? WHERE url = ? AND price = ?", (at, url, price))


def test_a_real_drop_comes_before_a_store_discount_and_small_moves_are_ignored():
    _seen("https://j.eg/a", "Galaxy A56", 26000, days_ago=5)
    _seen("https://j.eg/a", "Galaxy A56", 23000)                       # -12% vs our history
    _seen("https://j.eg/b", "Redmi 15", 10000, days_ago=3)
    _seen("https://j.eg/b", "Redmi 15", 9800)                          # -2%: not a deal
    _seen("https://j.eg/c", "Honor X5c", 8250, old=11000)              # store says -25%
    _seen("https://j.eg/d", "Oppo A6", 12899, old=13999)               # store says -8%: too small
    got = deals.candidates()
    assert [(d["title"], d["kind"], d["pct"]) for d in got] == [("Galaxy A56", "drop", 12), ("Honor X5c", "store", 25)]
    assert got[0]["was"] == 26000


def test_only_deals_still_at_that_price_on_their_page_are_listed(monkeypatch):
    _seen("https://j.eg/a", "Galaxy A56", 26000, days_ago=5)
    _seen("https://j.eg/a", "Galaxy A56", 23000)
    _seen("https://j.eg/c", "Honor X5c", 8250, old=11000)
    _seen("https://j.eg/e", "Tecno Spark", 7000, old=9000)

    async def checked(picks, options="", patient=False, use_llm=True):
        out = []
        for p in picks:
            if "Tecno" in p.title:
                continue  # sold out on its page
            if "Honor" in p.title:
                p.price = 9500  # dearer now: no longer a deal
            p.verified = True
            out.append(p)
        return out, [], 1
    monkeypatch.setattr(agent, "checked_only", checked)
    got = asyncio.run(deals.build())
    assert [(d["title"], d["price"], d["pct"]) for d in got["deals"]] == [("Galaxy A56", 23000, 12)]
    assert deals.current()["deals"][0]["title"] == "Galaxy A56"


def test_deals_page_and_cron(monkeypatch):
    from app import main
    monkeypatch.setenv("CRON_SECRET", "s3")
    c = TestClient(main.app)
    assert "لسه مفيش عروض" in c.get("/deals").text
    assert c.post("/api/cron/deals").status_code == 403
    _seen("https://j.eg/a", "Galaxy A56", 26000, days_ago=5)
    _seen("https://j.eg/a", "Galaxy A56", 23000)

    async def checked(picks, options="", patient=False, use_llm=True):
        return picks, [], 0
    monkeypatch.setattr(agent, "checked_only", checked)
    assert c.post("/api/cron/deals", headers={"X-Cron-Secret": "s3"}).json()["deals"] == 1
    page = c.get("/deals").text
    assert "Galaxy A56" in page and "23,000 ج.م" in page and "أقل سعر شفناه" in page and "data-ts=" in page
    assert "/deals" in c.get("/sitemap.xml").text


def test_many_candidates_are_checked_a_batch_at_a_time(monkeypatch):
    for i in range(45):
        _seen(f"https://j.eg/p{i}", f"Phone {i}", 8000, old=10000)
    sizes = []

    async def checked(picks, options="", patient=False, use_llm=True):
        sizes.append(len(picks))
        for p in picks:
            p.verified = True
        return picks, [], 0
    monkeypatch.setattr(agent, "checked_only", checked)
    monkeypatch.setattr(deals, "PER_CATEGORY", 100)
    got = asyncio.run(deals.build())
    assert sizes == [25, 20] and len(got["deals"]) == 45


def test_cheap_accessories_found_by_a_category_search_go_under_accessories():
    for i in range(12):
        _seen(f"https://a.eg/w{i}", f"Washer {i}", 20000 + i * 100, old=25000 + i * 100)
    _seen("https://a.eg/bag", "Shoe washing bag", 88, old=142)
    db._db.run("UPDATE catalog SET category = 'washers'")
    got = {d["title"]: d["category"] for d in deals.candidates()}
    assert got["Shoe washing bag"] == "accessories" and got["Washer 0"] == "washers"
    assert deals.category_name("accessories") == "إكسسوارات"


def test_the_deals_check_makes_no_llm_call_and_covers_every_batch(monkeypatch):
    for i in range(80):
        _seen(f"https://j.eg/q{i}", f"Phone {i}", 8000, old=10000)
    calls = []

    async def checked(picks, options="", patient=False, use_llm=True):
        calls.append((len(picks), use_llm))
        return [], [{"url": p.url} for p in picks], 0  # no page could be read without the LLM
    monkeypatch.setattr(agent, "checked_only", checked)
    monkeypatch.setattr(deals, "PER_CATEGORY", 100)
    asyncio.run(deals.build())
    assert calls == [(25, False), (25, False), (25, False), (5, False)]


def test_broad_sections_are_in_the_deals_too():
    db.save_catalog([{"url": "https://j.eg/sock", "store": "Jumia", "title": "Socks", "price": 50, "old_price": 100}],
                    "EG", "fashion")
    _seen("https://j.eg/p1", "Phone 1", 8000, old=10000)
    assert {d["title"] for d in deals.candidates()} == {"Phone 1", "Socks"}


def test_a_page_is_checked_without_the_llm_only_when_it_shows_the_listed_price(monkeypatch):
    monkeypatch.setattr(agent.recipes, "read", lambda url, html: None)
    shown = agent.Product(store="Jumia", title="A", price=8499, url="https://j.eg/a")
    other = agent.Product(store="Jumia", title="B", price=8499, url="https://j.eg/b")
    sized = agent.Product(store="Jumia", title="C", price=8499, url="https://j.eg/c")
    sold: set[int] = set()
    agent._check_without_llm([(shown, "<html><b>EGP 8,499</b></html>"), (other, "<html>EGP 9,100</html>")], sold, "")
    agent._check_without_llm([(sized, "<html>EGP 8,499</html>")], sold, "مقاس 42")
    assert shown.verified and not other.verified and not sized.verified and not sold


def test_a_trusted_recipe_reads_the_page_without_the_llm(monkeypatch):
    monkeypatch.setattr(agent.recipes, "read", lambda url, html: (8499.0, False) if "gone" in url else (8499.0, True))
    ok = agent.Product(store="Amazon", title="A", price=8499, url="https://a.eg/ok")
    gone = agent.Product(store="Amazon", title="B", price=8499, url="https://a.eg/gone")
    sold: set[int] = set()
    agent._check_without_llm([(ok, "<html>EGP 8,499</html>"), (gone, "<html>x</html>")], sold, "")
    assert ok.verified and not gone.verified and sold == {id(gone)}


def test_a_search_page_is_clear_without_the_llm_only_when_the_price_sits_by_the_product_name(monkeypatch):
    monkeypatch.setattr(agent.recipes, "read", lambda url, html: None)
    p = agent.Product(store="Jumia", title="Toshiba Washer 10 KG", price=17999, url="https://j.eg/w")
    far = "<html><h1>Toshiba Washer 10 KG</h1><p>EGP 19,500</p>" + "<p>menu item</p>" * 400 + \
          "<div>Related: Toshiba Washer 8 KG EGP 17,999</div></html>"
    near = "<html><h1>Toshiba Washer 10 KG</h1><p>EGP 17,999</p></html>"
    sold: set[int] = set()
    assert not agent._settled_without_llm(p, far, sold, "") and not p.verified  # the LLM reads this one
    assert not agent._settled_without_llm(p, near, sold, "مقاس 42")  # a size to check: the LLM reads it
    assert agent._settled_without_llm(p, near, sold, "") and p.verified and not sold
