"""Price history (app/pricechart.py): the chart, the real-or-fake discount label, and our /go store links."""
import time

import pytest
from fastapi.testclient import TestClient

from app import db, deals, llm, main, pricechart

DAY = 86400


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def _points(*days_prices, now=None):
    now = now or time.time()
    return [{"price": p, "in_stock": True, "seen_at": now - d * DAY} for d, p in days_prices]


def _seen(url, price, old=None, days_ago=0.0, title="Galaxy A56"):
    db.save_catalog([{"url": url, "store": "Jumia", "title": title, "price": price, "old_price": old}], "EG", "phones")
    if days_ago:
        db._db.run("UPDATE prices SET seen_at = ? WHERE url = ? AND price = ?", (time.time() - days_ago * DAY, url, price))


def test_a_price_under_anything_of_the_last_30_days_is_a_real_deal():
    v = pricechart.verdict(_points((40, 26000), (20, 25500), (8, 26000)), 23000)
    assert v["kind"] == "real" and v["low"] == 25500 and v["days"] == 30
    assert "خصم حقيقي" in v["text"]


def test_a_discount_from_a_before_price_never_asked_is_fake():
    v = pricechart.verdict(_points((25, 8300), (12, 8250), (3, 8250)), 8250, old_price=11000)
    assert v["kind"] == "fake" and "مش حقيقي" in v["text"]
    # the store really asked 11,000 lately: its discount is real enough, not fake
    assert pricechart.verdict(_points((25, 11000), (12, 8250)), 8250, old_price=11000)["kind"] != "fake"


def test_too_little_history_says_nothing_and_a_dearer_price_says_so():
    assert pricechart.verdict(_points((3, 9000)), 8000) is None  # 3 days: we don't know yet
    assert pricechart.verdict([], 8000) is None
    v = pricechart.verdict(_points((20, 9000), (10, 10000)), 10000)
    assert v["kind"] == "higher" and v["low"] == 9000


def test_the_chart_is_a_step_line_to_now_and_needs_two_readings():
    pts = _points((60, 26000), (30, 24000), (5, 25000))
    out = pricechart.svg(pts)
    assert out.startswith("<svg") and "26,000" in out and "24,000" in out and "النهارده" in out
    assert out.count("<title>") == 3 and "<style>" in out
    assert "<style>" not in pricechart.svg(pts, compact=True, style=False)
    assert pricechart.svg(_points((5, 25000))) == ""
    # a price older than the chart's window still holds at its start
    assert pricechart.svg(_points((200, 26000), (5, 25000)), days=90) != ""


def test_histories_reads_many_pages_at_once_with_the_price_that_held_before():
    _seen("https://j.eg/a", 26000, days_ago=50)
    _seen("https://j.eg/a", 25000, days_ago=10)
    _seen("https://j.eg/b", 9000, days_ago=2, title="Redmi 15")
    got = db.histories(["https://j.eg/a", "https://j.eg/b", "https://j.eg/none"], time.time() - 31 * DAY)
    assert [p["price"] for p in got["https://j.eg/a"]] == [26000, 25000]
    assert [p["price"] for p in got["https://j.eg/b"]] == [9000]
    assert "https://j.eg/none" not in got


def test_search_results_get_the_label_from_their_own_history():
    _seen("https://j.eg/a", 26000, days_ago=20)
    _seen("https://j.eg/a", 25800, days_ago=9)
    results = [{"url": "https://j.eg/a", "price": 23000}]
    db.add_price_context(results)
    assert results[0]["price_verdict"]["kind"] == "real"


def test_a_fake_store_discount_is_not_listed_as_a_deal():
    _seen("https://j.eg/c", 8250, old=11000, days_ago=20, title="Honor X5c")
    db._db.run("INSERT INTO prices (market, url, store, title, price, old_price, in_stock, seen_at) "
               "VALUES ('EG', 'https://j.eg/c', 'Jumia', 'Honor X5c', 8250, 11000, 1, ?)", (time.time(),))
    db.save_catalog([{"url": "https://j.eg/c", "store": "Jumia", "title": "Honor X5c", "price": 8250, "old_price": 11000}],
                    "EG", "phones", history=False)
    _seen("https://j.eg/e", 7000, old=9000, title="Tecno Spark")  # no history yet: the store's number, said so
    got = deals.candidates()
    assert [d["title"] for d in got] == ["Tecno Spark"]


def test_go_counts_the_click_and_opens_the_store_with_our_code(monkeypatch):
    monkeypatch.setenv("AFFILIATE_TAGS", "j.eg: tag=ours-21")
    _seen("https://j.eg/a", 25000)
    c = TestClient(main.app)
    r = c.get("/go", params={"u": "https://j.eg/a", "rank": 2, "q": "galaxy"}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "https://j.eg/a?tag=ours-21"
    rows = db._db.run("SELECT kind, detail, title, price, request, results FROM events")
    assert rows == [("click", "https://j.eg/a", "Galaxy A56", 25000, "galaxy", 2)]


def test_go_never_sends_anyone_to_an_address_we_never_showed():
    c = TestClient(main.app)
    r = c.get("/go", params={"u": "https://evil.example/login"}, follow_redirects=False)
    assert r.status_code == 200 and "evil.example" in r.text and "location" not in r.headers
    assert c.get("/go", params={"u": "javascript:alert(1)"}).status_code == 422
    assert "Disallow: /go" in c.get("/robots.txt").text


def test_history_api_gives_the_chart_and_the_label():
    _seen("https://j.eg/a", 26000, days_ago=20)
    _seen("https://j.eg/a", 25800, days_ago=9)
    _seen("https://j.eg/a", 23000)
    got = TestClient(main.app).get("/api/history", params={"url": "https://j.eg/a"}).json()
    assert got["svg"].startswith("<svg") and got["verdict"]["kind"] == "real" and len(got["points"]) == 3


def test_deals_page_links_through_go_and_shows_when_each_price_was_checked():
    now = time.time()
    html = main._deals_html({"checked_at": now, "deals": [{
        "title": "Galaxy A56", "store": "Jumia", "url": "https://www.jumia.com.eg/a.html", "image": None,
        "price": 23000, "was": 26000, "pct": 12, "kind": "drop", "category": "phones", "checked_at": now - 7200,
        "real": True, "hist": [[now - 20 * DAY, 26000], [now - 9 * DAY, 25800], [now, 23000]]}]}, "https://x.test")
    assert 'href="/go?src=deals&amp;u=https%3A%2F%2Fwww.jumia.com.eg%2Fa.html"' in html
    assert f'data-ago="{now - 7200:.0f}"' in html and 'class=spark>' in html and html.count("<style>") == 2
