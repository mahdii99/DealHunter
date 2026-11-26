"""Every free model resting: the shopper gets an answer in seconds from our own price list (each price checked on
its store page), not a minute of searching that ends empty."""
import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import agent, db, llm, main, websearch


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def _stock():
    db.save_catalog([
        {"url": "https://www.jumia.com.eg/galaxy-a15-1.html", "store": "Jumia",
         "title": "Samsung Galaxy A15 128GB 6GB Black", "price": 7999},
        {"url": "https://www.amazon.eg/dp/B0A15A15A1", "store": "Amazon.eg",
         "title": "Samsung Galaxy A15 4G 128GB Blue", "price": 7850},
        {"url": "https://www.amazon.eg/dp/B0A15CASE1", "store": "Amazon.eg",
         "title": "Samsung Galaxy A15 silicone back case", "price": 150},
        {"url": "https://www.jumia.com.eg/galaxy-a55.html", "store": "Jumia",
         "title": "Samsung Galaxy A55 256GB", "price": 17999},
        {"url": "https://www.jumia.com.eg/sony-buds.html", "store": "Jumia",
         "title": "Sony WF-C700N wireless earbuds noise cancelling", "price": 1899},
        {"url": "https://www.jumia.com.eg/jbl-buds.html", "store": "Jumia",
         "title": "JBL Tune Flex wireless earbuds", "price": 2600},
    ], "EG", "phones")
    # what the indexing step (an LLM, offline) wrote for each title: Arabic words too
    db.save_understood([
        ("https://www.amazon.eg/dp/B0A15CASE1", json.dumps({"kind": "phone case", "accessory": True}), "جراب سامسونج"),
        ("https://www.jumia.com.eg/galaxy-a15-1.html", json.dumps({"kind": "phone"}), "موبايل سامسونج جالاكسي"),
        ("https://www.amazon.eg/dp/B0A15A15A1", json.dumps({"kind": "phone"}), "موبايل سامسونج جالاكسي"),
        ("https://www.jumia.com.eg/sony-buds.html", json.dumps({"kind": "earbuds"}), "سماعة بلوتوث عزل الضوضاء سوني"),
        ("https://www.jumia.com.eg/jbl-buds.html", json.dumps({"kind": "earbuds"}), "سماعة بلوتوث جي بي ال"),
    ])


def test_price_list_offers_for_the_shoppers_own_words():
    _stock()
    got = agent._shelf_offers("موبايل سامسونج A15", "موبايل سامسونج A15")
    assert [p.price for p in got] == [7850, 7999]  # the phone itself (not its case, not the A55), cheapest first
    buds = agent._shelf_offers("سماعات بلوتوث عزل ضوضاء تحت 2000 جنيه", "سماعات بلوتوث عزل ضوضاء تحت 2000 جنيه")
    assert [f"{p.store}{p.price:g}" for p in buds] == ["Jumia1899"]  # Arabic words, the budget kept
    assert agent._shelf_offers("غسالة اطباق", "غسالة اطباق") == []


def test_free_in_says_how_long_every_model_rests(monkeypatch):
    async def listed(client):
        return [("groq", "m1", None), ("gemini", "m2", None)]
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "_attempts", listed)
    monkeypatch.setattr(llm, "_COOLDOWN", {"m1": time.time() + 600, "m2": time.time() + 120})
    assert 100 < asyncio.run(llm.free_in()) <= 120
    monkeypatch.setattr(llm, "_COOLDOWN", {"m1": time.time() + 600})
    assert asyncio.run(llm.free_in()) == 0


def test_every_model_resting_answers_at_once_with_checked_offers(monkeypatch):
    _stock()
    asked = []

    async def free_later():
        return 900.0

    async def no_model(*a, **kw):
        asked.append(kw.get("step"))
        raise llm.LLMError(details="every model is resting")

    async def no_web(*a, **kw):
        raise AssertionError("no web search when no model can judge its pages")

    async def checked(picks, options="", patient=False, quick=False, use_llm=True):
        for p in picks:
            p.verified = True
        return picks, [], 0
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "free_in", free_later)
    monkeypatch.setattr(llm, "ask_json", no_model)
    monkeypatch.setattr(websearch, "available", lambda: True)
    monkeypatch.setattr(agent, "_web_find", no_web)
    monkeypatch.setattr(agent, "shopper_hits", no_web)
    monkeypatch.setattr(agent, "checked_only", checked)
    monkeypatch.setattr(main, "DAILY_LIMIT", 0)
    c = TestClient(main.app)
    t = time.monotonic()
    d = c.post("/api/chat", json={"messages": [{"role": "user", "text": "موبايل سامسونج A15"}]}).json()
    assert time.monotonic() - t < 5
    assert asked == []  # no planner call either: it would only fail
    assert d["outage"] and d["deals_link"]
    assert [r["price"] for r in d["results"]] == [7850, 7999]
    assert all(r["unconfirmed"] for r in d["results"])  # no model compared them to the request: "check the model"
    assert "كمان 15 دقيقة" in d["summary"]

    d = c.post("/api/chat", json={"messages": [{"role": "user", "text": "غسالة اطباق بوش"}]}).json()
    assert d["results"] == [] and d["deals_link"] and "عروض النهارده" in d["summary"]


def test_no_model_never_shows_another_kind_or_a_brand_inside_a_model_code():
    db.save_catalog([
        {"url": "https://www.btech.com/sharp-tv-50.html", "store": "B.TECH",
         "title": "Sharp 50 Inch Smart TV , 4K UHD LED - 4T-C50FJ16EX", "price": 19499},
        {"url": "https://www.btech.com/sharp-fridge.html", "store": "B.TECH",
         "title": "ثلاجة شارب نوفروست 16 قدم - SJ-58C", "price": 37898},
        {"url": "https://www.amazon.eg/dp/B0SHARPFR1", "store": "Amazon.eg",
         "title": "Sharp NoFrost Top Freezer Refrigerator 384 Liter, Black", "price": 31495},
    ], "EG", "appliances")
    got = agent._shelf_offers("ثلاجة شارب 16 قدم", "ثلاجة شارب 16 قدم")
    assert got and all("TV" not in p.title for p in got)
    eng = "lg 50 inch tv"
    levon = agent.Product(store="B.TECH", title="Levon 50 Inch Smart TV , 4K UHD QLED - 50LG8070Q", price=1, url="x")
    lg = agent.Product(store="B.TECH", title="LG 50 Inch Smart TV , 4K UHD LED - 50UA85006LA", price=1, url="y")
    assert [agent._matches_request(p, eng) for p in (levon, lg)] == [False, True]
    tv = agent.Product(store="B.TECH", title="Sharp 50 Inch Smart TV - 4T-C50FJ16EX", price=1, url="z")
    phone = agent.Product(store="Amazon.eg", title="Samsung Galaxy A36 5G 256GB", price=1, url="w")
    assert not agent._matches_request(tv, "refrigerator sharp 16")
    assert agent._matches_request(phone, "phone samsung galaxy a36")  # a title naming no kind is kept
