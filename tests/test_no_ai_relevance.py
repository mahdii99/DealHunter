"""With no model to judge: the request understood by words (brand, kind, specs, features, model variant) and only
offers that are that product shown, never its accessories or another model (live 2026-11-26: "سماعة جي بي ال"
showed a cleaning kit and non-JBL earbuds, "لابتوب لينوفو للالعاب" Celeron laptops, stands and a bag)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app import agent, db, llm, main
from app.query import english_words, is_accessory, misses_feature, other_variant

P = agent.Product


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def test_requests_understood_by_words():
    assert english_words("سماعة جي بي ال") == "earbuds jbl"
    assert english_words("لابتوب لينوفو للالعاب") == "laptop lenovo gaming"
    assert english_words("شاشة ال جي 55 بوصة") == "tv lg 55 inch"
    assert english_words("لاب توب اتش بي") == "laptop hp"
    assert english_words(agent._without_budget("سماعات بلوتوث تحت 2000 جنيه")) == "earbuds bluetooth"


def _ok(eng, *titles):
    return [agent._matches_request(P("X", t, 100, f"u{i}"), eng) for i, t in enumerate(titles)]


def test_brand_kind_and_accessories():
    assert _ok("earbuds jbl", "Earbuds Cleaning Kit for AirPods, Laptop Cleaning Pen", "M10 TWS Wireless Earbuds",
               "JBL Flip 6 Portable Speaker", "JBL Wave Beam True Wireless Earbuds with Charging Case",
               "JBL Tune 520BT Wireless On-Ear Headphones") == [False, False, False, True, True]
    assert _ok("laptop lenovo", "Lenovo Laptop Stand Adjustable", "Lenovo B210 Laptop Backpack 15.6",
               "حقيبة لابتوب لينوفو", "Lenovo IdeaPad Slim 3 Laptop i5 with Backpack") == [False, False, False, True]
    assert _ok("vacuum bosch", "Bosch Vacuum Cleaner 2000W") == [True]  # a vacuum cleaner is not a "cleaner" part
    # asked for the accessory: it's what they want
    assert _ok("bag laptop lenovo", "Lenovo B210 Laptop Backpack 15.6") == [True]


def test_features_and_model_variants():
    assert _ok("laptop lenovo gaming", "Lenovo IdeaPad 1 Laptop Intel Celeron N4500",
               "Lenovo LOQ 15IAX9 Gaming Laptop i5 RTX 3050") == [False, True]
    assert misses_feature("Sharp Split Air Conditioner 1.5 HP Cool Only", "air conditioner sharp inverter")
    assert not misses_feature("شارب تكييف انفرتر 1.5 حصان", "air conditioner sharp inverter")
    assert _ok("realme 12 pro", "Realme 12 Pro 5G 256GB 12GB", "Realme 12 Pro+ 5G 512GB", "Realme 12 5G 256GB") == \
        [True, False, False]
    assert other_variant("Apple iPhone 16 Pro 256GB", "iphone 16 pro max")
    assert not other_variant("Apple iPhone 16 Pro Max 256GB", "iphone 16 pro max")
    assert not other_variant("Samsung 55 Inch Crystal UHD TV", "tv samsung 55")  # a size, not a model variant
    assert not is_accessory("Samsung Galaxy A15 128GB with charger")


def test_price_list_never_fills_places_with_accessories():
    db.save_catalog([
        {"url": "https://www.amazon.eg/dp/KIT", "store": "Amazon.eg",
         "title": "Earbuds Cleaning Kit, Cleaner Pen for JBL earbuds", "price": 120},
        {"url": "https://www.amazon.eg/dp/M10", "store": "Amazon.eg", "title": "M10 TWS Wireless Earbuds", "price": 150},
        {"url": "https://www.jumia.com.eg/jbl.html", "store": "Jumia", "title": "JBL Tune Flex wireless earbuds",
         "price": 2600},
    ], "EG", "phones")
    got = agent._shelf_offers("سماعة جي بي ال", "سماعة جي بي ال")
    assert [p.price for p in got] == [2600]


def test_live_check_can_ask_for_the_no_model_answer(monkeypatch):
    db.save_catalog([{"url": "https://www.jumia.com.eg/jbl.html", "store": "Jumia",
                      "title": "JBL Tune Flex wireless earbuds", "price": 2600}], "EG", "phones")
    planned = []

    async def plan(*a, **kw):
        planned.append(1)
        raise AssertionError("no planner")

    async def checked(picks, options="", patient=False, quick=False, use_llm=True):
        for p in picks:
            p.verified = True
        return picks, [], 0

    async def no_store(queries):
        return [], {}
    monkeypatch.setenv("CRON_SECRET", "s3")
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(agent, "plan_turn", plan)
    monkeypatch.setattr(agent, "shopper_hits", plan)
    monkeypatch.setattr(agent, "checked_only", checked)
    monkeypatch.setattr(agent, "_store_search", no_store)
    monkeypatch.setattr(agent.websearch, "available", lambda: True)
    c = TestClient(main.app)
    body = {"messages": [{"role": "user", "text": "سماعة جي بي ال"}]}
    lines = c.post("/api/chat/stream", json=body, headers={"x-cron-secret": "s3", "x-no-llm": "1"}).text.splitlines()
    import json
    done = next(json.loads(x)["data"] for x in lines if '"done"' in x)
    assert not planned and done["outage"] and [r["price"] for r in done["results"]] == [2600]
    assert done["no_llm_funnel"]["understood"] == "earbuds jbl" and done["no_llm_funnel"]["checked"] == 1
    # without the secret the header does nothing


def test_things_made_for_the_product_and_other_models():
    # live 2026-11-26: "ايفون 15" showed a selfie stick, a desk stand and a power bank; "ريلمي 12 برو" a C85 Pro
    assert _ok("iphone 15", "Mini Selfie Stick Tripod with Wireless Remote, Compatible with iPhone 15",
               "3-in-1 Magnetic Phone & Tablet Desk Stand for iPhone 15",
               "Anker Laptop Power Bank (25,000 mAh, 165W) for iPhone 15, MacBook",
               "Apple iPhone 15 , 128 GB , 6 GB , 5G , Single SIM - Black") == [False, False, False, True]
    assert _ok("realme 12 pro", "Realme C85 Pro , 256 GB , 8 GB") == [False]
    assert agent._why_not(P("X", "Realme C85 Pro", 1, "u"), "realme 12 pro") == "model"
