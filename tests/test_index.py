"""Phase B of the index-first plan: the price index grows from searches, the LLM understands its titles, and each
store's product page is read with a recipe the LLM learned once (no LLM call per page)."""
import asyncio
import json
import time

import pytest

from app import agent, catalog, db, llm, recipes


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def _page(price, sold=False, related=12999, title="Fresh FTM-8S 8KG"):
    return f"""<html><body><header><div class="cart">Cart</div></header>
<main><div id="product-main"><h1 class="product-title">{title}</h1>
<div class="price-box"><span class="price current-price">EGP {price:,}</span><del class="old-price">EGP 19,999</del></div>
<div class="installment">or {price // 12:,} EGP / month</div>
{'<div class="stock-msg out">Out of stock</div>' if sold else '<button class="btn add-to-cart" type="button">Add to cart</button>'}
</div>
<section class="related"><div class="card"><span class="price">EGP {related:,}</span>
<button class="btn add-to-cart">Add to cart</button></div></section></main></body></html>"""


PRICE_SEL = "div#product-main div.price-box span.price.current-price"
BUY_SEL = "div#product-main button.btn.add-to-cart"


def _llm(monkeypatch, calls, price_reads):
    """learn_page: picks the main price + buy button; read_price: reads the main price like the real model."""
    async def fake_ask(prompt, step="other", **kw):
        calls.append(step)
        if step == "learn_page":
            assert PRICE_SEL in prompt and "EGP 12,999" not in prompt.split("BUY AREA")[0]
            return {"price": PRICE_SEL, "buy": BUY_SEL, "sold_out": None}
        return {"same_product": True, "price": price_reads.pop(0), "currency": "EGP", "in_stock": True}
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)


def test_a_store_recipe_is_learned_confirmed_then_replaces_the_llm(monkeypatch):
    monkeypatch.setattr(recipes, "ENABLED", True)
    monkeypatch.setattr(agent.buybox, "read", lambda html, url: None)  # (this checks the recipes alone)
    monkeypatch.setattr(agent.pricemodel, "read", lambda html, url: None)
    calls = []
    prices = [15999, 15499, 14999, 13999, 16999]
    _llm(monkeypatch, calls, list(prices))
    url = "https://shop.example.com/p/{}"

    async def read(i, price):
        got = await agent._read_main_price(url.format(i), _page(price), "Fresh FTM-8S")
        await asyncio.gather(*agent._BACKGROUND)
        return got

    async def go():
        out = []
        for i, price in enumerate(prices):
            out.append(await read(i, price))
        return out
    got = asyncio.run(go())
    assert [p.price for p in got] == prices
    # 1st page: read + learned (the page leaves one choice: no LLM call to learn); 2nd, 3rd: read by the LLM and
    # the recipe agreed; then trusted: no LLM call
    assert calls == ["read_price", "read_price", "read_price"]
    row = db.get_recipe("shop.example.com")
    assert row["ok"] == 3 and row["uses"] == 2
    assert recipes.apply(json.loads(row["recipe"]), _page(12345)) == (12345, True)


def test_a_recipe_that_stops_working_is_learned_again(monkeypatch):
    monkeypatch.setattr(recipes, "ENABLED", True)
    monkeypatch.setattr(agent.pricemodel, "read", lambda html, url: None)
    db.save_recipe("shop.example.com", json.dumps({"price": "span.gone", "buy": BUY_SEL, "sold_out": None}))
    for _ in range(3):
        db.recipe_result("shop.example.com", True)
    calls = []
    _llm(monkeypatch, calls, [15999, 15999, 15999])

    async def go():
        for i in range(3):
            assert (await agent._read_main_price(f"https://shop.example.com/p/{i}", _page(15999), "t")).price == 15999
            await asyncio.gather(*agent._BACKGROUND)
    asyncio.run(go())
    # the recipe found nothing (twice on the first page): the LLM reads, then learns the store anew; the new
    # recipe is checked against the LLM again before it is trusted
    assert calls == ["read_price", "read_price", "read_price"]
    assert recipes.apply(json.loads(db.get_recipe("shop.example.com")["recipe"]), _page(12345)) == (12345, True)


def test_a_recipe_never_says_in_stock_without_the_buy_button():
    recipe = {"price": PRICE_SEL, "buy": BUY_SEL, "sold_out": None}
    assert recipes.apply(recipe, _page(15999)) == (15999, True)
    assert recipes.apply(recipe, _page(15999, sold=True)) == (15999, None)  # -> the LLM reads it


def test_offers_a_search_checked_go_into_the_index():
    results = [
        {"url": "https://smallshop.com/p/1", "store": "smallshop.com", "title": "Fresh FTM-8S", "price": 15499,
         "verified": True, "trust_score": 70},
        {"url": "https://shady.com/p/1", "store": "shady.com", "title": "Fresh FTM-8S", "price": 9000,
         "verified": True, "trust_score": 20},
        {"url": "https://other.com/p/1", "store": "other.com", "title": "Fresh FTM-8S", "price": 15000,
         "verified": False, "trust_score": 90}]
    assert catalog.remember_found(results) == 1
    got = catalog.offers(["ftm-8s"])
    assert [(p.store, p.trusted, p.trust_score) for p in got] == [("smallshop.com", False, 70)]
    # a big store's page found again by a search stays a big store's page in its category
    db.save_catalog([{"url": "https://www.jumia.com.eg/fresh.html", "store": "Jumia", "title": "Fresh FTM-8S",
                      "price": 16000}], "EG", "washers")
    catalog.remember_found([{"url": "https://www.jumia.com.eg/fresh.html", "store": "Jumia", "title": "Fresh FTM-8S",
                             "price": 15800, "verified": True, "trusted": True}])
    row = next(r for r in db.catalog_like(["fresh"], "EG", time.time() - 60) if "jumia" in r["url"])
    assert row["category"] == "washers" and row["trust"] is None and row["price"] == 15800
    assert all(r["category"] != "found" for r in db.catalog_since("EG", 0))  # deals: big stores only


def test_the_llm_understands_titles_so_arabic_words_find_them(monkeypatch):
    db.save_catalog([{"url": f"https://www.amazon.eg/dp/{i}", "store": "Amazon.eg", "title": t, "price": 15000 + i}
                     for i, t in enumerate(["Fresh Top Load 8KG FTM-8S", "Samsung Galaxy A56 256GB"])], "EG", "x")

    async def fake_ask(prompt, step="", **kw):
        assert step == "understand"
        i = next(int(line.split(".")[0]) for line in prompt.splitlines() if "FTM-8S" in line and line[:1].isdigit())
        return {"items": [{"i": i, "brand": "Fresh", "model": "FTM-8S", "kind": "washing machine", "kind_ar": "غسالة",
                           "size": "8 kg", "specs": ["top load", "automatic"], "accessory": False}]}  # 1 skipped
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    got = asyncio.run(catalog.understand())
    assert got == {"understood": 1, "not_understood": 1, "asked": 2}
    assert [p.title for p in catalog.offers(["غسالة"])] == ["Fresh Top Load 8KG FTM-8S"]
    assert db.catalog_to_understand("EG", 10) == []  # the skipped one isn't asked again and again


def _understood_index():
    rows = [("https://www.jumia.com.eg/fresh-8.html", "Jumia", "Fresh Automatic Top Load Washer 8KG FTM-8S", 15999,
             {"brand": "Fresh", "kind": "washing machine", "kind_ar": "غسالة", "size": "8 kg"}),
            ("https://www.amazon.eg/dp/F8", "Amazon.eg", "Fresh washing machine 8 kilo top load", 15499,
             {"brand": "Fresh", "kind": "washing machine", "kind_ar": "غسالة", "size": "8 kg"}),
            ("https://www.amazon.eg/dp/F10", "Amazon.eg", "Fresh washing machine 10 KG", 17999,
             {"brand": "Fresh", "kind": "washing machine", "size": "10 kg"}),
            ("https://www.amazon.eg/dp/T8", "Amazon.eg", "Toshiba washing machine 8 KG", 14000,
             {"brand": "Toshiba", "kind": "washing machine", "size": "8 kg"}),
            ("https://www.amazon.eg/dp/COVER", "Amazon.eg", "Fresh 8 kg washing machine cover", 300,
             {"brand": "Fresh", "kind": "washing machine cover", "size": "8 kg", "accessory": True})]
    db.save_catalog([{"url": u, "store": s, "title": t, "price": p} for u, s, t, p, _ in rows], "EG", "washers")
    db.save_understood([(u, json.dumps(a), catalog._words(a)) for u, _, _, _, a in rows])


def test_the_index_finds_what_the_shopper_asked_for_without_any_llm_call():
    _understood_index()
    plan = {"product": {"kind": "washing machine", "brand": "Fresh", "size": "8 kg"}}
    got = agent._index_offers(plan)
    assert [p.price for p in got] == [15499, 15999]  # not the 10 kg, not Toshiba, not the cover
    assert all(p.trusted for p in got)
    assert [p.price for p in agent._index_offers(plan, maxp=15600)] == [15499]
    assert [p.store for p in agent._index_offers({"product": {"kind": "washing machine", "model": "FTM-8S"}})] == ["Jumia"]
    assert agent._index_offers({"product": None}) == []


def test_the_index_offers_are_shown_first_unless_the_judged_ones_came_already(monkeypatch):
    _understood_index()
    sent = []

    async def checked(picks, options="", patient=False, quick=False):
        return [p for p in picks if p.price < 15600], [], 0
    monkeypatch.setattr(agent, "checked_only", checked)
    found = agent._index_offers({"product": {"kind": "washing machine", "brand": "Fresh", "size": "8 kg"}})

    async def go(judged):
        token = agent.PARTIAL.set(sent.append)
        try:
            return await agent._index_first(found, "", "غسالة فريش 8 كيلو", {"request": "غسالة فريش 8 كيلو"}, judged)
        finally:
            agent.PARTIAL.reset(token)
    shown = asyncio.run(go({}))
    assert [p.price for p in shown] == [15499] and sent[0]["results"][0]["price"] == 15499
    assert found[0].verified is None  # the judge later gets the offers as they were, not the checked copies
    sent.clear()
    asyncio.run(go({"sent": True}))
    assert sent == []


def test_the_same_first_message_is_planned_once_a_day(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main, websearch
    plans = []

    async def fake_plan(messages, current="", shown=None, price=(None, None), hits=None, hits_for=""):
        plans.append(messages[-1]["text"])
        return {"action": "search", "request": "غسالة فريش 8 كيلو", "reply": "هدورلك", "understood": "هدورلك",
                "suggestions": []}

    async def fake_run(q, plan=None, budget_text=None):
        return {"mode": "llm", "results": [], "request": q, "summary": ""}
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(websearch, "available", lambda: False)
    monkeypatch.setattr(agent, "plan_turn", fake_plan)
    monkeypatch.setattr(agent, "run", fake_run)
    c = TestClient(main.app)
    for text in ("غسالة فريش 8 كيلو", "غسالة  فريش 8 كيلو", "غسالة فريش 10 كيلو"):
        assert c.post("/api/chat", json={"messages": [{"role": "user", "text": text}]}).status_code == 200
    assert plans == ["غسالة فريش 8 كيلو", "غسالة فريش 10 كيلو"]
    # a follow-up in a chat is always planned (it depends on what came before)
    c.post("/api/chat", json={"messages": [{"role": "user", "text": "غسالة فريش 8 كيلو"},
                                           {"role": "assistant", "text": "x"}, {"role": "user", "text": "أرخص"}]})
    assert len(plans) == 3


def test_the_best_matching_products_come_first_however_big_the_index():
    """Live 2026-11-24: with 6000 products, "samsung" matched TVs and fridges too, and a newest-first cut of 300
    rows left 2 offers for "Samsung A56 256GB"."""
    db.save_catalog([{"url": "https://www.jumia.com.eg/a56.html", "store": "Jumia", "title": "Samsung Galaxy A56 256GB",
                      "price": 19000}], "EG", "phones")
    db.save_catalog([{"url": f"https://www.amazon.eg/dp/TV{i}", "store": "Amazon.eg", "title": f"Samsung Smart TV {i}",
                      "price": 9000 + i} for i in range(30)], "EG", "tvs")
    rows = db.catalog_like(["a56", "256gb", "samsung"], "EG", time.time() - 60, limit=5)
    assert rows[0]["title"] == "Samsung Galaxy A56 256GB"


def test_background_llm_work_waits_while_a_shopper_searches(monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    order = []

    async def fake(prompt, *a, **kw):
        order.append(prompt)
        return {}
    monkeypatch.setattr(llm, "_ask_json", fake)

    async def go():
        async def shopper():
            with llm.live():
                await asyncio.sleep(1.5)
                await llm.ask_json("shopper")

        async def cron():
            llm.background()
            await asyncio.sleep(0.1)
            await llm.ask_json("cron")
        await asyncio.gather(shopper(), cron())
    asyncio.run(go())
    assert order == ["shopper", "cron"]


def test_a_cut_off_answer_leaves_the_titles_for_the_next_run(monkeypatch):
    db.save_catalog([{"url": f"https://www.amazon.eg/dp/{i}", "store": "Amazon.eg", "title": f"Phone {i}",
                      "price": 1000 + i} for i in range(10)], "EG", "phones")

    async def fake_ask(prompt, step="", **kw):
        return {"items": [{"i": 0, "kind": "smartphone"}]}  # 1 of 10: the answer was cut
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    asyncio.run(catalog.understand())
    assert len(db.catalog_to_understand("EG", 50)) == 9
