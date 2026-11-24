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
    # 1st page: read + learned; 2nd, 3rd: read by the LLM and the recipe agreed; then trusted: no LLM call
    assert calls == ["read_price", "learn_page", "read_price", "read_price"]
    row = db.get_recipe("shop.example.com")
    assert row["ok"] == 3 and row["uses"] == 2 and json.loads(row["recipe"])["price"] == PRICE_SEL


def test_a_recipe_that_stops_working_is_learned_again(monkeypatch):
    monkeypatch.setattr(recipes, "ENABLED", True)
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
    assert calls == ["read_price", "read_price", "learn_page", "read_price"]
    assert json.loads(db.get_recipe("shop.example.com")["recipe"])["price"] == PRICE_SEL


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
