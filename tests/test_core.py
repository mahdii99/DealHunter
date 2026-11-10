import asyncio
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import agent, llm
from app.extract import extract_products
from app.query import parse_query
from app.ranking import rank
from app.stores import BTech, Jumia

FIX = Path(__file__).parent / "fixtures"



@pytest.fixture(autouse=True)
def _llm_state_in_tmp(tmp_path, monkeypatch):
    from app import llm
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    from app import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    yield
    db.close()

def test_parse_arabic_query():
    q = parse_query("عايز باور بانك 10000 مللي تحت 1000 جنيه")
    assert q.keywords == ["power", "bank"]
    assert q.must_have == ["10000mah"]
    assert q.max_price == 1000


def test_jumia_parser():
    items = Jumia().parse((FIX / "jumia.html").read_text(encoding="utf-8"))
    assert len(items) == 4 and items[0].price == 1150 and items[0].old_price == 1400


def test_btech_real_page_generic_extraction():
    items = BTech().parse((FIX / "btech.html").read_text(encoding="utf-8"))
    assert len(items) >= 15
    wiwu = next(p for p in items if p.title.startswith("Wiwu Wi-P032"))
    assert wiwu.price == 399 and wiwu.coupon_price == 349 and wiwu.old_price is None
    assert wiwu.url == "https://btech.com/en/p/wiwu-wi-p032-wired-power-bank-10000-mah-2-ports-black"
    assert wiwu.image.startswith("https://media.btech.com/catalogs/8/f/0/c/8f0c6156")


def test_generic_extraction_jsonld():
    html = '<script type="application/ld+json">{"@type":"ItemList","itemListElement":[{"@type":"Product",' \
           '"name":"Test Power Bank 10000mAh","url":"/p/1","offers":{"price":"450.00"}}]}</script>'
    [p] = extract_products(html, "X", "https://x.com")
    assert (p.price, p.url) == (450, "https://x.com/p/1")


def test_rules_ranking_on_real_btech_page():
    products = BTech().parse((FIX / "btech.html").read_text(encoding="utf-8"))
    ranked = rank(products, parse_query("باور بانك 10000 مللي تحت 1000"))
    assert ranked and all("10000" in p.title.replace(",", "") and p.price <= 1000 for p in ranked)
    assert not any("Earbuds" in p.title or "Watch" in p.title for p in ranked)
    assert (ranked[0].price, ranked[0].coupon_price) == (399, 349)


def test_agent_with_fake_llm(monkeypatch):
    page = (FIX / "btech.html").read_text(encoding="utf-8")
    store = agent.ALL_STORES[0]

    async def fake_search(client, text):
        return store.parse(page)

    replies = iter([
        {"understood": "باور بانك 10000", "queries": ["power bank 10000mah"],
         "requirements": ["10000 mAh"], "avoid": ["bundles"], "max_price": 900, "min_price": None},
        {"matches": [{"id": 1, "note": "تلات منافذ"}, {"id": 0, "note": "أرخص واحد"}], "summary": "خد الـ Wiwu"},
    ])

    async def fake_ask(prompt, timeout=30, prefer=None, spread=0, budget=None):
        return next(replies)

    monkeypatch.setattr(store, "search", fake_search)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    out = asyncio.run(agent.run("باور بانك 10000 تحت 900"))
    assert out["mode"] == "llm" and out["summary"] == "خد الـ Wiwu"
    eff = [r["coupon_price"] or r["price"] for r in out["results"]]
    assert eff == sorted(eff)
    assert all(r["price"] <= 900 for r in out["results"]) and out["results"][0]["note"] == "أرخص واحد"


def test_llm_skips_busy_model_and_remembers(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm._COOLDOWN.clear()
    llm._LAST_GOOD.clear()
    calls = []

    async def fake_gemini(client, model, prompt):
        calls.append(model)
        if model == "gemini-flash-lite-latest":
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]})

    async def no_list(client, provider):
        return []

    monkeypatch.setattr(llm, "_list_models", no_list)
    monkeypatch.setattr(llm, "_gemini", fake_gemini)
    assert asyncio.run(llm.ask_json("x")) == {"ok": 1}
    assert calls == ["gemini-flash-lite-latest", "gemini-flash-latest"]
    calls.clear()
    assert asyncio.run(llm.ask_json("y")) == {"ok": 1}
    assert calls == ["gemini-flash-latest"]  # busy one is skipped, last good one goes first


def test_llm_falls_back_to_groq_with_listed_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    llm._MODEL_CACHE.clear()
    llm._COOLDOWN.clear()
    llm._LAST_GOOD.clear()

    async def fake_list(client, provider):
        return ["gemini-3-flash", "gemini-3-flash-image"] if provider == "gemini" else ["whisper-large-v3", "new-llama-x"]

    async def busy_gemini(client, model, prompt):
        return httpx.Response(429, json={"error": "quota"})

    used = []

    async def fake_groq(client, model, prompt):
        used.append(model)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 2}'}}]})

    async def no_sleep(_):
        return None

    monkeypatch.setattr(llm, "_list_models", fake_list)
    monkeypatch.setattr(llm, "_gemini", busy_gemini)
    monkeypatch.setattr(llm, "_groq", fake_groq)
    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)
    assert asyncio.run(llm.ask_json("x")) == {"ok": 2}
    assert used == ["new-llama-x"]


def test_image_proxy_rejects_other_hosts():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    assert c.get("/img", params={"u": "https://evil.example.com/x.jpg"}).status_code == 400
    assert c.get("/img", params={"u": "http://media.btech.com/x.jpg"}).status_code == 400


def test_agent_web_mode_finds_stores_itself(monkeypatch):
    from app import trust, websearch
    page = (FIX / "btech.html").read_text(encoding="utf-8")

    searched = []

    async def fake_search(client, query, max_results=8, exclude_domains=None, include_domains=None):
        searched.append("famous" if include_domains else "other" if exclude_domains else "check: " + query)
        return [{"url": "https://btech.com/en/s?q=power+bank", "title": "", "snippet": ""},
                {"url": "https://www.youtube.com/watch?v=x", "title": "", "snippet": ""},
                {"url": "https://unknown-shop.com/p/1", "title": "", "snippet": ""}]

    async def fake_fetch(client, sem, url):
        if "btech" in url:
            ps = extract_products(page, trust.store_name(url), "https://btech.com", "https://btech.com/en/p/{slug}")
            for p in ps:
                p.store, p.trusted = trust.store_name(p.url), trust.is_known(p.url)
            return ps, page
        raise RuntimeError("blocked")

    replies = iter([
        {"understood": "x", "web_queries": ["power bank 10000mah price egypt"], "queries": ["power bank"],
         "requirements": [], "avoid": [], "max_price": None, "min_price": None},
        {"matches": [{"id": 0, "note": "ok"}], "summary": "s"},
    ])

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        if "Use these results to check" in prompt:  # the plan check sees the web results for the shopper's words
            assert "power bank 10000mah price egypt" in prompt
            return {"understood": "باور بانك", "web_queries": ["سعر باور بانك 10000", "power bank 10000mAh egypt"],
                    "model_codes": [], "requirements": ["10000 mAh"], "changed": True}
        return next(replies)

    monkeypatch.setattr(websearch, "available", lambda: True)
    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(agent, "_fetch_page", fake_fetch)

    async def direct_must_not_run(queries):
        raise AssertionError("no store-specific search in web mode")

    monkeypatch.setattr(agent, "_search_all", direct_must_not_run)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    out = asyncio.run(agent.run("باور بانك"))
    assert out["source"] == "web"
    assert out["results"][0]["store"] == "B.TECH" and out["results"][0]["trusted"] is True
    assert "unknown-shop.com" in out["store_errors"] and "youtube.com" not in out["store_errors"]
    # one check search of the shopper's own words, then each checked query on famous stores and on the other shops
    assert searched[0] == "check: باور بانك" and sorted(searched[1:]) == ["famous", "famous", "other", "other"]
    assert out["queries"] == ["سعر باور بانك 10000", "power bank 10000mAh egypt"]


def test_trust_helpers():
    from app import trust
    assert trust.store_name("https://www.amazon.eg/dp/X") == "Amazon.eg" and trust.is_known("https://fakejumia.com.eg/x") is False
    assert trust.is_known("https://www.jumia.com.eg/x") and not trust.is_shop_candidate("https://m.youtube.com/x")


def test_shopify_style_listing_cents_link_and_condition():
    html = """<html><body>
    <div class="card"><a href="/products/hp-zbook-fury-15-g7">HP ZBook Fury 15 G7 | 15.6 FHD</a>
      <span class="price">LE 36,499.00</span></div>
    <div class="card"><a href="/products/hp-zbook-14u-g6">HP ZBook 14u G6</a><span>LE 19,999.00</span></div>
    <script>var meta = {"products":[
      {"id": 111, "handle": "hp-zbook-fury-15-g7", "title": "HP ZBook Fury 15 G7 | 15.6 FHD", "price": 3649900,
       "tags": ["Pre-Owned", "HP"]},
      {"id": 222, "title": "HP ZBook 14u G6", "price": 1999900, "product_type": "Laptop"}
    ]};</script></body></html>"""
    items = {p.title: p for p in extract_products(html, "montag", "https://shop.example/collections/zbook")}
    fury, g6 = items["HP ZBook Fury 15 G7 | 15.6 FHD"], items["HP ZBook 14u G6"]
    assert fury.price == 36499 and g6.price == 19999
    assert fury.url == "https://shop.example/products/hp-zbook-fury-15-g7"
    assert g6.url == "https://shop.example/products/hp-zbook-14u-g6"  # found via the page's own link
    assert fury.condition == "used" and g6.condition is None


def test_variants_collapse_and_cents_outlier():
    from app.models import Product
    u = "https://s.example/products/x"
    ps = [Product("s", "X", 40499, u), Product("s", "X", 4349900, u), Product("s", "X", 51999, u),
          Product("s", "Y", 1.0, "https://s.example/products/y")]
    out = agent._cheapest_variant(ps)
    assert [(p.title, p.price) for p in out] == [("X", 40499)]


def test_llm_reads_page_without_structured_data(monkeypatch):
    html = '<html><body><div><a href="/dp/B01">HP ZBook Firefly 14 G10</a> EGP 52,999</div></body></html>'

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        assert "HP ZBook Firefly 14 G10 [link: /dp/B01]" in prompt and "52,999" in prompt
        return {"products": [{"title": "HP ZBook Firefly 14 G10", "price": 52999, "url": "/dp/B01", "condition": "new"},
                             {"title": "junk", "price": "n/a"}]}

    monkeypatch.setattr(llm, "ask_json", fake_ask)
    [p] = asyncio.run(agent._llm_read_page("https://www.amazon.eg/s?k=zbook", html, "hp zbook"))
    assert (p.price, p.url, p.store, p.trusted) == (52999, "https://www.amazon.eg/dp/B01", "Amazon.eg", True)


def test_guessed_budget_is_ignored(monkeypatch):
    page = (FIX / "btech.html").read_text(encoding="utf-8")
    store = agent.ALL_STORES[0]

    async def fake_search(client, text):
        return store.parse(page)

    replies = iter([
        {"understood": "x", "queries": ["power bank"], "requirements": [], "avoid": [], "max_price": 300, "min_price": None},
        {"matches": [{"id": 0, "note": "n"}], "summary": "s"},
    ])

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        return next(replies)

    monkeypatch.setattr(store, "search", fake_search)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    out = asyncio.run(agent.run("عايز باور بانك سعره حلو"))
    assert out["candidates_checked"] > 0 and out["max_price"] is None and out["results"]


def test_small_shop_trust_signals():
    from app import trust
    good = """<html><body><p>Hotline: 0100 123 4567</p><p>الدفع عند الاستلام متاح</p>
      <a href="/pages/return-policy">Return Policy</a><p>ضمان سنة</p><p>12 شارع عباس العقاد، مدينة نصر، القاهرة</p>
      <a href="https://facebook.com/smallshop">fb</a></body></html>"""
    score, signals = trust.page_signals(good, "https://smallshop.com.eg/p/1", True)
    assert score >= 80 and "دفع عند الاستلام" in signals and "رقم تليفون مصري" in signals
    shady, sig2 = trust.page_signals("<p>High copy master copy watch, order now</p>", "http://cheapx.shop/p", False)
    assert shady < 20 and any("كوبي" in x for x in sig2)
    assert trust.page_signals("", "https://www.amazon.eg/dp/1", False)[0] == 100


def test_discount_percent():
    from app.models import Product
    assert agent._discount(Product("s", "x", 750, "u", old_price=1000)) == 25
    assert agent._discount(Product("s", "x", 399, "u", coupon_price=349)) == 13
    assert agent._discount(Product("s", "x", 500, "u")) is None


def test_budget_needs_price_words():
    assert not agent._states_budget("مكواه باناسونيك 2200 وات ياباني بخار 650")
    assert not agent._states_budget("عايز موبايل سعره حلو")
    assert agent._states_budget("باور بانك تحت 700")
    assert agent._states_budget("سماعة في حدود 1500 جنيه")
    assert agent._states_budget("laptop under 30000")


def test_sold_out_detection():
    page = """<html><head><script type="application/ld+json">{"@type":"Product","name":"سخان مياه غاز فريش، 6 لتر، غاز طبيعي",
      "offers":{"price":"3330","priceCurrency":"EGP","availability":"https://schema.org/InStock"}}</script></head>
      <body><h1>سخان مياه غاز فريش، 6 لتر، غاز طبيعي</h1><span>3,330 جنيه</span><button>غير متوفر</button>
      <a>اعلمني</a><div class="related">Other heater 2,000 Add to cart</div></body></html>"""
    [p] = extract_products(page, "elghazawy", "https://elghazawy.com/p/1")
    assert p.in_stock is False  # the buy box wins over the page data that still says InStock

    data_oos = '<script type="application/ld+json">{"@type":"Product","name":"Olympic Hero Plus 50L",' \
               '"offers":{"price":"3900","availability":"http://schema.org/OutOfStock"}}</script><h1>Olympic Hero Plus 50L</h1>'
    assert extract_products(data_oos, "e", "https://e.com/p")[0].in_stock is False

    listing = """<div class="product outofstock"><a href="/p/heater-a">Heater A</a> 1,000</div>
      <div class="product instock"><a href="/p/heater-b">Heater B</a> 1,200 <button>Add to cart</button></div>
      <script>var d={"items":[{"name":"Heater A","url":"/p/heater-a","price":1000},
      {"name":"Heater B","url":"/p/heater-b","price":1200}]};</script>"""
    items = {p.title: p.in_stock for p in extract_products(listing, "s", "https://s.com/c/heaters")}
    assert items == {"Heater A": False, "Heater B": None}


def test_dedupe_same_item():
    from app.models import Product
    a = Product("H", "سخان أوليمبيك هيرو 50 لتر", 4270, "https://alhaythamgroup.com/en/product/x")
    b = Product("H", "سخان  أوليمبيك هيرو 50 لتر!", 4270, "https://alhaythamgroup.com/")
    c = Product("H", "سخان أوليمبيك هيرو 50 لتر", 4300, "https://alhaythamgroup.com/en/product/y")
    out = agent._dedupe([b, a, c])
    assert len(out) == 2 and any(p.url.endswith("/x") for p in out)


def test_small_price_never_divided_by_100():
    html = """<p>Hero 50L</p><span>4,399.00 EGP</span><span>rating 43.99</span>
      <script>var x={"items":[{"name":"Olympic Hero 50L heater","price":4399,"url":"/p/hero"}]};</script>"""
    [p] = extract_products(html, "s", "https://s.com/c")
    assert p.price == 4399


def test_chat_follow_up_refines_previous_request(monkeypatch):
    """'سامسونج بس' after a phone search: the planner sees the chat + current request and the search uses the
    merged request, and the budget guard checks what the shopper actually wrote."""
    page = (FIX / "btech.html").read_text(encoding="utf-8")
    store = agent.ALL_STORES[0]
    prompts = []

    async def fake_search(client, text):
        return store.parse(page)

    replies = iter([
        {"action": "search", "request": "موبايل سامسونج مش غالي بطاريته تكمل يومين", "need_based": True,
         "understood": "هدور على سامسونج بس", "suggested_products": ["Samsung Galaxy A16 (5000 mAh)"],
         "queries": ["samsung galaxy a16"], "requirements": ["Samsung", "battery 5000 mAh or more"],
         "avoid": [], "max_price": 7000, "min_price": None},
        {"matches": [{"id": 0, "note": "ok"}], "summary": "s"},
    ])

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(store, "search", fake_search)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    from app import main, websearch
    monkeypatch.setattr(websearch, "available", lambda: False)
    c = TestClient(main.app)
    r = c.post("/api/chat", json={
        "request": "موبايل مش غالي بطاريته تكمل يومين",
        "messages": [{"role": "user", "text": "عايز موبايل مش غالي بطاريته تكمل يومين"},
                     {"role": "assistant", "text": "Shown: 1) Redmi 14C - 5500 EGP (B.TECH)"},
                     {"role": "user", "text": "لا انا عايز سامسونج بس"}]})
    d = r.json()
    assert r.status_code == 200 and d["action"] == "search"
    assert d["request"] == "موبايل سامسونج مش غالي بطاريته تكمل يومين"
    assert d["suggested_products"] == ["Samsung Galaxy A16 (5000 mAh)"]
    assert d["max_price"] is None  # the shopper never wrote a budget, the planner guessed 7000
    assert "Current request (from the previous search): موبايل مش غالي" in prompts[0]
    assert prompts[0].rstrip().endswith("}") and "لا انا عايز سامسونج بس" in prompts[0]
    assert "Samsung Galaxy A16" in prompts[1]  # the judge knows which products fit the need


def test_chat_asks_when_too_vague(monkeypatch):
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        return {"action": "ask", "question": "الهدية لمين؟", "options": ["لراجل", "لست", 3]}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    from app import main
    d = TestClient(main.app).post("/api/chat", json={"messages": [{"role": "user", "text": "عايز هدية"}]}).json()
    assert d["action"] == "ask" and d["question"] == "الهدية لمين؟" and d["options"] == ["لراجل", "لست"]
    assert d["results"] == []


def test_chat_without_llm_appends_follow_up(monkeypatch):
    seen = []

    async def fake_rules(raw, note):
        seen.append(raw)
        return {"mode": "rules", "results": [], "request": raw, "store_errors": {}, "candidates_checked": 0}

    monkeypatch.setattr(llm, "available", lambda: False)
    monkeypatch.setattr(agent, "run_rules", fake_rules)
    from app import main
    d = TestClient(main.app).post("/api/chat", json={
        "request": "power bank", "messages": [{"role": "user", "text": "anker"}]}).json()
    assert seen == ["power bank anker"] and d["action"] == "search"


def test_reader_fallback_respects_robots_and_reads_blocked_pages(monkeypatch):
    from app import fetchers
    fetchers._ROBOTS.clear()
    monkeypatch.setenv("TAVILY_API_KEY", "t")

    def handler(request: httpx.Request):
        u = str(request.url)
        if "shop-d.com" in u and "r.jina.ai" not in u:
            raise httpx.ConnectTimeout("slow")                         # store doesn't answer us at all
        if u.endswith("/robots.txt"):
            if "r.jina.ai" in u:
                return httpx.Response(200, text="Title: robots\n\nUser-agent: *\nDisallow: /wp-admin/")
            if "shop-a" in u:
                return httpx.Response(200, text="User-agent: *\nDisallow: /cart")
            if "shop-b" in u:
                return httpx.Response(200, text="User-agent: *\nDisallow: /")
            return httpx.Response(404)
        if "api.tavily.com/extract" in u:
            return httpx.Response(200, json={"results": [
                {"url": "https://shop-a.com/p/heater", "raw_content": "Olympic Hero 50L heater EGP 3,629 " * 20}]})
        if "r.jina.ai" in u:
            return httpx.Response(200, text="Heater page text EGP 3,999 " * 20)
        return httpx.Response(500)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetchers.read_pages(client, ["https://shop-a.com/p/heater", "https://shop-b.com/p/heater",
                                                      "https://shop-c.com/p/heater", "https://shop-d.com/p/heater"])

    texts, why = asyncio.run(go())
    assert "EGP" in texts["https://shop-a.com/p/heater"]             # via Tavily Extract or Jina, whichever is first
    assert "3,999" in texts["https://shop-c.com/p/heater"]          # robots.txt missing -> allowed, via Jina
    assert "3,999" in texts["https://shop-d.com/p/heater"]          # robots.txt read through the reader
    assert why == {"https://shop-b.com/p/heater": "robots.txt disallows (or unreachable)"}


def test_foreign_store_pages_and_domains_are_skipped():
    from app import trust
    from app.extract import extract_products
    saudi = ('<script type="application/ld+json">{"@type":"ItemList","itemListElement":[{"@type":"Product",'
             '"name":"Midea Refrigerator 18 Cubic Feet Silver","url":"/p/1","offers":{"price":"1738"}}]}</script>'
             '<script>var c={"currency":"SAR"};</script>')
    assert extract_products(saudi, "shakersa.com", "https://shakersa.com/s") == []
    assert not trust.is_shop_candidate("https://www.eurohome.sa/fridge")
    assert not trust.is_shop_candidate("https://www.amazon.ae/x")
    assert not trust.is_shop_candidate("https://us.shein.com/x")
    assert trust.is_shop_candidate("https://myxprs.com/x") and trust.is_shop_candidate("https://www.2b.com.eg/x")


def test_cents_price_fixed_from_other_copy_of_same_product():
    from app.extract import extract_products
    html = ('<script>var meta={"products":[{"id":1,"handle":"lg-washer-f4x5","title":"LG Washer F4X5RRG23BN",'
            '"price":4699900}]};var x={"currency":"EGP"};</script>'
            '<script type="application/json">{"items":[{"handle":"lg-washer-f4x5","title":"LG Washer F4X5RRG23BN",'
            '"price":46999.0}]}</script><div>LG Washer</div>')
    [p] = extract_products(html, "myxprs.com", "https://myxprs.com/search")
    assert p.price == 46999


def test_daily_quota_rests_model_until_reset_and_survives_restart(monkeypatch):
    from app import llm
    llm._COOLDOWN.clear(); llm._LAST_GOOD.clear()
    r = httpx.Response(429, text='{"error": {"message": "Quota exceeded for metric: generate_content_free_tier_requests, '
                                 'limit: 20, model: gemini-2.5-flash", "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}')
    llm._rest("gemini-flash-latest", r)
    assert llm._COOLDOWN["gemini-flash-latest"] - time.time() > 600          # not just 3 minutes
    llm._rest("busy-model", httpx.Response(503, text="overloaded"))
    assert llm._COOLDOWN["busy-model"] - time.time() <= llm.BUSY_SECONDS + 1
    llm._COOLDOWN.clear()
    llm._load_state()                                                          # after a server restart
    assert "gemini-flash-latest" in llm._COOLDOWN


def test_price_db_cache_history_and_drop():
    from app import db
    key = db.cache_key("  Anker R50i NC ", "EG")
    assert key == db.cache_key("anker r50i nc", "EG") != db.cache_key("anker r50i nc", "SA")
    assert db.cached_search(key, 3600) is None
    db.save_search(key, "EG", {"results": [1]})
    assert db.cached_search(key, 3600) == {"results": [1]} and db.cached_search(key, -1) is None

    url = "https://shop.com/p/r50i"
    db.record_prices([{"url": url, "store": "shop.com", "title": "R50i", "price": 1350, "in_stock": True}], "EG")
    db.record_prices([{"url": url, "store": "shop.com", "title": "R50i", "price": 1350}], "EG")  # same price: not saved
    now = [{"url": url, "price": 1199}]
    db.add_price_context(now)
    assert now[0]["previous_price"] == 1350 and now[0]["lowest_seen"] == 1350
    db.record_prices([{"url": url, "store": "shop.com", "title": "R50i", "price": 1199}], "EG")
    assert [p["price"] for p in db.history(url)] == [1350, 1199]


def test_market_switch_changes_currency_and_local_domains(monkeypatch):
    from app import trust
    from app.extract import foreign_currency
    sar = '<script>{"priceCurrency":"SAR"}</script>'
    assert foreign_currency(sar) == "SAR" and not trust.is_shop_candidate("https://extra.sa/x")
    monkeypatch.setenv("MARKET", "SA")
    assert foreign_currency(sar) is None and trust.is_shop_candidate("https://extra.sa/x")
    assert not trust.is_shop_candidate("https://www.2b.com.eg/x")


def test_page_reads_spread_over_groq_models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    llm._MODEL_CACHE.clear(); llm._COOLDOWN.clear(); llm._LAST_GOOD.clear()

    async def fake_list(client, provider):
        return ["gemini-flash-latest"] if provider == "gemini" else ["model-a", "model-b", "model-c"]

    used = []

    async def fake_groq(client, model, prompt):
        used.append(model)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"products": []}'}}]})

    async def never_gemini(client, model, prompt):
        raise AssertionError("page reads should not spend Gemini quota while Groq answers")

    monkeypatch.setattr(llm, "_list_models", fake_list)
    monkeypatch.setattr(llm, "_groq", fake_groq)
    monkeypatch.setattr(llm, "_gemini", never_gemini)
    for i in range(3):
        llm._LAST_GOOD.clear()
        asyncio.run(llm.ask_json("page", prefer="bulk", spread=i))
    assert sorted(used) == ["model-a", "model-b", "model-c"]


def test_slow_pages_dont_hold_up_the_search():
    async def fast():
        return "ok"

    async def slow():
        await asyncio.sleep(5)
        return "late"

    async def boom():
        raise ValueError("bad")

    t = time.monotonic()
    res = asyncio.run(agent._within(0.2, [fast(), slow(), boom()]))
    assert time.monotonic() - t < 1
    assert res[0] == "ok" and isinstance(res[1], TimeoutError) and isinstance(res[2], ValueError)


def test_more_free_providers_and_fenced_json(monkeypatch):
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CEREBRAS_API_KEY", "c")
    llm._MODEL_CACHE.clear(); llm._COOLDOWN.clear(); llm._LAST_GOOD.clear()
    assert llm.available()
    seen = []

    def handler(request: httpx.Request):
        seen.append(str(request.url))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-oss-120b"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": 'Sure:\n```json\n{"ok": 3}\n```'}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(llm.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(llm.ask_json("x")) == {"ok": 3}
    assert seen[-1] == "https://api.cerebras.ai/v1/chat/completions"


def test_search_shares_free_quotas_and_falls_back(monkeypatch):
    from app import websearch
    for k in websearch.KEYS.values():
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setenv("EXA_API_KEY", "e")
    websearch._REST.clear()
    hits = []

    def handler(request: httpx.Request):
        hits.append(request.url.host)
        if request.url.host == "api.tavily.com":
            return httpx.Response(432, text="usage limit exceeded")
        body = request.read().decode()
        assert "includeDomains" in body and "btech.com" in body
        return httpx.Response(200, json={"results": [{"url": "https://btech.com/p/1", "title": "Hero 50L"}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return [await websearch.search(c, "hero 50l", include_domains=["btech.com"]) for _ in range(3)]

    res = asyncio.run(go())
    assert all(r == [{"url": "https://btech.com/p/1", "title": "Hero 50L", "snippet": ""}] for r in res)
    assert hits.count("api.tavily.com") == 1        # out of quota once -> rested, Exa takes over
    assert websearch._REST["tavily"] - time.time() > 3600


def test_openrouter_picks_big_free_models_and_rests_refused_ones():
    av = ["thinkingmachines/inkling-2-small:free", "liquid/lfm-3-1.7b:free", "openai/gpt-oss-120b:free",
          "meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen3-coder-480b:free"]
    picks = llm._pick(av, llm.COMPAT["openrouter"][2], None, "openrouter")
    assert picks[0] == "openrouter/free" and "liquid/lfm-3-1.7b:free" not in picks and "openai/gpt-oss-120b:free" in picks
    llm._COOLDOWN.clear()
    llm._rest("inkling", httpx.Response(403, text="only available on agentic harnesses"))
    assert llm._COOLDOWN["inkling"] - time.time() > 600


def test_chat_stream_sends_progress_then_the_answer(monkeypatch):
    import json as _json
    page = (FIX / "btech.html").read_text(encoding="utf-8")
    store = agent.ALL_STORES[0]

    async def fake_search(client, text):
        return store.parse(page)

    replies = iter([
        {"action": "search", "request": "باور بانك 10000", "understood": "هدورلك على باور بانك 10000",
         "queries": ["power bank 10000"], "requirements": [], "avoid": []},
        {"matches": [{"id": 0, "note": "ok"}], "summary": "s"},
    ])

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        return next(replies)

    monkeypatch.setattr(store, "search", fake_search)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    from app import main, websearch
    monkeypatch.setattr(websearch, "available", lambda: False)
    r = TestClient(main.app).post("/api/chat/stream", json={"request": "",
                                                            "messages": [{"role": "user", "text": "باور بانك 10000"}]})
    events = [_json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    statuses = [e["text"] for e in events if e["type"] == "status"]
    assert statuses[0] == "هدورلك على باور بانك 10000" and any("بقارن" in t for t in statuses)
    assert events[-1]["type"] == "done" and events[-1]["data"]["action"] == "search" and events[-1]["data"]["results"]


def test_reader_returns_what_it_has_when_time_is_up(monkeypatch):
    from app import fetchers
    fetchers._ROBOTS.clear()
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    async def handler(request: httpx.Request):
        u = str(request.url)
        if u.endswith("/robots.txt"):
            return httpx.Response(404)
        if "slow-shop" in u:
            await asyncio.sleep(5)
        return httpx.Response(200, text="Anker R50i NC EGP 1,599 " * 30)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetchers.read_pages(client, ["https://fast-shop.com/p", "https://slow-shop.com/p"], deadline=0.5)

    t = time.monotonic()
    texts, why = asyncio.run(go())
    assert time.monotonic() - t < 2
    assert list(texts) == ["https://fast-shop.com/p"] and why == {"https://slow-shop.com/p": "reader too slow"}


def test_question_about_results_is_answered_without_a_new_search(monkeypatch):
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        return {"action": "answer", "answer": "بي تك وكيل رسمي، بس السعر ده أقل بكتير من السوق فاتأكد من الموديل.",
                "request": "اول نتيجه دي اصليه؟"}

    async def no_search(*a, **k):
        raise AssertionError("a question about the results must not start a new search")

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    monkeypatch.setattr(agent, "run", no_search)
    from app import main
    r = TestClient(main.app).post("/api/chat", json={
        "request": "سماعة انكر R50i NC",
        "messages": [{"role": "user", "text": "anker r50i nc"},
                     {"role": "assistant", "text": "Shown: 1) R50i NC - 649 EGP (B.TECH)"},
                     {"role": "user", "text": "اول نتيجه دي مضمون انها اصليه؟"}]})
    d = r.json()
    assert d["action"] == "answer" and "بي تك" in d["answer"] and d["request"] == "سماعة انكر R50i NC"


def test_price_far_below_the_others_is_flagged():
    from app.models import Product
    picks = [Product(store="B.TECH", title="R50i NC", price=649, url="u1", best_deal=True)] + [
        Product(store=s, title="R50i NC", price=p, url=s) for s, p in [("Amazon", 1120), ("Raya", 1619), ("x", 1599)]]
    agent._flag_price_outliers(picks)
    assert picks[0].unconfirmed and not picks[0].best_deal and "أقل بكتير" in picks[0].note
    assert not any(p.unconfirmed for p in picks[1:])


def test_retired_model_is_replaced_by_another_listed_one(monkeypatch):
    """NVIDIA answers 410 for retired models it still lists: the next pick must skip it."""
    llm._COOLDOWN.clear()
    listed = ["openai/gpt-oss-120b", "meta/llama-3.3-70b-instruct", "meta/llama-3.1-405b-instruct",
              "nvidia/nemotron-nano-9b-v2"]
    picks = llm._pick(listed, llm.COMPAT["nvidia"][2], None, "nvidia")
    assert picks[0] == "openai/gpt-oss-120b" and "nvidia/nemotron-nano-9b-v2" not in picks
    llm._rest("openai/gpt-oss-120b", httpx.Response(410, text="end of life"))
    llm._rest("meta/llama-3.3-70b-instruct", httpx.Response(503, text="busy"))
    picks = llm._pick(listed, llm.COMPAT["nvidia"][2], None, "nvidia")
    assert "openai/gpt-oss-120b" not in picks  # retired: gone from the choices
    assert "meta/llama-3.3-70b-instruct" in picks and "meta/llama-3.1-405b-instruct" in picks  # busy is only rested
    llm._COOLDOWN.clear()


def test_dedupe_short_and_long_title_of_same_listing():
    from app.models import Product
    long = Product("A", "Soundcore by Anker R50i NC Wireless Bluetooth Earphones, Noise Cancelling, Black", 1130,
                   "https://www.amazon.eg/dp/B0D1")
    short = Product("A", "Soundcore by Anker R50i NC Wireless Bluetooth Earphones", 1130, "https://www.amazon.eg/s?k=r50i")
    other_price = Product("A", "Soundcore by Anker R50i NC Wireless Bluetooth Earphones", 1179, "https://www.amazon.eg/dp/B0D2")
    other_shop = Product("N", "Soundcore by Anker R50i NC Wireless Bluetooth Earphones", 1130, "https://www.noon.com/egypt-ar/x")
    out = agent._dedupe([short, long, other_price, other_shop])
    assert len(out) == 3 and all(any(p is q for q in out) for p in (long, other_price, other_shop))


def test_well_known_stores_come_from_the_plan_for_any_category():
    from app import trust
    plan = {"stores": ["https://www.defacto.com.eg/", "namshi.com", "not a domain", "zara.com/eg", "amazon.sa"]}
    sites = agent._famous_sites(plan)
    assert sites[:3] == agent.MARKETPLACES and "defacto.com.eg" in sites and "namshi.com" in sites
    assert "zara.com" in sites and "amazon.sa" not in sites and "not a domain" not in sites
    assert agent._famous_sites({}) == trust.FAMOUS_SITES  # nothing named: our own list


def test_planner_suggestions_never_carry_guessed_prices(monkeypatch):
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None):
        return {"action": "search", "request": "تيشرت", "suggested_products": [
            "Zara Men Black Crew Neck T-Shirt 100% Cotton – 199 EGP", "Samsung Galaxy A16 (5000 mAh)"]}
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    plan = asyncio.run(agent.plan_turn([{"role": "user", "text": "تيشرت ماركة"}]))
    assert plan["suggested_products"] == ["Zara Men Black Crew Neck T-Shirt 100% Cotton", "Samsung Galaxy A16 (5000 mAh)"]


def test_page_reading_goes_to_fast_groq_models_first(monkeypatch):
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(k, "k")
    for k in ("MISTRAL_API_KEY", "NVIDIA_API_KEY", "CLOUDFLARE_API_KEY", "CEREBRAS_API_KEY", "GROQ_MODEL"):
        monkeypatch.delenv(k, raising=False)
    llm._MODEL_CACHE.clear()
    llm._COOLDOWN.clear()
    llm._LAST_GOOD[:] = ["openrouter/free"]  # even when another model answered last time
    used = []

    async def fake_list(client, provider):
        return {"gemini": ["gemini-flash-lite-latest"], "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
                "openrouter": ["openrouter/free"]}[provider]

    async def answer(client, model, prompt):
        used.append(model)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    monkeypatch.setattr(llm, "_list_models", fake_list)
    monkeypatch.setattr(llm, "_groq", answer)
    monkeypatch.setattr(llm, "_caller", lambda p: answer)
    asyncio.run(llm.ask_json("x", prefer="bulk", spread=1))
    asyncio.run(llm.ask_json("x", prefer="bulk", spread=2))
    assert used and all(m.startswith("openai/") for m in used)  # Groq first, not the slow free models
    llm._LAST_GOOD.clear()


def test_sites_that_never_open_are_skipped_after_two_searches():
    from app import db
    from app.models import Product
    agent._remember_sites(["https://zara.com/eg/a", "https://btech.com/p/1"], [Product("B", "t", 10, "https://btech.com/p/1")],
                          {"zara.com": "403 / reader too slow"})
    assert db.blocked_sites() == set()  # one bad moment isn't enough
    agent._remember_sites(["https://zara.com/eg/b", "https://busy.com/x"], [], {"zara.com": "403", "busy.com": "LLM read failed: TimeoutError"})
    assert db.blocked_sites() == {"zara.com"}  # our busy LLM doesn't count against busy.com
    agent._remember_sites(["https://us-shop.com/x"], [], {"us-shop.com": "prices in USD"})
    assert db.blocked_sites() == {"zara.com", "us-shop.com"}
    assert db.unblock("zara.com") and db.blocked_sites() == {"us-shop.com"}
