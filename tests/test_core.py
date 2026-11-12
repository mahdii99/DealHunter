import asyncio
import os
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
    monkeypatch.delenv("DATABASE_URL", raising=False)  # .env may point at the real database
    if pg := os.getenv("TEST_DATABASE_URL"):  # also run against a real Postgres (an empty test database!)
        monkeypatch.setenv("DATABASE_URL", pg)
        for table in ("searches", "prices", "blocked_sites", "usage", "reports", "feedback"):
            db._db.run(f"DROP TABLE IF EXISTS {table}")
        db.close()
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

    async def fake_ask(prompt, timeout=30, prefer=None, spread=0, budget=None, **kw):
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


def test_image_proxy_only_public_addresses_and_shrinks_big_photos(monkeypatch):
    import io
    from fastapi.testclient import TestClient
    from PIL import Image
    from app import main
    ips = {"cdn.shop.com": "93.184.216.34", "inside.shop.com": "10.0.0.7", "metadata.local": "169.254.169.254"}

    async def fake_resolve(host):
        return [(0, 0, 0, "", (ips.get(host, "127.0.0.1"), 443))]

    big = io.BytesIO()
    Image.new("RGB", (3000, 2000), "red").save(big, "JPEG")
    calls = []

    async def fake_get(url, headers=None):
        calls.append(url)
        if url.endswith("/moved.jpg"):
            return httpx.Response(302, headers={"location": "https://metadata.local/x"}, request=httpx.Request("GET", url))
        return httpx.Response(200, content=big.getvalue(), headers={"content-type": "image/jpeg"},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(main, "_resolve", fake_resolve)
    monkeypatch.setattr(main._img_client, "get", fake_get)
    main._thumbs.clear()
    c = TestClient(main.app)
    assert c.get("/img", params={"u": "https://inside.shop.com/x.jpg"}).status_code == 400   # private network
    assert c.get("/img", params={"u": "http://cdn.shop.com/x.jpg"}).status_code == 400       # not https
    assert c.get("/img", params={"u": "https://cdn.shop.com/moved.jpg"}).status_code == 400  # redirect inside
    r = c.get("/img", params={"u": "https://cdn.shop.com/big.jpg"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert max(Image.open(io.BytesIO(r.content)).size) <= main.THUMB_PX
    n = len(calls)
    c.get("/img", params={"u": "https://cdn.shop.com/big.jpg"})
    assert len(calls) == n  # the second time it's served from memory


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

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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
    assert all(r == [{"url": "https://btech.com/p/1", "title": "Hero 50L", "snippet": "", "content": ""}] for r in res)
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

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
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


def test_page_we_cant_open_is_read_from_the_search_services_copy(monkeypatch):
    from app import fetchers, websearch
    copy = "DeFacto Basic Crew Neck T-Shirt 399 EGP [link: /p/basic-tee] " + "x " * 500

    async def fake_search(client, query, max_results=8, exclude_domains=None, include_domains=None):
        return [{"url": "https://www.defacto.com.eg/basic-tees", "title": "", "snippet": "", "content": copy}]

    async def blocked(client, sem, url):
        raise RuntimeError("403")

    async def reader_must_not_run(*a, **k):
        raise AssertionError("the search service's copy is enough")

    read = []

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
        read.append(prompt)
        return {"products": [{"title": "DeFacto Basic Crew Neck T-Shirt", "price": 399, "url": "/p/basic-tee"}]}

    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(agent, "_fetch_page", blocked)
    monkeypatch.setattr(fetchers, "read_pages", reader_must_not_run)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    products, errors = asyncio.run(agent._web_find(["basic t-shirt"]))
    assert [p.price for p in products] == [399] and "DeFacto Basic Crew Neck" in read[0] and not errors


def test_low_trust_shops_never_reach_the_judge(monkeypatch):
    from app import websearch
    from app.models import Product
    shady = Product("2segypt.com", "basic tee", 319, "https://2segypt.com/p/1", trust_score=25)
    ok = Product("defacto.com.eg", "basic tee", 399, "https://defacto.com.eg/p/1", trust_score=65)

    async def fake_find(queries, seed=None, famous=None):
        return [shady, ok], {}

    judged = []

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
        judged.append(prompt)
        return {"matches": [{"id": 0, "sure": True, "note": "ok"}], "summary": "s"}

    monkeypatch.setattr(websearch, "available", lambda: False)
    monkeypatch.setattr(agent, "_search_all", lambda q: fake_find(q))
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    out = asyncio.run(agent.run("تيشرت", plan={"action": "search", "request": "تيشرت", "queries": ["tee"]}))
    assert "2segypt" not in judged[0] and [r["store"] for r in out["results"]] == ["defacto.com.eg"]


def test_rate_limit_waits_as_long_as_the_provider_says():
    r = httpx.Response(429, text='{"error":{"message":"Rate limit reached ... Please try again in 1.53s. Need more tokens?"}}')
    assert llm.retry_hint(r) == 1.53
    assert llm.retry_hint(httpx.Response(429, text="try again in 1m2.5s")) == 62.5
    assert llm.retry_hint(httpx.Response(429, text="Please try again in 450ms.")) == 0.45
    assert llm.retry_hint(httpx.Response(429, text='"retryDelay": "17s"')) == 17
    assert llm.retry_hint(httpx.Response(429, headers={"retry-after": "3"}, text="")) == 3
    assert llm.retry_hint(httpx.Response(429, text="busy")) is None
    llm._COOLDOWN.clear()
    llm._rest("m-short", httpx.Response(429, text="Please try again in 1.5s."))
    assert llm._COOLDOWN["m-short"] - time.time() < 3  # not the fixed minute: the limit frees up in 1.5 s
    llm._COOLDOWN.clear()


def test_all_models_busy_waits_for_the_first_free_one_instead_of_giving_up(monkeypatch):
    """Live test 4.1: every Groq model was resting after per-minute 429s, the one model left was out of quota, and
    the planner gave up in 0.7 s with 24 s left -> a crude rules search with no results."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm._COOLDOWN.clear()
    llm._LAST_GOOD.clear()
    calls = []

    async def one_model(client, provider):
        return ["m1"]

    async def groq(client, model, prompt):
        calls.append(time.monotonic())
        if len(calls) == 1:
            return httpx.Response(429, text="Rate limit reached on tokens per minute (TPM). Please try again in 0.2s.")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    monkeypatch.setattr(llm, "_list_models", one_model)
    monkeypatch.setattr(llm, "_groq", groq)
    assert asyncio.run(llm.ask_json("x", prefer="groq", budget=10)) == {"ok": 1}
    assert len(calls) == 2 and calls[1] - calls[0] >= 0.2
    llm._COOLDOWN.clear()
    llm._LAST_GOOD.clear()


def test_question_about_results_is_not_searched_when_the_llm_is_down(monkeypatch):
    """Live test 2.2: the planner failed, and 'is the first one original?' was glued onto the old request and searched."""
    async def down(*a, **k):
        raise llm.LLMError("busy")

    async def no_search(*a, **k):
        raise AssertionError("a question must not become a search")

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", down)
    monkeypatch.setattr(agent, "run_rules", no_search)
    from app import main
    d = TestClient(main.app).post("/api/chat", json={
        "request": "سماعة انكر R50i NC",
        "messages": [{"role": "user", "text": "سماعة انكر R50i NC"}, {"role": "assistant", "text": "Shown: 1) ..."},
                     {"role": "user", "text": "اول نتيجة دي اصلية ولا ممكن تكون تقليد؟"}]}).json()
    assert d["action"] == "answer" and d["request"] == "سماعة انكر R50i NC" and d["results"] == []


def test_pages_are_read_for_the_whole_request_not_the_first_query(monkeypatch):
    """Live test 3.1: with one query per brand, pages found by the 2nd/3rd query were read for the 1st brand only."""
    from app import websearch
    page = "<html><body>" + "LC Waikiki Basic Crew Neck T-Shirt 299 EGP " * 60 + "</body></html>"

    async def fake_search(client, query, max_results=8, exclude_domains=None, include_domains=None):
        return [{"url": "https://www.lcwaikiki.eg/basic-tees", "title": "", "snippet": "", "content": ""}]

    async def page_without_data(client, sem, url):
        return [], page

    read = []

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
        read.append(prompt)
        return {"products": []}

    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(agent, "_fetch_page", page_without_data)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    asyncio.run(agent._web_find(["تيشرت بيزك TREE", "LC Waikiki basic t-shirt"],
                                want="تيشرتات بيزك سادة ماركة نضيفة (well-known brand)"))
    assert read and "تيشرتات بيزك سادة ماركة نضيفة (well-known brand)" in read[0]


def test_dedupe_one_page_in_two_languages_or_with_its_page_title():
    """Live test 6.1: Vodafone's iPhone 15 page showed 3 times (its <title>, the product name, the /en/ page)."""
    from app.models import Product
    u = "https://eshop.vodafone.com.eg/ar/prod/iphone-15"
    page_title = Product("V", "ايفون 15 | السعر والألوان والمواصفات في مصر 2025", 54999, u)
    name = Product("V", "iPhone 15 128GB Black", 54999, u)
    english = Product("V", "iPhone 15 5G in Egypt | 6GB | Latest Offers", 54999, u.replace("/ar/", "/en/"))
    other_price = Product("V", "iPhone 15 128GB Blue", 57999, u)
    assert agent._dedupe([page_title, name, english, other_price]) == [name, other_price]
    # two different items at one price on a listing page are both kept
    a = Product("S", "Basic tee black", 299, "https://shop.com/collections/basic-tees")
    b = Product("S", "Basic tee white", 299, "https://shop.com/collections/basic-tees")
    assert len(agent._dedupe([a, b])) == 2


def test_too_cheap_for_a_genuine_one_is_flagged_even_with_few_offers(monkeypatch):
    """Live test 8.1: Dior Sauvage 100 ml at 1,250 EGP was shown as a trusted pick: with 1-2 offers there's no
    median to compare with, so the judge's 'typical price of a genuine new one' is the reference."""
    from app.models import Product
    from app import websearch
    fake = Product("cezma.com", "عطر سوفاج من ديور للرجال 100 مل", 1250, "https://cezma.com/products/1", trust_score=65)
    real = Product("Amazon.eg", "Dior Sauvage EDP 100ml", 7900, "https://www.amazon.eg/dp/B1", trusted=True)

    async def fake_find(queries):
        return [fake, real], {}

    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
        assert "typical_price" in prompt
        return {"matches": [{"id": 0, "sure": True, "note": "من متجر موثوق"}, {"id": 1, "sure": True, "note": "أصلي"}],
                "typical_price": 8000, "best_deal_id": 0, "summary": "s"}

    monkeypatch.setattr(websearch, "available", lambda: False)
    monkeypatch.setattr(agent, "_search_all", fake_find)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    out = asyncio.run(agent.run("برفان ديور سوفاج 100 مل",
                                plan={"action": "search", "request": "برفان ديور سوفاج 100 مل", "queries": ["dior sauvage"]}))
    cheap, ok = out["results"]
    assert cheap["unconfirmed"] and not cheap["best_deal"] and "تقليد" in cheap["note"]
    assert not ok["unconfirmed"]


def test_slow_or_big_stores_are_not_skipped_and_blocks_expire():
    """noon.com got skipped for good after two searches where its ~1 MB pages were slower than our limit."""
    from app import db
    agent._remember_sites(["https://slowshop.com/p"], [], {"slowshop.com": "TimeoutError / reader too slow"})
    agent._remember_sites(["https://slowshop.com/p"], [], {"slowshop.com": "ReadTimeout"})
    agent._remember_sites(["https://www.noon.com/egypt-en/x"], [], {"noon.com": "403"})
    agent._remember_sites(["https://www.noon.com/egypt-en/y"], [], {"noon.com": "403"})
    assert db.blocked_sites() == set()  # slow != blocked; a well-known store is never skipped for it
    agent._remember_sites(["https://old.com/p"], [], {"old.com": "prices in SAR"})
    assert db.blocked_sites() == {"old.com"}
    db._db.run("UPDATE blocked_sites SET last_failed = ? WHERE domain = 'old.com'", (time.time() - 8 * 86400,))
    assert db.blocked_sites() == set()  # tried again after a week
    db.unblock("old.com")


def test_every_store_gets_a_turn_before_any_gets_a_second_page():
    """Live test: the well-known group's results were mostly Amazon pages, so noon/Jumia got no page budget."""
    urls = ["https://www.amazon.eg/a", "https://www.amazon.eg/b", "https://www.amazon.eg/c", "https://www.noon.com/x",
            "https://www.amazon.eg/d", "https://www.jumia.com.eg/y", "https://www.noon.com/z"]
    assert agent._take_turns(urls) == [
        "https://www.amazon.eg/a", "https://www.noon.com/x", "https://www.jumia.com.eg/y",
        "https://www.amazon.eg/b", "https://www.noon.com/z", "https://www.amazon.eg/c", "https://www.amazon.eg/d"]


def test_our_own_page_requests_respect_robots_txt(monkeypatch):
    from app import fetchers
    fetchers._ROBOTS.clear()
    fetched = []

    def handler(request: httpx.Request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /search")
        fetched.append(request.url.path)
        return httpx.Response(200, text="<html></html>")

    async def go(url):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await agent._fetch_page(c, asyncio.Semaphore(1), url)

    with pytest.raises(agent.RobotsDisallowed):
        asyncio.run(go("https://shop.test/search?q=x"))
    asyncio.run(go("https://shop.test/p/1"))
    assert fetched == ["/p/1"]
    fetchers._ROBOTS.clear()


def test_openrouter_daily_limit_rests_until_tomorrow():
    llm._COOLDOWN.clear()
    llm._rest("openrouter/free", httpx.Response(429, text='{"error":{"message":"Rate limit exceeded: free-models-per-day."}}'))
    assert llm._COOLDOWN["openrouter/free"] - time.time() > 600  # not retried every minute all day
    llm._COOLDOWN.clear()


def test_price_in_the_wrong_unit_is_dropped():
    """Live test 14.1 / 17.1: an office chair at 850,000 EGP and a vacuum at 829,900 (piasters read as pounds)."""
    from app.models import Product
    picks = [Product("arkan", "كرسي مكتب شبك", 7499, "u1"), Product("wood", "كرسي اداري شبك", 9900, "u2"),
             Product("wood", "كرسي شبك طبي اداري", 850000, "u3")]
    agent._flag_price_outliers(picks)
    assert [p.price for p in picks] == [7499, 9900]
    alone = [Product("x", "Panasonic MC-CL605", 829900, "u4")]
    agent._flag_price_outliers(alone, typical=5000)
    assert alone == []


def test_llm_use_is_counted_per_step(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm._COOLDOWN.clear(); llm._LAST_GOOD.clear(); llm.reset_usage()

    async def one_model(client, provider):
        return ["m1"]

    async def groq(client, model, prompt):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}], "usage": {"total_tokens": 1234}})

    monkeypatch.setattr(llm, "_list_models", one_model)
    monkeypatch.setattr(llm, "_groq", groq)
    asyncio.run(llm.ask_json("x", step="judge"))
    asyncio.run(llm.ask_json("x", step="judge"))
    assert llm.USAGE["calls"] == 2 and llm.USAGE["tokens"] == 2468 and llm.USAGE["by_step"]["judge"]["calls"] == 2
    llm._LAST_GOOD.clear(); llm.reset_usage()


def test_reply_without_text_tries_the_next_model(monkeypatch):
    """Live run 2: Cloudflare's gpt-oss used its default 256 tokens thinking and returned content=None, which
    crashed the whole chat turn (TypeError) instead of moving on to another model."""
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "k")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    llm._COOLDOWN.clear(); llm._LAST_GOOD.clear()
    bodies = []

    def handler(request: httpx.Request):
        import json as _j
        body = _j.loads(request.content)
        bodies.append(body)
        if body["model"] == "@cf/openai/gpt-oss-120b":
            return httpx.Response(200, json={"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 2}'}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(llm.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert asyncio.run(llm.ask_json("x")) == {"ok": 2}
    assert bodies[0]["max_tokens"] >= 1024
    llm._LAST_GOOD.clear()


def _groq_only(monkeypatch, models, extra_env=()):
    for k in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "NVIDIA_API_KEY", "CLOUDFLARE_API_KEY",
              "CEREBRAS_API_KEY", "GROQ_MODEL", "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    for k in extra_env:
        monkeypatch.setenv(k, "k")
    llm._MODEL_CACHE.clear(); llm._COOLDOWN.clear(); llm._LAST_GOOD.clear()

    async def fake_list(client, provider):
        return models.get(provider, [])
    monkeypatch.setattr(llm, "_list_models", fake_list)
    used = []

    async def answer(client, model, prompt):
        used.append(model)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}],
                                         "candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]})
    monkeypatch.setattr(llm, "_groq", answer)
    monkeypatch.setattr(llm, "_gemini", answer)
    monkeypatch.setattr(llm, "_caller", lambda p: answer)
    return used


def test_judge_uses_a_strong_model_not_the_last_page_reader(monkeypatch):
    """Live test 11 Nov, 4.1: the judge took 'the model that answered last', a page-reading model
    (llama-3.3-70b on Cloudflare), although gpt-oss-120b was free. Page reads no longer decide who judges."""
    used = _groq_only(monkeypatch, {"groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]})
    asyncio.run(llm.ask_json("page", prefer="bulk", spread=1, step="read_page"))
    assert used[-1] != "openai/gpt-oss-120b" and llm._LAST_GOOD == []  # reads keep off the strong model
    llm._LAST_GOOD[:] = ["openai/gpt-oss-20b"]  # even when a weak model answered last
    asyncio.run(llm.ask_json("judge", prefer="strong", step="judge"))
    assert used[-1] == "openai/gpt-oss-120b"
    assert llm.USAGE["by_step"]["judge"]["model"] == "openai/gpt-oss-120b"
    llm._LAST_GOOD.clear(); llm.reset_usage()


def test_judge_waits_a_few_seconds_for_a_strong_model(monkeypatch):
    """A strong model resting for its per-minute limit is worth a short wait: a weak judge lets no-name items in."""
    used = _groq_only(monkeypatch, {"groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]})
    llm._COOLDOWN["openai/gpt-oss-120b"] = time.time() + 0.3
    asyncio.run(llm.ask_json("judge", prefer="strong", budget=10, step="judge"))
    assert used == ["openai/gpt-oss-120b"]
    llm._COOLDOWN["openai/gpt-oss-120b"] = time.time() + 600  # resting long: the weaker one answers now
    asyncio.run(llm.ask_json("judge", prefer="strong", budget=10, step="judge"))
    assert used[-1] == "openai/gpt-oss-20b"
    llm._COOLDOWN.clear(); llm._LAST_GOOD.clear()


def test_page_reads_use_small_models_then_gemini_before_the_strong_one(monkeypatch):
    used = _groq_only(monkeypatch, {"groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
                                    "gemini": ["gemini-flash-lite-latest"]}, extra_env=("GEMINI_API_KEY",))
    asyncio.run(llm.ask_json("page", prefer="bulk", spread=1))
    assert used == ["openai/gpt-oss-20b"]
    llm._COOLDOWN["openai/gpt-oss-20b"] = time.time() + 600  # small one busy: Gemini, not the judge's model
    asyncio.run(llm.ask_json("page", prefer="bulk", spread=2))
    assert used[-1] == "gemini-flash-lite-latest"
    llm._COOLDOWN.clear(); llm._LAST_GOOD.clear()


def test_news_and_blog_addresses_are_not_shops():
    """Live test: news sites (tahiamasr, timesbull, al-ain, blog.btech) were opened as if they were stores."""
    from app import trust
    for u in ("https://www.tahiamasr.com/News/123", "https://al-ain.com/article/xiaomi-phones",
              "https://blog.btech.com/best-phones", "https://site.com/2025/05/best-phones", "https://x.com.eg/blog/tv"):
        assert not trust.is_shop_candidate(u), u
    for u in ("https://btech.com/ar/p/olympic-50l-945105411", "https://www.amazon.eg/dp/B0X", "https://shop.com/products/a"):
        assert trust.is_shop_candidate(u), u


def test_article_pages_are_not_read_as_stores(monkeypatch):
    news = ('<html><head><meta property="og:type" content="article"></head><body>'
            '<p>سعر سامسونج A16 في مصر 8999 جنيه</p></body></html>')
    shop = news.replace("</body>", '<button>Add to cart</button></body>')
    assert agent._is_article(news) and not agent._is_article(shop)
    assert agent._is_article('<script type="application/ld+json">{"@type": "NewsArticle"}</script>')

    replies = iter([{"page": "article", "currency": "EGP", "products": [{"title": "A16", "price": 8999}]},
                    {"page": "shop", "currency": "USD", "products": [{"title": "A16", "price": 199}]},
                    {"page": "shop", "currency": "EGP", "products": [
                        {"title": "A16", "price": 8999, "url": "/p/a16"},
                        {"title": "A16 (other country)", "price": 8000, "url": "https://www.amazon.in/dp/x"}]}])

    async def fake_ask(prompt, **kw):
        return next(replies)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    with pytest.raises(agent.NotAStore):
        asyncio.run(agent._llm_read_page("https://news.example.com/x", "<p>x</p>", "samsung"))
    with pytest.raises(agent.NotAStore):
        asyncio.run(agent._llm_read_page("https://shop.com/x", "<p>x</p>", "samsung"))
    got = asyncio.run(agent._llm_read_page("https://shop.com/x", "<p>x</p>", "samsung"))
    assert [p.url for p in got] == ["https://shop.com/p/a16"]  # links to another country's store are dropped


def test_page_text_sends_the_product_part_once():
    """Token use: the LLM gets the window with the products (search words + prices), each link once, no images."""
    menu = " ".join(f"<a href='/c/{i}'>Category {i}</a>" for i in range(400))
    cards = "".join(f"<div><a href='/dp/{i}'><img></a><a href='/dp/{i}'>Samsung Galaxy A{i} 5000mAh</a>"
                    f"<a href='/dp/{i}'>4.5 stars</a><span>EGP {8000 + i},00</span></div>" for i in range(30))
    html = f"<html><body><nav>{menu}</nav>{cards}<footer>{menu}</footer></body></html>"
    text = agent._page_text(html, ["samsung", "galaxy", "5000mah"], base_url="https://www.amazon.eg/s?k=a")
    assert len(text) <= agent.PAGE_CHARS
    assert text.count("Samsung Galaxy") == 30 and "Category 200 " not in text  # not the menus
    assert text.count("[link: /dp/5]") == 1
    md = "![img](https://cdn.x.com/a.jpg) [Galaxy A5 - EGP 8,999](https://www.amazon.eg/dp/5?ref=abc&tag=x)"
    out = agent._page_text(md, ["galaxy"], base_url="https://www.amazon.eg/s")
    assert "cdn.x.com" not in out and "](/dp/5)" in out


def test_judge_down_still_shows_the_candidates(monkeypatch):
    """Live test 11 Nov, 17.1: every model was busy at judging time and the simple rules wanted '2000w' in
    Arabic titles ("2000 وات"), so 28 found offers became 0 results. Now they're shown, all marked to check."""
    from app import websearch
    from app.models import Product
    items = [Product("Jumia", "مكنسة كهربائية توشيبا 2000 وات", 3500, "https://www.jumia.com.eg/p1", trusted=True),
             Product("B.TECH", "مكنسة باناسونيك 2000 وات", 4200, "https://btech.com/p2", trusted=True)]

    async def fake_find(queries):
        return items, {}

    async def busy(prompt, **kw):
        raise llm.LLMError("busy")
    monkeypatch.setattr(websearch, "available", lambda: False)
    monkeypatch.setattr(agent, "_search_all", fake_find)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", busy)
    out = asyncio.run(agent.run("مكنسة كهربائية 2000 وات",
                                plan={"action": "search", "request": "مكنسة كهربائية 2000 وات", "queries": ["vacuum"]}))
    assert out["results"] and out["results"][0]["price"] == 3500
    assert all(r["unconfirmed"] for r in out["results"]) and "مشغولة" in out["summary"]


def test_too_cheap_alternative_is_flagged_too():
    """Live test 11 Nov, 8.1: the judge found no exact match and offered a 1,250 EGP 'Dior Sauvage 100 ml' as an
    alternative (a genuine one is ~11,000 EGP); the safety net skipped alternatives, so it looked fine."""
    from app.models import Product
    alt = Product("cezma.com", "عطر سوفاج من ديور للرجال 100 مل", 1250, "https://cezma.com/p/1")
    alt.alternative = True
    agent._flag_price_outliers([alt], typical=11300)
    assert alt.unconfirmed and "تقليد" in alt.note
    close = Product("Amazon.eg", "Dior Sauvage EDT 60ml", 7000, "https://amazon.eg/dp/2")
    close.alternative = True
    agent._flag_price_outliers([close], typical=11300)
    assert not close.unconfirmed


def test_cloudflare_daily_allocation_rests_until_tomorrow():
    """Cloudflare says 'you have used up your daily free allocation of 10,000 neurons': that is a daily limit,
    not a one-minute one, so its models aren't retried on every call for the rest of the day."""
    llm._COOLDOWN.clear()
    r = httpx.Response(429, json={"errors": [{"message": "AiError: you have used up your daily free allocation of "
                                                         "10,000 neurons, please upgrade"}]})
    llm._rest("@cf/openai/gpt-oss-120b", r)
    assert llm._COOLDOWN["@cf/openai/gpt-oss-120b"] > time.time() + 600
    llm._COOLDOWN.clear()


def test_no_judge_fallback_keeps_the_searched_brand_and_grade():
    """Live test 11 Nov, 12.1 / 16.1 with every model out: the word rules showed ATA/Haier/Hisense TVs for
    'Samsung 55 inch' and 10W-40 oil for '5W-30'. Rare words of the searches now decide (no product rules)."""
    from app.models import Product
    tvs = ["ATA 55 Inch Smart TV , 4K UHD LED", "Haier 55 Inch Frameless Smart TV , 4K UHD", "Hisense 55 Inch Smart TV , 4K",
           "Samsung 55 Inch Smart TV , 4K LED", "LG 55 Inch Smart TV , 4K UHD LED", "شاشة سمارت سامسونج 55 بوصة 4K"]
    pool = [Product("B.TECH", t, 1000 + i, f"https://btech.com/p{i}") for i, t in enumerate(tvs)]
    got = agent._fallback_pick(pool, ["شاشة سامسونج 55 بوصة سمارت 4K", "Samsung 55 inch 4K smart TV price Egypt"])
    assert [p.title for p in got] == [tvs[3], tvs[5]]
    oils = ["Mobil 1 FS 5W-30 Motor Oil - 4 Liters", "Mobil Super Synthetic Technology X1 10W-40 4 Litres",
            "PROFI-CAR Eco-Drive Longlife 1 SAE 5W-30 Synthetic Engine Oil 4L", "MOBIL 1 ESP Formula 5W-30 Motor Oil 4L",
            "Mobil 1 10,000 km FS 5W-40Motor Oil, 4L", "1 10, 000 km 4 Liters 5W40 FS Motor Oil"]
    pool = [Product("A", t, 1000 + i, f"https://amazon.eg/dp/{i}") for i, t in enumerate(oils)]
    got = [p.title for p in agent._fallback_pick(pool, ["زيت عربية موبيل 1 5w30 4 لتر", "Mobil 1 5W-30 4L motor oil price Egypt"])]
    assert oils[0] in got and oils[3] in got and not any("40" in t for t in got)


def test_currency_block_needs_a_real_currency_and_spares_known_stores():
    """Live test 11 Nov: cairobook.com (an Egyptian bookshop) was blocked for 'prices in ABC' (a template
    placeholder), and samsung.com for a page of its India section (INR), which also hid samsung.com/eg."""
    from app import db
    from app.extract import foreign_currency
    assert foreign_currency('<script>var t = {currency: "ABC"}</script>') is None
    assert foreign_currency('<script>{"priceCurrency": "SAR"}</script>') == "SAR"
    agent._remember_sites(["https://www.samsung.com/in/x", "https://saudi-shop.com/p"], [],
                          {"samsung.com": "prices in INR", "saudi-shop.com": "prices in SAR"})
    assert "samsung.com" not in db.blocked_sites() and "saudi-shop.com" in db.blocked_sites()


def test_daily_limit_per_visitor_and_wrong_result_reports(monkeypatch):
    from app import db, main
    runs = []

    async def fake_run(q, plan=None, budget_text=None):
        runs.append(q)
        return {"mode": "rules", "results": [{"title": "t", "price": 10.0, "store": "s", "url": f"https://s.com/{q}"}]}

    monkeypatch.setattr(llm, "available", lambda: False)
    monkeypatch.setattr(agent, "run", fake_run)
    monkeypatch.setattr(main, "DAILY_LIMIT", 1)
    c = TestClient(main.app)
    ask = lambda text, ip: c.post("/api/chat", json={"messages": [{"role": "user", "text": text}]},
                                  headers={"X-Forwarded-For": ip}).json()
    assert ask("سخان 50 لتر", "1.1.1.1")["action"] == "search"
    blocked = ask("غسالة 8 كيلو", "1.1.1.1")
    assert blocked["action"] == "answer" and blocked["limit"] and runs == ["سخان 50 لتر"]
    assert ask("غسالة 8 كيلو", "2.2.2.2")["action"] == "search"  # another visitor has their own limit
    # a faked left entry or the host's internal hops don't give a visitor a fresh limit
    assert ask("غسالة 8 كيلو", "9.9.9.9, 1.1.1.1, 10.0.0.5")["action"] == "answer"
    r = c.post("/api/report", json={"reason": "wrong_price", "request": "سخان", "url": "https://s.com/x",
                                    "store": "s", "title": "t", "price": 10}, headers={"X-Forwarded-For": "1.1.1.1"})
    assert r.json() == {"ok": True}
    [rep] = db.list_reports()
    assert rep["reason"] == "wrong_price" and rep["request"] == "سخان"
    assert c.post("/api/report", json={"reason": "nonsense", "url": "u"}).status_code == 422
    assert "1.1.1.1" not in str(db._db.run("SELECT * FROM usage"))  # only a hash is stored


def test_price_db_keeps_working_when_postgres_is_unreachable(monkeypatch, caplog):
    from app import db
    monkeypatch.setenv("DATABASE_URL", "host=127.0.0.1 port=1 user=x dbname=x connect_timeout=1")
    db.close()
    db._db._pg_down_until = 0.0
    db.count_search("abc")  # no error: falls back to the local file
    assert db.searches_today("abc") == 1 and not db._db.pg
    assert "can't reach DATABASE_URL" in caplog.text
    db._db._pg_down_until = 0.0



def test_feedback_box_is_saved_and_config_lists_markets():
    from fastapi.testclient import TestClient
    from app import db
    from app.main import app
    c = TestClient(app)
    assert c.post("/api/feedback", json={"kind": "suggestion", "message": "ضيفوا مقارنة بين منتجين"}).json() == {"ok": True}
    assert c.post("/api/feedback", json={"kind": "spam", "message": "hello"}).status_code == 422
    assert c.post("/api/feedback", json={"kind": "complaint", "message": ""}).status_code == 422
    rows = db.list_feedback()
    assert len(rows) == 1 and rows[0]["kind"] == "suggestion" and rows[0]["contact"] == ""
    cfg = c.get("/api/config").json()
    assert cfg["market"] == "EG" and {m["code"] for m in cfg["markets"]} == {"EG", "SA", "AE"}
    assert [m["code"] for m in cfg["markets"] if m["enabled"]] == ["EG"]
    assert c.get("/").status_code == 200 and c.get("/logo.svg").status_code == 200


def test_one_shot_search_has_the_daily_limit_too_and_reports_are_capped(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main

    async def fake_run(q, plan=None, budget_text=None):
        return {"mode": "rules", "results": []}

    monkeypatch.setattr(agent, "run", fake_run)
    monkeypatch.setattr(main, "DAILY_LIMIT", 1)
    monkeypatch.setitem(main.SPAM_LIMITS, "report", 2)
    main._sent.clear()
    c = TestClient(main.app)
    ip = {"X-Forwarded-For": "5.5.5.5"}
    assert c.get("/api/search", params={"q": "غسالة"}, headers=ip).status_code == 200
    assert c.get("/api/search", params={"q": "تلاجة"}, headers=ip).status_code == 429
    rep = {"reason": "other", "url": "https://s.com/x"}
    assert [c.post("/api/report", json=rep, headers=ip).status_code for _ in range(3)] == [200, 200, 429]


def _jsonld_page(name, price, availability):
    return f"""<html><body><h1>{name}</h1><script type="application/ld+json">{{"@type": "Product", "name": "{name}",
    "offers": {{"@type": "Offer", "price": "{price}", "priceCurrency": "EGP",
    "availability": "https://schema.org/{availability}"}}}}</script></body></html>"""


def test_results_are_checked_on_their_own_page_before_showing(monkeypatch):
    """A listing can say "in stock" at an old price: each result's own page is opened just before showing it.
    Sold out there = dropped; a new price there = the shown price; a page we can't open = shown, marked unchecked."""
    from app import fetchers
    from app.models import Product
    pages = {"https://shop-a.com/p/olympic-hero-50": _jsonld_page("Olympic Hero 50L", 3849, "InStock"),
             "https://shop-b.com/p/olympic-hero-50": _jsonld_page("Olympic Hero 50L", 3700, "OutOfStock")}

    async def fake_fetch(client, sem, url):
        if url not in pages:
            raise httpx.ConnectError("blocked")
        return agent._cheapest_variant(extract_products(pages[url], "x", url)), pages[url]

    async def fake_read(client, urls, deadline=12):
        return {u: "Olympic Hero 50L 3,999 EGP غير متوفر حاليا" for u in urls if "shop-c" in u}, {}

    monkeypatch.setattr(agent, "_fetch_page", fake_fetch)
    monkeypatch.setattr(fetchers, "read_pages", fake_read)
    picks = [Product("A", "Olympic Hero 50L", 3999, "https://shop-a.com/p/olympic-hero-50"),
             Product("B", "Olympic Hero 50L", 3700, "https://shop-b.com/p/olympic-hero-50"),
             Product("C", "Olympic Hero 50L", 3999, "https://shop-c.com/p/olympic-hero-50"),
             Product("D", "Olympic Hero 50L", 4100, "https://shop-d.com/p/olympic-hero-50")]
    kept, gone = asyncio.run(agent._verify_stock(picks))
    assert gone == 2 and [p.store for p in kept] == ["A", "D"]
    assert kept[0].price == 3849 and kept[0].verified is True and kept[0].in_stock is True
    assert kept[1].verified is False  # couldn't open it: still shown, the page asks the shopper to check


def test_store_the_shopper_names_is_searched_on_its_own(monkeypatch):
    from app import websearch
    searched = []

    async def fake_search(client, query, max_results=8, exclude_domains=None, include_domains=None):
        searched.append(tuple(include_domains or ()))
        if include_domains == ["jumia.com.eg"]:
            return [{"url": "https://www.jumia.com.eg/olympic-hero-50l-oye05061wn.html", "title": "", "snippet": ""}]
        return [{"url": f"https://shop{len(searched)}.com/p/{i}", "title": "", "snippet": ""} for i in range(30)]

    opened = []

    async def fake_fetch(client, sem, url):
        opened.append(url)
        raise httpx.ConnectError("x")

    async def no_read(client, urls, deadline=12):
        return {}, {}

    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(agent, "_fetch_page", fake_fetch)
    monkeypatch.setattr(agent.fetchers, "read_pages", no_read)
    monkeypatch.setattr(agent, "_blocked_sites", lambda: asyncio.sleep(0, set()))
    monkeypatch.setattr(agent, "_remember_sites", lambda *a: None)
    assert agent._domains(["jumia.com.eg", "https://www.Noon.com/egypt-ar", "not a domain"]) == ["jumia.com.eg", "noon.com"]
    asyncio.run(agent._web_find(["سخان اوليمبيك هيرو 50"], famous=["amazon.eg"], named=["jumia.com.eg"]))
    assert ("jumia.com.eg",) in searched
    assert opened[0] == "https://www.jumia.com.eg/olympic-hero-50l-oye05061wn.html"


def test_search_box_ideas_follow_what_is_searched_and_trending(monkeypatch):
    """The examples under the search box come from the site's recent searches + the web's trends (one LLM call,
    kept for 12 hours); with no LLM or a bad answer the page keeps its built-in ones."""
    from app import db, main, trending, websearch
    monkeypatch.setattr(main, "DEMO", False)
    monkeypatch.setattr(trending, "_memory", {})
    db.save_search(db.cache_key("سخان كهربا 50 لتر", "EG"), "EG", {"results": []})
    db.save_search(db.cache_key("ايفون 15", "EG"), "EG", {"results": []})
    prompts = []

    async def fake_ask(prompt, **kw):
        prompts.append(prompt)
        return {"ar": ["عروض الجمعة البيضاء على الموبايلات", "سخان ٥٠ لتر رخيص", "x" * 80, "دفاية كهربا للشتا"],
                "en": ["Black Friday phone deals", "Cheap 50L water heater", "x", "Electric heater for winter"]}

    async def fake_search(client, query, max_results=8, **kw):
        return [{"url": "https://news.example/x", "title": "الأكثر مبيعا: دفايات", "snippet": "الشتا"}]

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    monkeypatch.setattr(websearch, "available", lambda: True)
    monkeypatch.setattr(websearch, "search", fake_search)
    assert db.recent_requests("EG") == ["ايفون 15", "سخان كهربا 50 لتر"]
    with TestClient(main.app) as c:
        d = c.get("/api/suggestions").json()
        assert d["ar"] == ["عروض الجمعة البيضاء على الموبايلات", "سخان ٥٠ لتر رخيص", "دفاية كهربا للشتا"]
        assert d["en"] == ["Black Friday phone deals", "Cheap 50L water heater", "Electric heater for winter"]
        assert "ايفون 15" in prompts[0] and "الأكثر مبيعا: دفايات" in prompts[0]
        assert c.get("/api/suggestions").json() == d and len(prompts) == 1  # kept, not made again
    assert db.cache_key(db.TRENDING_KEY, "EG").removeprefix("EG:") not in db.recent_requests("EG")
    monkeypatch.setattr(trending, "_memory", {})  # after a restart: read back from the database
    monkeypatch.setattr(llm, "available", lambda: False)
    assert asyncio.run(trending.suggestions())["ar"][0] == "عروض الجمعة البيضاء على الموبايلات"


def test_search_box_ideas_drop_english_when_it_does_not_line_up():
    from app import trending
    assert trending._pairs(["هدية لأمي", "سماعة"], ["Gift for mom"]) == (["هدية لأمي", "سماعة"], [])
    assert trending._pairs(["هدية لأمي", "سماعة"], ["Gift for mom", ""]) == (["هدية لأمي", "سماعة"], [])
    assert trending._pairs("x", None) == ([], [])


def test_search_engines_can_find_and_verify_the_site(monkeypatch):
    from app import main
    with TestClient(main.app) as c:
        r = c.get("/robots.txt")
        assert "Disallow: /api/" in r.text and "Sitemap: http://testserver/sitemap.xml" in r.text
        assert "<loc>http://testserver/</loc>" in c.get("/sitemap.xml").text
        assert c.get("/google1a2b3c.html").status_code == 404  # not set up yet
        monkeypatch.setenv("GOOGLE_SITE_VERIFICATION", "google1a2b3c.html")
        assert c.get("/google1a2b3c.html").text == "google-site-verification: google1a2b3c.html"
        assert c.get("/googlezzz.html").status_code == 404
        assert "Easy Found" in c.get("/").text


def test_more_stores_reach_the_judge_and_deals_come_first():
    from app.models import Product

    def P(store, price, old=None):
        p = Product(store, f"{store} {price}", price, f"https://{store}.com/p/{price}", old_price=old)
        p.discount_pct = agent._discount(p)
        return p
    big = [P("big", 100 + i) for i in range(60)]  # one store's long listing page, all cheaper
    others = [P("small", 300), P("mid", 400), P("mid", 900, old=1200)]
    pool = agent._candidates(big + others)
    assert len(pool) == agent.MAX_CANDIDATES and {"small", "mid"} <= {p.store for p in pool}
    assert any(p.price == 900 for p in pool)  # discounted: always gets a place
    assert agent._candidates([P("only", 100 + i) for i in range(10)])[-1].price == 109  # one store: all of it
    shown = agent._show_order([P("a", 100), P("a", 101), P("a", 102), P("a", 103), P("b", 106), P("c", 108, old=150)])
    assert shown[0].price == 100  # the cheapest stays first
    assert shown[1].store == "c"  # a discount comes before slightly cheaper offers
    assert [p.store for p in shown[2:5]] != ["a", "a", "a"]  # another store breaks a run of one store's offers
