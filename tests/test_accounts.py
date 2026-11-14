"""Google sign-in, region and preferences, followed products, price-drop notifications."""
import os
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import accounts, agent, db, main, market
from app.models import Product


@pytest.fixture(autouse=True)
def _accounts_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    for name in ("DATABASE_URL", "BREVO_API_KEY", "MAIL_FROM", "LIVE_MARKETS", "MARKET"):
        monkeypatch.delenv(name, raising=False)
    if pg := os.getenv("TEST_DATABASE_URL"):  # also run these against a real Postgres (an empty test database!)
        monkeypatch.setenv("DATABASE_URL", pg)
        for table in ("notifications", "watches", "chats", "users"):
            accounts._db.run(f"DROP TABLE IF EXISTS {table}")
        accounts.close()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    monkeypatch.setenv("CRON_SECRET", "cron-secret")
    yield
    accounts.close()
    db.close()


def _google(monkeypatch, sub="123", email="mona@gmail.com"):
    async def fake(credential, client=None):
        assert credential == "x" * 30
        return {"id": "g:" + sub, "email": email, "name": "Mona Ali", "picture": ""}
    monkeypatch.setattr(accounts, "verify_google", fake)


def _signed_in(monkeypatch, **kw) -> TestClient:
    _google(monkeypatch, **kw)
    c = TestClient(main.app)
    d = c.post("/api/auth/google", json={"credential": "x" * 30}).json()
    assert d["user"]["market"] is None  # region not chosen yet: the page asks for it
    return c


def test_session_cookie_is_signed_and_expires():
    tok = accounts.make_session("g:1")
    assert accounts.read_session(tok) == "g:1"
    assert accounts.read_session(tok.replace("g:1", "g:2")) is None  # edited
    assert accounts.read_session(accounts.make_session("g:1", now=time.time() - 31 * 86400)) is None
    assert accounts.read_session(None) is None and accounts.read_session("junk") is None


def test_verify_google_checks_the_token_is_for_us(monkeypatch):
    good = {"aud": "test-client.apps.googleusercontent.com", "iss": "https://accounts.google.com",
            "exp": str(int(time.time()) + 600), "email_verified": "true", "email": "a@gmail.com", "sub": "42",
            "name": "A"}

    async def check(token_info):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=token_info))
        async with httpx.AsyncClient(transport=transport) as client:
            return await accounts.verify_google("tok", client)

    import asyncio
    assert asyncio.run(check(good))["id"] == "g:42"
    for bad in ({"aud": "other-site"}, {"email_verified": "false"}, {"exp": "1"}, {"iss": "evil.com"}):
        with pytest.raises(accounts.AuthError):
            asyncio.run(check({**good, **bad}))


def test_signup_region_and_preferences(monkeypatch):
    c = _signed_in(monkeypatch)
    assert c.get("/api/me").json()["user"]["email"] == "mona@gmail.com"
    d = c.patch("/api/me", json={"market": "SA", "email_alerts": False, "prefs": {"hide_unsure_shops": True,
                                                                                    "unknown": True}}).json()
    assert d["user"]["market"] == "SA" and d["user"]["email_alerts"] is False
    assert d["user"]["prefs"] == {"hide_unsure_shops": True}
    # Saudi search isn't live yet: told so in a notification
    notes = c.get("/api/notifications").json()
    assert notes["unseen"] == 1 and "السعودية" in notes["notifications"][0]["title"]
    c.post("/api/notifications/seen")
    assert c.get("/api/me").json()["unseen"] == 0
    assert c.patch("/api/me", json={"market": "FR"}).status_code == 422
    # signing in again keeps the saved choices
    _google(monkeypatch)
    again = TestClient(main.app).post("/api/auth/google", json={"credential": "x" * 30}).json()
    assert again["user"]["market"] == "SA"


def test_signed_out_and_sign_in_off(monkeypatch):
    c = TestClient(main.app)
    assert c.get("/api/me").json() == {"user": None, "unseen": 0}
    assert c.get("/api/watches").status_code == 401
    assert c.cookies.get(accounts.COOKIE) is None
    monkeypatch.delenv("GOOGLE_CLIENT_ID")
    assert c.get("/api/config").json()["google_client_id"] == ""
    assert c.post("/api/auth/google", json={"credential": "x" * 30}).status_code == 404


def test_follow_price_and_drop_notification(monkeypatch):
    c = _signed_in(monkeypatch)
    c.patch("/api/me", json={"market": "EG"})
    c.post("/api/notifications/seen")
    url = "https://btech.com/ar/kettle-123"
    w = c.post("/api/watches", json={"url": url, "title": "Kettle", "store": "B.TECH", "price": 1000}).json()["watch"]
    assert w["first_price"] == 1000 and w["market"] == "EG"
    # a search sees it more expensive: no alert; then cheaper: alert
    assert accounts.prices_seen([{"url": url, "price": 1100}]) == []
    made = accounts.prices_seen([{"url": url, "price": 900}, {"url": "https://other.com/x", "price": 5}])
    assert len(made) == 1 and "1,100" in made[0]["body"] and "900" in made[0]["body"] and "10%" in made[0]["body"]
    assert accounts.prices_seen([{"url": url, "price": 900}]) == []  # same price again: told once
    assert accounts.prices_seen([{"url": url, "price": 500, "in_stock": False}]) == []  # sold out isn't a deal
    notes = c.get("/api/notifications").json()
    assert notes["unseen"] == 1 and notes["notifications"][0]["url"] == url
    # a target price: only at or below it
    c.post("/api/watches", json={"url": url, "title": "Kettle", "price": 900, "target_price": 700})
    assert accounts.prices_seen([{"url": url, "price": 800}]) == []
    assert len(accounts.prices_seen([{"url": url, "price": 700}])) == 1
    assert c.delete(f"/api/watches/{w['id']}").json() == {"ok": True}
    assert c.get("/api/watches").json()["watches"] == []


def test_watches_are_per_user(monkeypatch):
    a = _signed_in(monkeypatch, sub="1", email="a@gmail.com")
    wid = a.post("/api/watches", json={"url": "https://x.com/p", "title": "P", "price": 10}).json()["watch"]["id"]
    b = _signed_in(monkeypatch, sub="2", email="b@gmail.com")
    assert b.get("/api/watches").json()["watches"] == []
    assert b.delete(f"/api/watches/{wid}").status_code == 404
    assert len(a.get("/api/watches").json()["watches"]) == 1


def test_delete_account_removes_everything(monkeypatch):
    c = _signed_in(monkeypatch)
    c.post("/api/watches", json={"url": "https://x.com/p", "title": "P", "price": 10})
    c.delete("/api/me")
    assert accounts.get_user("g:123") is None and accounts.list_watches("g:123") == []
    assert c.get("/api/me").json()["user"] is None


def test_chat_uses_user_preferences_and_limit(monkeypatch):
    c = _signed_in(monkeypatch)
    c.patch("/api/me", json={"market": "AE", "prefs": {"hide_unsure_shops": True}})
    url = "https://btech.com/p1"
    c.post("/api/watches", json={"url": url, "title": "P1", "price": 100})
    seen_market = []

    async def fake_turn(turn, client, limit):
        seen_market.append((market.current().code, client, limit))
        return {"action": "search", "results": [
            {"url": url, "price": 90, "trusted": True},
            {"url": "https://shady.shop/p", "price": 50, "trust_score": 40}], "count": 2}
    monkeypatch.setattr(main, "_chat_turn_in_market", fake_turn)
    monkeypatch.setattr(main, "USER_DAILY_LIMIT", 15)
    d = c.post("/api/chat", json={"messages": [{"role": "user", "text": "غلاية"}]}).json()
    assert [r["url"] for r in d["results"]] == [url] and d["results"][0]["watched"] is True
    assert "الإمارات" in d["region_note"]  # UAE picked, but searching there isn't live yet
    mk, client, limit = seen_market[0]
    assert mk == "EG" and client.startswith("u") and limit == 15
    monkeypatch.setenv("LIVE_MARKETS", "EG,AE")  # once it's live, the shopper's region is searched
    d = c.post("/api/chat", json={"messages": [{"role": "user", "text": "غلاية"}]}).json()
    assert seen_market[1][0] == "AE" and "region_note" not in d
    assert market.current().code == "EG"  # only for that request


def test_scheduled_check_finds_drop(monkeypatch):
    c = _signed_in(monkeypatch)
    url = "https://btech.com/ar/kettle-123"
    c.post("/api/watches", json={"url": url, "title": "Kettle", "store": "B.TECH", "price": 1000})
    assert TestClient(main.app).post("/api/cron/check-watches").status_code == 403
    monkeypatch.setattr(main, "CHECK_HOURS", 0)

    async def fake_fetch(client, sem, u):
        return [Product(store="B.TECH", title="Kettle", price=850, url=u + "/")], "<html>"
    monkeypatch.setattr(agent, "_fetch_page", fake_fetch)
    d = c.post("/api/cron/check-watches", headers={"X-Cron-Secret": "cron-secret"}).json()
    assert d == {"checked": 1, "prices": 1, "alerts": 1, "failed": {}}
    assert c.get("/api/watches").json()["watches"][0]["price"] == 850
    assert db.history(url)[-1]["price"] == 850


def test_accounts_refuse_the_temporary_fallback_database(monkeypatch):
    """DATABASE_URL set but Postgres down: db.py uses a local file for a while. Accounts must not be written
    there (they'd be lost), so account requests fail and searching goes on as a visitor."""
    c = _signed_in(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    monkeypatch.setattr(db._db, "_pg_down_until", time.time() + 60)
    r = c.patch("/api/me", json={"market": "EG"})
    assert r.status_code == 503 and "الحسابات" in r.json()["detail"]
    assert c.get("/api/me").json()["user"] is None


def test_privacy_page():
    r = TestClient(main.app).get("/privacy")
    assert r.status_code == 200 and "سياسة الخصوصية" in r.text and "delete my account" in r.text


def test_logo_files():
    c = TestClient(main.app)
    assert c.get("/logo.svg").headers["content-type"].startswith("image/svg")
    for path in ("/logo-120.png", "/logo-512.png", "/favicon.ico"):
        r = c.get(path)
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"


def test_scheduled_check_reads_pages_it_cant_open_through_the_search_service(monkeypatch):
    """Amazon's robots.txt keeps our own requests off its product pages: the price comes from the search
    service's copy of that page instead (the same product, whatever the link's language part or tracking)."""
    from app import llm, websearch
    c = _signed_in(monkeypatch)
    url = "https://www.amazon.eg/-/en/Puma-Electron-Running-Unisex-Silver/dp/B0CJZY1CTP/ref=sr_1_46"
    c.post("/api/watches", json={"url": url, "title": "Puma Unisex Running Sneaker", "store": "Amazon.eg", "price": 2739})
    monkeypatch.setattr(main, "CHECK_HOURS", 0)

    async def blocked(client, sem, u):
        raise agent.RobotsDisallowed()

    async def fake_search(client, query, max_results=8, include_domains=None, only=None, **kw):
        assert query == "Puma Unisex Running Sneaker b0cjzy1ctp" and include_domains == ["amazon.eg"]
        return [{"url": "https://www.amazon.eg/dp/B0OTHER123", "content": "x" * 300},
                {"url": "https://www.amazon.eg/Puma-Unisex-Running-Sneaker/dp/B0CJZY1CTP", "content": "Puma ... EGP2,385.00 " * 20}]

    async def fake_ask(prompt, **kw):
        assert "EGP2,385.00" in prompt
        return {"page": "shop", "currency": "EGP", "products": [
            {"title": "Puma Unisex Running Sneaker", "price": 2385, "old_price": 2739, "url": None, "in_stock": True}]}
    monkeypatch.setattr(agent, "_fetch_page", blocked)
    monkeypatch.setattr(websearch, "configured", lambda: ["tavily"])
    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    d = c.post("/api/cron/check-watches", headers={"X-Cron-Secret": "cron-secret"}).json()
    assert d == {"checked": 1, "prices": 1, "alerts": 1, "failed": {}}
    assert c.get("/api/watches").json()["watches"][0]["price"] == 2385
    body = c.get("/api/notifications").json()["notifications"][0]["body"]
    assert "2,385" in body and "اتأكد منه في صفحة المنتج" in body  # read from a search copy: says so


def test_product_links_match_by_their_product_number():
    assert agent._same_product_link("https://www.amazon.eg/-/en/Puma-SMASH/dp/B09YXNHF8N/ref=sr_1_33",
                                    "https://www.amazon.eg/dp/B09YXNHF8N")
    assert not agent._same_product_link("https://www.amazon.eg/dp/B09YXNHF8N", "https://www.amazon.eg/dp/B0CJZY1CTP")
    assert not agent._same_product_link("https://www.amazon.eg/dp/B09YXNHF8N", "https://noon.com/dp/B09YXNHF8N")


def test_saved_chats_follow_the_shopper(monkeypatch):
    c = _signed_in(monkeypatch)
    data = {"history": [{"role": "user", "text": "سخان 50 لتر"}], "currentRequest": "سخان 50 لتر",
            "turns": [{"u": "سخان 50 لتر", "d": {"action": "search", "results": []}}]}
    assert c.put("/api/chats/abcdef123456", json={"title": "سخان 50 لتر", "data": data}).json() == {"ok": True}
    assert c.put("/api/chats/zzzz9999yyyy", json={"title": "غسالة", "data": {"turns": []}}).status_code == 200
    chats = c.get("/api/chats").json()["chats"]
    assert [x["title"] for x in chats] == ["غسالة", "سخان 50 لتر"]  # newest first, without the conversations
    assert "data" not in chats[0]
    got = c.get("/api/chats/abcdef123456").json()
    assert got["data"] == data and got["title"] == "سخان 50 لتر"
    # saving again updates the same chat
    c.put("/api/chats/abcdef123456", json={"title": "سخان 50 لتر", "data": {**data, "currentRequest": "سخان 80 لتر"}})
    assert c.get("/api/chats/abcdef123456").json()["data"]["currentRequest"] == "سخان 80 لتر"
    assert len(c.get("/api/chats").json()["chats"]) == 2
    # someone else can't read it, bad ids and huge chats are refused
    other = _signed_in(monkeypatch, sub="999", email="x@gmail.com")
    assert other.get("/api/chats/abcdef123456").status_code == 404 and other.get("/api/chats").json()["chats"] == []
    assert c.get("/api/chats/../../etc").status_code == 404 and c.put("/api/chats/short", json={"title": "x", "data": {}}).status_code == 422
    assert c.put("/api/chats/abcdef123456", json={"title": "x", "data": {"s": "x" * 500_000}}).status_code == 413
    assert TestClient(main.app).get("/api/chats").status_code == 401
    assert c.delete("/api/chats/abcdef123456").json() == {"ok": True}
    assert c.get("/api/chats/abcdef123456").status_code == 404 and c.delete("/api/chats/abcdef123456").status_code == 404
    c.delete("/api/me")
    assert accounts.list_chats("g:123") == []


def test_only_the_newest_chats_are_kept(monkeypatch):
    monkeypatch.setattr(accounts, "MAX_CHATS", 3)
    accounts.upsert_user({"id": "g:1", "email": "a@b.c", "name": "A", "picture": ""})
    for i in range(5):
        accounts.save_chat("g:1", f"chat{i:08d}", f"t{i}", {})
        time.sleep(0.01)
    assert [x["title"] for x in accounts.list_chats("g:1")] == ["t4", "t3", "t2"]


def test_followed_product_found_on_a_crawled_listing_page(monkeypatch):
    from app import llm, websearch
    url = "https://www.amazon.eg/-/en/Puma-Electron/dp/B0CJZY1CTP/ref=sr_1_46"

    async def fake_search(client, query, max_results=8, include_domains=None, only=None, **kw):
        return [{"url": "https://www.amazon.eg/dp/B0CJZY1CTP", "content": "", "snippet": "Puma sneaker"},
                {"url": "https://www.amazon.eg/s?k=puma+sneaker", "content": "listing " * 200}]

    async def fake_ask(prompt, **kw):
        return {"page": "shop", "currency": "EGP", "products": [
            {"title": "Other", "price": 999, "url": "/dp/B0OTHER999"},
            {"title": "Puma Unisex Running Sneaker", "price": 2385, "url": "/-/en/Puma/dp/B0CJZY1CTP/ref=sr_1_3"}]}
    monkeypatch.setattr(websearch, "configured", lambda: ["tavily"])
    monkeypatch.setattr(websearch, "search", fake_search)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    import asyncio
    import httpx

    async def go():
        async with httpx.AsyncClient() as client:
            return await agent.price_from_search(client, url, "Puma Unisex Running Sneaker")
    p, why = asyncio.run(go())
    assert why == "" and p.price == 2385 and p.url == url
