"""The owner's /admin page: password, usage counting (visits, chat messages, store clicks), the stats."""
import asyncio
import os
import time

import pytest
import httpx
from fastapi.testclient import TestClient

from app import accounts, admin, affiliate, db, llm, main


@pytest.fixture(autouse=True)
def _db_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    for name in ("DATABASE_URL", "ADMIN_PASSWORD", "AFFILIATE_TAGS", "GOOGLE_CLIENT_ID"):
        monkeypatch.delenv(name, raising=False)
    if pg := os.getenv("TEST_DATABASE_URL"):  # also run these against a real Postgres (an empty test database!)
        monkeypatch.setenv("DATABASE_URL", pg)
        for table in ("events", "feedback", "reports", "blocked_sites", "notifications", "watches", "chats", "users"):
            db._db.run(f"DROP TABLE IF EXISTS {table}")
        db.close()
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    admin._tries.clear()
    yield
    db.close()


def _admin(monkeypatch) -> TestClient:
    monkeypatch.setenv("ADMIN_PASSWORD", "right-one")
    c = TestClient(main.app)
    assert c.post("/api/admin/login", json={"password": "right-one"}).status_code == 200
    return c


def test_admin_is_off_without_a_password_and_needs_the_right_one(monkeypatch):
    c = TestClient(main.app)
    assert c.get("/admin").status_code == 404
    assert c.post("/api/admin/login", json={"password": "x"}).status_code == 404
    assert c.get("/api/admin/stats").status_code == 401
    monkeypatch.setenv("ADMIN_PASSWORD", "right-one")
    page = c.get("/admin")
    assert page.status_code == 200 and "noindex" in page.headers["x-robots-tag"]
    assert c.post("/api/admin/login", json={"password": "wrong"}).status_code == 401
    assert c.get("/api/admin/stats").status_code == 401
    assert c.post("/api/admin/login", json={"password": "right-one"}).status_code == 200
    assert c.get("/api/admin/stats?days=7&tz=120").status_code == 200
    # a new password signs every browser out
    monkeypatch.setenv("ADMIN_PASSWORD", "another")
    assert c.get("/api/admin/stats").status_code == 401
    assert c.post("/api/admin/logout").status_code == 200
    assert "Disallow: /admin" in c.get("/robots.txt").text


def test_too_many_wrong_passwords_lock_the_form(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "right-one")
    c = TestClient(main.app)
    for _ in range(admin.MAX_TRIES):
        assert c.post("/api/admin/login", json={"password": "guess"}).status_code == 401
    assert c.post("/api/admin/login", json={"password": "right-one"}).status_code == 429  # even the right one, for now


def test_cookie_is_signed_and_expires(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "right-one")
    good = admin.make_cookie()
    assert admin.is_admin(good)
    assert not admin.is_admin(good[:-1] + ("0" if good[-1] != "0" else "1"))
    assert not admin.is_admin(admin.make_cookie(now=time.time() - (admin.ADMIN_DAYS + 1) * 86400))
    assert not admin.is_admin("") and not admin.is_admin(None)


def test_visits_chat_messages_and_store_clicks_are_counted(monkeypatch):
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
        return {"action": "ask", "question": "الهدية لمين؟", "options": ["لراجل", "لست"]}
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    shopper = TestClient(main.app)
    assert shopper.post("/api/event", json={"kind": "visit", "source": "https://www.facebook.com/some/post?x=1",
                                            "mobile": True}).status_code == 200
    assert shopper.post("/api/event", json={"kind": "visit", "source": ""}).status_code == 200
    assert shopper.post("/api/chat", json={"messages": [{"role": "user", "text": "عايز هدية"}]}).json()["action"] == "ask"
    click = {"kind": "click", "url": "https://www.amazon.eg/dp/B0TEST", "store": "Amazon.eg", "title": "Puma shoes",
             "price": 1500, "request": "جزمة بوما", "rank": 0}
    assert shopper.post("/api/event", json=click).status_code == 200
    assert shopper.post("/api/event", json={**click, "url": "javascript:alert(1)"}).status_code == 422
    assert shopper.post("/api/event", json={"kind": "purchase"}).status_code == 422
    assert shopper.post("/api/event", json={**click, "kind": "save", "rank": 250}).status_code == 200
    long = {**click, "url": "https://www.amazon.eg/dp/B0TEST?" + "x" * 3000, "store": "S" * 300, "rank": 1}
    assert shopper.post("/api/event", json={**long, "kind": "share"}).status_code == 200  # long text is cut, not lost
    db.add_event("turn", "other", detail="شاحن ابل", request="شاحن ابل اصلي", action="search", results=0, cached=False)
    db.add_event("turn", "other2", detail="شاحن ابل", request="شاحن ابل اصلي",
                 action="search", results=4, cached=True)

    c = _admin(monkeypatch)
    c.post("/api/event", json={"kind": "visit"})  # the owner's own visits aren't counted
    s = c.get("/api/admin/stats?days=7&tz=120").json()
    totals = {t["kind"]: t for t in s["totals"]}
    assert totals["visit"]["n"] == 2 and totals["visit"]["people"] == 1
    assert totals["turn"]["n"] == 3 and totals["click"]["n"] == 1
    assert totals["save"]["n"] == 1 and totals["share"]["n"] == 1
    assert {a["action"]: a["n"] for a in s["actions"]} == {"search": 2, "ask": 1}
    assert s["searches"][0]["request"] == "شاحن ابل اصلي" and s["searches"][0]["n"] == 2
    assert s["searches"][0]["avg_results"] == 2.0 and s["searches"][0]["cached"] == 1
    assert [e["request"] for e in s["empty_searches"]] == ["شاحن ابل اصلي"]
    assert s["stores"] == [{"store": "Amazon.eg", "n": 1, "people": 1, "affiliate": False}]
    assert s["products"][0]["url"] == click["url"] and s["products"][0]["price"] == 1500
    assert s["ranks"] == [{"rank": 0, "n": 1}]
    assert {r["source"] for r in s["sources"]} == {"facebook.com", ""}
    assert {d["device"] for d in s["devices"]} == {"mobile", "desktop"}
    assert s["recent"][0]["kind"] == "turn" and len(s["recent"][0]["client"]) == 6
    assert s["accounts"]["users"] == 0
    today = int(time.time() + 120 * 60) // 86400
    assert {d["day"] for d in s["daily"]} == {today}


def test_old_events_are_deleted(monkeypatch):
    db.add_event("visit", "a")
    db._db.run("UPDATE events SET at = ?", (time.time() - (db.EVENTS_DAYS + 1) * 86400,))
    monkeypatch.setattr(db, "_pruned", [0.0])
    db.add_event("visit", "b")
    assert [r[0] for r in db._db.run("SELECT client FROM events")] == ["b"]


def test_affiliate_codes_reach_the_page(monkeypatch):
    monkeypatch.setenv("AFFILIATE_TAGS", "www.Amazon.eg: tag=easyfound-21; noon.com: utm_source=x&utm_medium=aff; bad; "
                       "jumia.com.eg: https://ad.admitad.com/g/abc/?ulp={url}; "
                       "btech.com: http://insecure.example/?u={url}; raya.com: https://net.example/no-placeholder")
    assert affiliate.rules() == {"amazon.eg": {"params": {"tag": "easyfound-21"}},
                                 "noon.com": {"params": {"utm_source": "x", "utm_medium": "aff"}},
                                 "jumia.com.eg": {"template": "https://ad.admitad.com/g/abc/?ulp={url}"}}
    assert TestClient(main.app).get("/api/config").json()["affiliate"]["amazon.eg"] == {"params": {"tag": "easyfound-21"}}


def test_affiliate_links(monkeypatch):
    monkeypatch.setenv("AFFILIATE_TAGS", "amazon.eg: tag=ef-21; jumia.com.eg: https://ad.admitad.com/g/abc/?ulp={url}")
    assert affiliate.link("https://www.amazon.eg/dp/B0X?th=1&tag=old") == "https://www.amazon.eg/dp/B0X?th=1&tag=ef-21"
    assert (affiliate.link("https://www.jumia.com.eg/p-1.html?a=1&b=2")
            == "https://ad.admitad.com/g/abc/?ulp=https%3A%2F%2Fwww.jumia.com.eg%2Fp-1.html%3Fa%3D1%26b%3D2")
    assert affiliate.link("https://btech.com/x") == "https://btech.com/x"  # no code for this store: left alone
    assert affiliate.link("https://fakeamazon.eg/x") == "https://fakeamazon.eg/x"  # only the domain and its subdomains
    assert affiliate.link("javascript:alert(1)") == "javascript:alert(1)"
    assert affiliate.store_has_code("amazon.eg") and affiliate.store_has_code("www.jumia.com.eg")
    assert not affiliate.store_has_code("btech.com") and not affiliate.store_has_code("")


def test_alert_email_link_carries_the_code(monkeypatch):
    monkeypatch.setenv("AFFILIATE_TAGS", "amazon.eg: tag=ef-21")
    monkeypatch.setenv("BREVO_API_KEY", "k")
    monkeypatch.setenv("MAIL_FROM", "a@b.c")
    sent = {}

    class Client:
        async def post(self, url, **kw):
            sent.update(kw["json"])
            return httpx.Response(201, request=httpx.Request("POST", url))

    assert asyncio.run(accounts.send_email(Client(), "x@y.z", "Mona", "s", "t", "https://www.amazon.eg/dp/B0X"))
    assert "https://www.amazon.eg/dp/B0X?tag=ef-21" in sent["htmlContent"]


def test_admin_marks_stores_with_a_code(monkeypatch):
    monkeypatch.setenv("AFFILIATE_TAGS", "amazon.eg: tag=ef-21")
    monkeypatch.setattr(main.admin, "is_admin", lambda c: True)
    monkeypatch.setattr(main.db, "usage_stats", lambda days, tz: {"stores": [{"store": "amazon.eg", "n": 3, "people": 2},
                                                                              {"store": "btech.com", "n": 1, "people": 1}]})
    stores = TestClient(main.app).get("/api/admin/stats").json()["stores"]
    assert [s["affiliate"] for s in stores] == [True, False]


def test_every_schema_statement_is_whole():
    """The schema is run one statement at a time, split on ';': a ';' inside a comment would cut a table in half
    (it only shows on Postgres, which runs them one by one)."""
    for stmt in filter(str.strip, db._SCHEMA.split(";")):
        code = "\n".join(line.split("--")[0] for line in stmt.splitlines()).strip()
        assert code.startswith("CREATE"), stmt[:120]


def test_streamed_chat_messages_are_counted(monkeypatch):
    """The page uses the streamed chat: the stream ends right after the answer, which must not cancel the counting."""
    async def fake_ask(prompt, timeout=10, prefer=None, spread=0, budget=None, **kw):
        return {"action": "ask", "question": "الكوتشي لمين؟", "options": ["لراجل", "لست"]}
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    with TestClient(main.app) as c:
        r = c.post("/api/chat/stream", json={"messages": [{"role": "user", "text": "عايز كوتشي"}]})
        assert r.status_code == 200 and '"done"' in r.text
        for _ in range(50):
            if db._db.run("SELECT detail FROM events WHERE kind = 'turn'"):
                break
            time.sleep(0.05)
    assert [r[0] for r in db._db.run("SELECT detail FROM events WHERE kind = 'turn'")] == ["عايز كوتشي"]


def test_server_problems_go_to_the_admin_page_not_to_the_shopper(monkeypatch):
    """mahdy, 22 Nov: LLM limits and search failures belong on /admin; the shopper sees a plain message."""
    from app import problems
    real = problems.__dict__["_save"]
    monkeypatch.setattr(problems, "_last", {})
    real("llm_busy", "step plan: openai/gpt-oss-120b: 429, gemini-flash-latest: TimeoutError", "كوتشي بوما")
    db.add_event("turn", "abc", False, detail="hi", request="hi", action="search", results=0)
    s = _admin(monkeypatch).get("/api/admin/stats?days=7").json()
    assert s["problems"][0]["kind"] == "llm_busy" and s["problems"][0]["n"] == 1
    assert "429" in s["problem_log"][0]["detail"] and s["problem_log"][0]["request"] == "كوتشي بوما"
    assert all(r["kind"] != "problem" for r in s["recent"])
