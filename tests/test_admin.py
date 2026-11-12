"""The owner's /admin page: password, usage counting (visits, chat messages, store clicks), the stats."""
import os
import time

import pytest
from fastapi.testclient import TestClient

from app import admin, db, llm, main


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
    db.add_event("turn", "other", detail="شاحن ابل", request="شاحن ابل اصلي", action="search", results=0, cached=False)
    db.add_event("turn", "other2", detail="شاحن ابل", request="شاحن ابل اصلي",
                 action="search", results=4, cached=True)

    c = _admin(monkeypatch)
    c.post("/api/event", json={"kind": "visit"})  # the owner's own visits aren't counted
    s = c.get("/api/admin/stats?days=7&tz=120").json()
    totals = {t["kind"]: t for t in s["totals"]}
    assert totals["visit"]["n"] == 2 and totals["visit"]["people"] == 1
    assert totals["turn"]["n"] == 3 and totals["click"]["n"] == 1
    assert {a["action"]: a["n"] for a in s["actions"]} == {"search": 2, "ask": 1}
    assert s["searches"][0]["request"] == "شاحن ابل اصلي" and s["searches"][0]["n"] == 2
    assert s["searches"][0]["avg_results"] == 2.0 and s["searches"][0]["cached"] == 1
    assert [e["request"] for e in s["empty_searches"]] == ["شاحن ابل اصلي"]
    assert s["stores"] == [{"store": "Amazon.eg", "n": 1, "people": 1}]
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
    monkeypatch.setenv("AFFILIATE_TAGS", "www.Amazon.eg: tag=easyfound-21; noon.com: utm_source=x&utm_medium=aff; bad")
    assert main.affiliate_tags() == {"amazon.eg": {"tag": "easyfound-21"},
                                     "noon.com": {"utm_source": "x", "utm_medium": "aff"}}
    assert TestClient(main.app).get("/api/config").json()["affiliate"]["amazon.eg"] == {"tag": "easyfound-21"}


def test_every_schema_statement_is_whole():
    """The schema is run one statement at a time, split on ';': a ';' inside a comment would cut a table in half
    (it only shows on Postgres, which runs them one by one)."""
    for stmt in filter(str.strip, db._SCHEMA.split(";")):
        code = "\n".join(line.split("--")[0] for line in stmt.splitlines()).strip()
        assert code.startswith("CREATE"), stmt[:120]
