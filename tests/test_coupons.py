"""Promo codes (app/coupons.py): only from Admitad's feed or added by the owner, never shown once expired."""
import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import admin, coupons, db, deals, llm, main


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    for name in ("DATABASE_URL", "AFFILIATE_TAGS", "ADMITAD_CLIENT_ID", "ADMITAD_CLIENT_SECRET", "ADMITAD_WEBSITE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    admin._tries.clear()
    yield
    db.close()


def _admin(monkeypatch) -> TestClient:
    monkeypatch.setenv("ADMIN_PASSWORD", "right-one")
    c = TestClient(main.app)
    assert c.post("/api/admin/login", json={"password": "right-one"}).status_code == 200
    return c


GOOD = {"store": "https://www.noon.com/egypt-ar/", "code": "SAVE10", "title": "10% على الإلكترونيات",
        "link": "https://www.noon.com/egypt-ar/", "where": "ArabClicks"}


def test_only_the_owner_adds_codes_and_the_page_shows_where_each_came_from(monkeypatch):
    anyone = TestClient(main.app)
    assert anyone.post("/api/admin/coupons", json=GOOD).status_code == 401
    c = _admin(monkeypatch)
    assert c.post("/api/admin/coupons", json={**GOOD, "link": "http://noon.com/x"}).status_code == 400
    assert c.post("/api/admin/coupons", json={**GOOD, "expires": "2020-01-01"}).status_code == 400
    assert c.post("/api/admin/coupons", json=GOOD).status_code == 200
    page = anyone.get("/coupons").text
    assert "SAVE10" in page and "نون" in page and "من ArabClicks" in page
    got = anyone.get("/api/coupons").json()["coupons"]
    assert got[0]["store"] == "noon.com" and got[0]["code"] == "SAVE10"
    cid = c.get("/api/admin/coupons").json()["coupons"][0]["id"]
    assert anyone.delete(f"/api/admin/coupons/{cid}").status_code == 401
    assert c.delete(f"/api/admin/coupons/{cid}").status_code == 200
    assert "SAVE10" not in anyone.get("/coupons").text


def test_an_expired_code_is_never_shown():
    db.add_coupon("EG", "noon.com", "OLD", "قديم", "https://noon.com", "owner:noon", time.time() - 60)
    db.add_coupon("EG", "noon.com", "NEW", "جديد", "https://noon.com", "owner:noon", time.time() + 3600)
    assert [c["code"] for c in coupons.current()] == ["NEW"]


def test_owner_links_carry_our_affiliate_code(monkeypatch):
    monkeypatch.setenv("AFFILIATE_TAGS", "amazon.eg: tag=me-21")
    db.add_coupon("EG", "amazon.eg", "X", "خصم", "https://www.amazon.eg/deals", "owner:amazon", None)
    assert coupons.current()[0]["link"].endswith("tag=me-21")


def test_admitad_feed_replaces_its_own_codes_and_skips_unusable_ones(monkeypatch):
    for k, v in (("ADMITAD_CLIENT_ID", "id"), ("ADMITAD_CLIENT_SECRET", "s"), ("ADMITAD_WEBSITE_ID", "42")):
        monkeypatch.setenv(k, v)
    db.add_coupon("EG", "noon.com", "MINE", "كود المالك", "https://noon.com", "owner:noon", None)
    db.replace_feed_coupons("EG", "admitad", [{"store": "noon.com", "code": "GONE", "title": "x", "link": "https://a"}])
    feed = [
        {"id": 1, "name": "Extra 15% off", "promocode": "NOON15", "discount": "15%", "date_end": "2099-01-01T00:00:00",
         "goto_link": "https://ad.admitad.com/g/1/", "campaign": {"site_url": "https://www.noon.com/egypt-en/"}},
        {"id": 2, "name": "Free delivery", "promocode": "NOT REQUIRED", "date_end": None,
         "goto_link": "https://ad.admitad.com/g/2/", "campaign": {"site_url": "https://www.jumia.com.eg"}},
        {"id": 3, "name": "Ended", "promocode": "OLD", "date_end": "2001-01-01T00:00:00",
         "goto_link": "https://ad.admitad.com/g/3/", "campaign": {"site_url": "https://noon.com"}},
    ]

    def answer(request: httpx.Request):
        if request.url.path == "/token/":
            assert request.headers["authorization"].startswith("Basic ")
            return httpx.Response(200, json={"access_token": "t"})
        assert request.url.path == "/coupons/website/42/" and request.url.params["region"] == "EG"
        return httpx.Response(200, json={"results": feed})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
            return await coupons.read_admitad(client)
    got = asyncio.run(run())
    assert [(c["store"], c["code"]) for c in got] == [("noon.com", "NOON15"), ("jumia.com.eg", "")]
    assert "15%" in got[0]["title"]
    db.replace_feed_coupons("EG", "admitad", got)
    codes = {c["code"]: c for c in coupons.current()}
    assert set(codes) == {"MINE", "NOON15", ""}
    assert codes["NOON15"]["link"] == "https://ad.admitad.com/g/1/" and "Admitad" in codes["NOON15"]["from"]


def test_the_feed_is_off_without_keys():
    assert asyncio.run(coupons.refresh()) == {"admitad": "off"}


def test_deals_page_has_store_and_category_filters_and_a_verified_badge(monkeypatch):
    payload = {"checked_at": time.time(), "deals": [
        {"title": "Galaxy A56", "store": "Jumia", "url": "https://www.jumia.com.eg/a.html", "image": None,
         "price": 23000, "was": 26000, "pct": 12, "kind": "drop", "category": "phones"},
        {"title": "LG TV", "store": "Amazon.eg", "url": "https://www.amazon.eg/dp/B1", "image": None,
         "price": 9000, "was": 12000, "pct": 25, "kind": "store", "category": "tvs"}]}
    monkeypatch.setattr(deals, "current", lambda: payload)
    page = TestClient(main.app).get("/deals").text
    assert 'data-f="store"' in page and 'data-v="amazon.eg"' in page and 'data-v="jumia.com.eg"' in page
    assert 'data-f="cat"' in page and "-25%" in page and "السعر متأكد منه" in page
    assert 'href="/coupons"' in page
