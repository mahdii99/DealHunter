"""noon's public product sitemaps as "see it on noon" links without a price (app/noon_index.py)."""
import asyncio
from urllib.parse import quote

import httpx
import pytest
from fastapi.testclient import TestClient

from app import db, llm, noon_index


@pytest.fixture(autouse=True)
def _own_database(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "STATE_FILE", tmp_path / "llm_state.json")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    yield
    db.close()


def _url(slug, sku, ver="1700000000"):
    return (f"<url><loc>https://www.noon.com/egypt-en/{slug}/{sku}/p/</loc><lastmod>2026-11-20T00:00:00+00:00</lastmod>"
            f'<xhtml:link rel="alternate" href="https://www.noon.com/egypt-ar/{slug}/{sku}/p/" hreflang="ar-eg" />'
            f"<image:image><image:loc>https://f.nooncdn.com/p/v{ver}/{sku}_1.jpg</image:loc></image:image></url>")


SITEMAP = ("<?xml version='1.0' encoding='utf-8'?><urlset>" +
           _url("samsung-galaxy-a56-5g-dual-sim-awesome-graphite-8gb-ram-256gb", "N70100001V") +
           _url("samsung-galaxy-a56-5g-dual-sim-awesome-pink-8gb-ram-128gb", "N70100002V") +
           _url("samsung-galaxy-a56-case-shockproof-cover-black", "Z11111111111111111111Z") +
           _url("samsung-galaxy-a36-5g-dual-sim-8gb-ram-256gb", "N70100003V") +
           _url("refurbished-galaxy-a56-5g-graphite-8gb-ram-256gb", "Z33333333333333333333Z") +
           _url("galaxy-a56-5g-dual-sim-awesome-olive-8gb-ram-256gb-middle-east-version", "Z44444444444444444444Z") +
           _url("", "Z22222222222222222222Z") + _url("None", "N70100004A") +
           "<url><loc>https://www.noon.com/egypt-en/samsung-tv/N1/p/</loc></url></urlset>")
INDEX = ("<sitemapindex><sitemap><loc>https://www.noon.com/sitemap/sitemap-eg-pdp-0.xml</loc></sitemap>"
         "<sitemap><loc>https://www.noon.com/sitemap/sitemap-ae-pdp-0.xml</loc></sitemap></sitemapindex>")


def _build():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        assert request.headers["user-agent"].startswith("DealHunterBot/")
        return httpx.Response(200, text=INDEX if request.url.path.endswith("sitemap-index.xml") else SITEMAP)
    report = asyncio.run(noon_index.build(mk="EG", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    return report, seen


def test_sitemap_read_skips_products_without_a_name():
    rows = noon_index.read_sitemap(SITEMAP)
    assert [r[1] for r in rows] == ["N70100001V", "N70100002V", "Z11111111111111111111Z", "N70100003V", "Z33333333333333333333Z", "Z44444444444444444444Z", "N1"]
    assert rows[0] == ("samsung", "N70100001V", "samsung-galaxy-a56-5g-dual-sim-awesome-graphite-8gb-ram-256gb", "1700000000")
    assert rows[-1][3] == ""  # no picture listed


def test_build_reads_only_the_markets_files_and_replaces_them():
    report, seen = _build()
    assert report["products"] == 7 and report["files"] == {0: 7}
    assert seen[-1].endswith("/sitemap-eg-pdp-0.xml") and not any("ae-pdp" in u for u in seen)
    _build()  # next week's run puts the file's products in place of the old ones, not next to them
    assert db.noon_count("eg") == 7


def test_links_fit_the_request_and_leave_out_accessories():
    _build()
    got = noon_index.links({"brand": "Samsung", "kind": "smartphone", "model": "Galaxy A56", "size": "256GB"}, mk="eg")
    # noon's own catalogue product first, then a seller's listing found by the product line; never the refurbished one
    assert [g["url"].split("/")[-3] for g in got] == ["N70100001V", "Z44444444444444444444Z"]
    assert got[0]["title"].startswith("Samsung Galaxy A56 5G") and "price" not in got[0]
    assert got[0]["image"] == "https://f.nooncdn.com/p/v1700000000/N70100001V_1.jpg"
    assert got[0]["go"] == "/go?src=noon&u=" + quote(got[0]["url"], safe="")
    both = noon_index.links({"brand": "Samsung", "kind": "smartphone", "model": "Galaxy A56"}, mk="eg")
    assert len(both) == 3  # the colours, never the case
    case = noon_index.links({"brand": "Samsung", "kind": "phone case", "model": "Galaxy A56"}, mk="eg")
    assert any("case" in g["url"] for g in case)
    assert noon_index.links({"brand": "Samsung", "model": "Galaxy A56"}, mk="EG") == both  # the site's market code
    assert noon_index.links({"kind": "smartphone"}, mk="eg") == []  # no brand: too vague to point anywhere
    assert noon_index.links({"brand": "Samsung", "model": "A56"}, mk="sa") == []  # no index for that market yet


def test_go_opens_only_noon_products_from_the_index(monkeypatch):
    from app import main
    _build()
    c = TestClient(main.app)
    good = "https://www.noon.com/egypt-ar/samsung-galaxy-a36-5g-dual-sim-8gb-ram-256gb/N70100003V/p/"
    r = c.get("/go", params={"u": good, "src": "noon"}, follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith(good)
    made_up = "https://www.noon.com/egypt-ar/samsung-galaxy-a36-5g-dual-sim-8gb-ram-256gb/N99999999V/p/"
    r = c.get("/go", params={"u": made_up, "src": "noon"}, follow_redirects=False)
    assert r.status_code == 200 and "مش منتج احنا عرضناه" in r.text


def test_index_endpoint_needs_the_cron_secret(monkeypatch):
    from app import main
    monkeypatch.setenv("CRON_SECRET", "s3cret")
    assert TestClient(main.app).post("/api/cron/noon-index").status_code == 403
