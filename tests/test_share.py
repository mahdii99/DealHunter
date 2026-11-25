"""Link previews (Facebook, WhatsApp, X) need full addresses in the share tags, the server's own."""
from fastapi.testclient import TestClient

from app import main


def test_share_tags_carry_this_servers_address():
    c = TestClient(main.app, base_url="https://example.test")
    home = c.get("/").text
    assert '<meta property="og:image" content="https://example.test/logo-512.png' in home
    assert '<meta name="twitter:card" content="summary">' in home and main.SHARE_SITE not in home
    deals = c.get("/deals").text
    assert '<meta property="og:url" content="https://example.test/deals">' in deals
    assert 'og:title" content="أحسن العروض' in deals


def test_deal_pictures_come_from_the_store_first():
    html = main._deals_html({"checked_at": 1, "deals": [
        {"url": "https://www.amazon.eg/dp/B0X", "store": "Amazon.eg", "category": "phones", "kind": "drop",
         "pct": 20, "title": "Phone", "price": 100, "was": 125, "image": "https://m.media-amazon.com/images/I/a.jpg"}]},
        "https://example.test")
    assert 'src="https://m.media-amazon.com/images/I/a.jpg"' in html and 'data-alt="/img?u=https%3A' in html


def test_our_warm_up_job_is_not_a_visitor(monkeypatch):
    from starlette.requests import Request
    monkeypatch.setenv("CRON_SECRET", "s3")
    req = lambda h: Request({"type": "http", "headers": [(b"x-cron-secret", h)]})
    assert main._from_cron(req(b"s3")) and not main._from_cron(req(b"nope"))
    monkeypatch.delenv("CRON_SECRET")
    assert not main._from_cron(req(b""))
