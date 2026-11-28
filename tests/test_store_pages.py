"""Store pages open in parallel, sites that never answer stop holding slots, and robot-check pages aren't products
(live 2026-11-27: noon timed out every search and pages queued behind it hit the deadline)."""
import asyncio
import time

import httpx
import pytest

from app import agent, catalog, fetchers


@pytest.fixture(autouse=True)
def _allowed(monkeypatch):
    async def allowed(client, url):
        return True
    monkeypatch.setattr(agent.fetchers, "robots_allows", allowed)
    yield
    fetchers._ROBOTS.clear()


def _run(handler, urls, sem_size=20):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            sem = asyncio.Semaphore(sem_size)
            return await asyncio.gather(*(agent._fetch_page(c, sem, u) for u in urls), return_exceptions=True)
    return asyncio.run(go())


def test_site_that_never_answers_is_skipped_after_two_timeouts(monkeypatch):
    monkeypatch.setattr(agent, "PAGE_TIMEOUT", 0.2)
    calls = []

    async def handler(request):
        calls.append(request.url.host)
        await asyncio.sleep(5)  # a tarpit
        return httpx.Response(200, text="late")

    for _ in range(2):
        assert isinstance(_run(handler, ["https://tarpit.test/p/1"])[0], (asyncio.TimeoutError, TimeoutError))
    t = time.monotonic()
    res = _run(handler, ["https://tarpit.test/p/2"])[0]
    assert isinstance(res, agent.HostSlow) and time.monotonic() - t < 0.1
    assert len(calls) == 2  # the third page wasn't asked for: it goes to the reader service instead


def test_success_resets_the_timeouts(monkeypatch):
    monkeypatch.setattr(agent, "PAGE_TIMEOUT", 0.2)
    slow = [True]

    async def handler(request):
        if slow[0]:
            await asyncio.sleep(5)
        return httpx.Response(200, text="<html><h1>x</h1></html>")

    _run(handler, ["https://shop.test/p/1"])
    slow[0] = False
    _run(handler, ["https://shop.test/p/2"])
    slow[0] = True
    _run(handler, ["https://shop.test/p/3"])
    assert agent._host("https://shop.test/")["slow_until"] == 0.0  # one timeout after a success: not slow yet


def test_pages_open_side_by_side_not_queued(monkeypatch):
    async def handler(request):
        await asyncio.sleep(0.3)
        return httpx.Response(200, text="<html></html>")

    urls = [f"https://s{i}.test/p/1" for i in range(10)]
    t = time.monotonic()
    res = _run(handler, urls)
    assert all(not isinstance(r, Exception) for r in res)
    assert time.monotonic() - t < 1.0  # ten sites at once, not five then five


def test_one_site_gets_at_most_two_requests_at_once():
    now, peak = [0], [0]

    async def handler(request):
        now[0] += 1
        peak[0] = max(peak[0], now[0])
        await asyncio.sleep(0.05)
        now[0] -= 1
        return httpx.Response(200, text="<html></html>")

    _run(handler, [f"https://big.test/p/{i}" for i in range(6)])
    assert peak[0] == agent.HOST_PARALLEL


def test_crawler_name_first_where_only_it_is_let_in():
    agents = []

    def handler(request):
        ua = request.headers["user-agent"]
        agents.append(ua)
        return httpx.Response(200 if ua == catalog.BOT_HEADERS["User-Agent"] else 403, text="<html></html>")

    _run(handler, ["https://jumia.test/p/1"])
    _run(handler, ["https://jumia.test/p/2"])
    assert agents == [agent.HEADERS["User-Agent"], catalog.BOT_HEADERS["User-Agent"], catalog.BOT_HEADERS["User-Agent"]]


def test_robot_check_page_is_not_a_page():
    page = ("<html><head><title>Amazon.eg</title></head><body><form action='/errors/validateCaptcha'>"
            "Type the characters you see in this image</form></body></html>")
    res = _run(lambda r: httpx.Response(200, text=page), ["https://www.amazon.eg/dp/B0TEST"])[0]
    assert isinstance(res, agent.RobotCheck)


def test_product_page_mentioning_captcha_still_reads():
    page = ('<html><script type="application/ld+json">{"@type":"Product","name":"X","offers":{"@type":"Offer",'
            '"price":"100","priceCurrency":"EGP"}}</script><div class="g-recaptcha"></div>challenge-platform</html>')
    res = _run(lambda r: httpx.Response(200, text=page), ["https://shop.test/p/1"])[0]
    assert not isinstance(res, Exception)


def test_site_nobody_can_open_is_left_out_for_a_while():
    agent._note_unreadable({"noon.com"}, set())
    assert not agent.site_skipped("https://www.noon.com/egypt-en/x/p/")
    agent._note_unreadable({"noon.com"}, set())
    assert agent.site_skipped("https://www.noon.com/egypt-en/x/p/")
    agent._note_unreadable(set(), {"noon.com"})  # it opened again (e.g. a later search reached it): back in
    assert not agent.site_skipped("https://www.noon.com/egypt-en/x/p/")


def test_page_probe_needs_the_secret(monkeypatch):
    from starlette.testclient import TestClient
    from app import main
    monkeypatch.setenv("CRON_SECRET", "s")
    c = TestClient(main.app)
    assert c.post("/api/cron/page-probe", params={"urls": "https://shop.test/p/1"}).status_code == 403
    assert c.post("/api/cron/page-probe", params={"urls": "http://127.0.0.1/"},
                  headers={"X-Cron-Secret": "s"}).status_code == 400


def test_reader_copy_of_a_store_error_page_is_no_page():
    def handler(request):
        if "amazon" in str(request.url):
            return httpx.Response(200, text="Title: عذرًا!\n\nURL Source: x\n\nWarning: Target URL returned error 503: "
                                            "Service Unavailable\n\nMarkdown Content:\n" + "x" * 500)
        return httpx.Response(200, text="Title: Shop\n\nMarkdown Content:\n" + "y" * 500)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await fetchers._jina(c, "https://www.amazon.eg/dp/X"), await fetchers._jina(c, "https://s.test/p")
    bad, good = asyncio.run(go())
    assert bad == "" and good.startswith("Title: Shop")
