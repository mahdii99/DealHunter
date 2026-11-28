"""The server keeps answering (Render's health check gives 5 s) while a search reads big store pages."""
import asyncio
import threading
import time

import httpx

from app import agent, fetchers, main


def test_health_and_home_page_answer_on_the_event_loop():
    # async endpoints: never queued behind busy worker threads
    import inspect
    assert inspect.iscoroutinefunction(main.health) and inspect.iscoroutinefunction(main.index)


def test_store_pages_are_read_off_the_event_loop(monkeypatch):
    async def allowed(client, url):
        return True
    monkeypatch.setattr(agent.fetchers, "robots_allows", allowed)
    where = []
    slow = agent.extract_products

    def heavy(html, store, url):  # a big page on a slow CPU
        where.append(threading.current_thread() is threading.main_thread())
        time.sleep(0.6)
        return slow(html, store, url)
    monkeypatch.setattr(agent, "extract_products", heavy)
    page = "<html><body>" + "<div>x</div>" * 1000 + "</body></html>"

    async def go():
        lags = []

        async def ping():  # what /health needs: the loop free every few moments
            while True:
                t = time.monotonic()
                await asyncio.sleep(0.05)
                lags.append(time.monotonic() - t - 0.05)
        pinger = asyncio.create_task(ping())
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=page))) as c:
            await asyncio.gather(*(agent._fetch_page(c, asyncio.Semaphore(4), f"https://shop{i}.example/p/{i}")
                                   for i in range(4)))
        pinger.cancel()
        return lags
    lags = asyncio.run(go())
    assert where and not any(where)  # parsed in worker threads
    assert max(lags) < 0.4  # the loop was never held for a page's parsing


def test_huge_pages_are_cut():
    r = httpx.Response(200, text="<html>" + "a" * (agent.MAX_PAGE_CHARS + 10), request=httpx.Request("GET", "https://x.example/p/1"))
    _, html = agent._parse_page("https://x.example/p/1", r)
    assert len(html) == agent.MAX_PAGE_CHARS

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, text="b" * (fetchers.MAX_TEXT_CHARS + 5)))) as c:
            return await fetchers._jina(c, "https://x.example/p/1")
    assert len(asyncio.run(go())) == fetchers.MAX_TEXT_CHARS


def test_page_text_stays_fast_on_broken_pages():
    from app import extract, utils
    n = 200_000
    broken = ["<" * n, "<div " * (n // 5), "<script>" * (n // 8), "](https://a.b/" + "c" * n, "[" * n, "![" * (n // 2),
              '<a href="x">' * (n // 12), "<h1>" * (n // 4)]
    for page in broken:
        t = time.monotonic()
        agent._page_text(page, ["Toshiba"], 3000, "https://www.amazon.eg/dp/x")
        extract._visible_text(page)
        extract.read_text_copy(page, "Toshiba")
        extract.page_title(page)
        list(utils.script_bodies(page))
        assert time.monotonic() - t < 2, page[:20]


def test_linear_helpers_match_the_old_patterns():
    from app import utils
    html = '<p>a</p><SCRIPT type="x">var s = "<b>";</script><style>b{}</style><h1 class="t">Name <b>X</b></h1><script>open'
    assert utils.drop_blocks(html) == "<p>a</p>  <h1 class=\"t\">Name <b>X</b></h1> "
    assert utils.strip_tags("<p>a</p><b>b</b>") == " a  b "
    assert utils.first_block(html, "h1").group(1) == "Name <b>X</b>"
    assert list(utils.script_bodies(html)) == ['var s = "<b>";', "open"]


def test_the_server_starts_without_waiting_for_the_database(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(main.agent.llm, "load_reported", lambda: time.sleep(3))
    t = time.monotonic()
    with TestClient(main.app) as c:
        assert c.get("/health").json() == {"ok": True}
        assert time.monotonic() - t < 2


def test_a_query_stuck_ahead_never_holds_the_event_loop():
    from app import db
    db._db._lock.acquire()  # another thread's query hangs with the connection
    try:
        async def go():
            t = time.monotonic()
            try:
                db.cached_search("x", 60)
            except TimeoutError:
                pass
            return time.monotonic() - t
        assert asyncio.run(go()) < db.LOOP_LOCK_WAIT + 0.5
    finally:
        db._db._lock.release()
