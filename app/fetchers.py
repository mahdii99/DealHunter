"""Second way to read a store page when our own request fails (blocked, timed out, server error, or a page
that only renders with JavaScript).

It works like the page reader Claude uses: a reader service fetches the page from its own infrastructure and
returns the readable text, which the LLM then reads for products and prices.
  1. Tavily Extract (same TAVILY_API_KEY as search; free credits, 5 pages = 1 credit)
  2. Jina Reader (https://r.jina.ai, free, no key needed; JINA_API_KEY optional for higher limits)

Site rules are respected: a page is only sent to a reader if the store's robots.txt allows crawling it.
"""
import asyncio
import logging
import os
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger("uvicorn.error")
_ROBOTS: dict[str, tuple[float, RobotFileParser | None]] = {}
ROBOTS_TTL = 6 * 3600
MIN_TEXT = 300  # less than this is an empty shell / error page


async def _robots_via_reader(client: httpx.AsyncClient, robots_url: str) -> RobotFileParser | None:
    """The store didn't answer us (timeout/blocked): read its robots.txt through the reader service instead."""
    try:
        r = await client.get("https://r.jina.ai/" + robots_url, timeout=15,
                             headers={"Accept": "text/plain", "X-Return-Format": "text"})
    except httpx.HTTPError:
        return None
    if r.status_code != 200 or "user-agent" not in r.text.lower():
        return None
    rp = RobotFileParser()
    rp.parse(r.text.splitlines())
    return rp


async def robots_allows(client: httpx.AsyncClient, url: str) -> bool:
    """Standard crawler rules: robots.txt missing (4xx) = allowed; unreachable (5xx/timeout) = read it through
    the reader service, and if that fails too = not allowed."""
    parts = urlparse(url)
    host = f"{parts.scheme}://{parts.netloc}"
    hit = _ROBOTS.get(host)
    if not hit or time.time() - hit[0] > ROBOTS_TTL:
        rp: RobotFileParser | None
        try:
            r = await client.get(host + "/robots.txt", timeout=8, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (compatible; price-agent)"})
            if r.status_code >= 500:
                rp = None
            else:
                rp = RobotFileParser()
                rp.parse(r.text.splitlines() if r.status_code < 400 else [])
        except httpx.HTTPError:
            rp = None
        if rp is None:
            rp = await _robots_via_reader(client, host + "/robots.txt")
        _ROBOTS[host] = hit = (time.time(), rp)
    rp = hit[1]
    return rp is not None and rp.can_fetch("*", url)


async def _tavily_extract(client: httpx.AsyncClient, urls: list[str]) -> dict[str, str]:
    key = os.getenv("TAVILY_API_KEY")
    if not key or not urls:
        return {}
    r = await client.post("https://api.tavily.com/extract", timeout=15,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"urls": urls, "extract_depth": "basic", "format": "markdown"})
    if r.status_code != 200:
        log.warning("tavily extract %s: %s", r.status_code, r.text[:150])
        return {}
    return {x["url"]: x.get("raw_content") or "" for x in r.json().get("results", [])}


async def _jina(client: httpx.AsyncClient, url: str) -> str:
    headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
    if key := os.getenv("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    r = await client.get("https://r.jina.ai/" + url, headers=headers, timeout=15)
    return r.text if r.status_code == 200 else ""


async def read_pages(client: httpx.AsyncClient, urls: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Returns ({url: page text}, {url: why it couldn't be read})."""
    allowed, why = [], {}
    checks = await asyncio.gather(*(robots_allows(client, u) for u in urls))
    for u, ok in zip(urls, checks):
        (allowed.append(u) if ok else why.__setitem__(u, "robots.txt disallows"))
    texts: dict[str, str] = {}
    try:
        got = await _tavily_extract(client, allowed)
        texts.update({u: t for u, t in got.items() if len(t) >= MIN_TEXT})
    except (httpx.HTTPError, ValueError) as e:
        log.warning("tavily extract failed: %s", type(e).__name__)
    rest = [u for u in allowed if u not in texts]
    if rest:
        jina = await asyncio.gather(*(_jina(client, u) for u in rest), return_exceptions=True)
        for u, t in zip(rest, jina):
            if isinstance(t, str) and len(t) >= MIN_TEXT:
                texts[u] = t
            else:
                why[u] = "reader couldn't open it"
    log.info("reader: opened %d of %d pages %s", len(texts), len(urls), list(map(lambda u: urlparse(u).netloc, texts)))
    return texts, why
