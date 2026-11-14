"""Second way to read a store page when our own request fails (blocked, timed out, server error, or a page
that only renders with JavaScript).

It works like the page readers AI assistants use: a reader service fetches the page from its own infrastructure and
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
import re

import httpx

log = logging.getLogger("uvicorn.error")
_ROBOTS: dict[str, tuple[float, "Robots | None"]] = {}
ROBOTS_TTL = 6 * 3600
MIN_TEXT = 300  # less than this is an empty shell / error page


class Robots:
    """robots.txt read the standard way (RFC 9309, as Google and Bing read it): the group for any crawler ("*"),
    `*` and `$` in paths, and the most specific (longest) matching rule wins, Allow on a tie. Python's own
    parser takes the first rule listed and no wildcards, which wrongly blocks pages on big stores whose files
    list a broad Disallow before a longer Allow."""

    def __init__(self, text: str):
        self.rules: list[tuple[int, bool, re.Pattern, str]] = []
        groups, agents, in_rules = [], [], False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (x.strip() for x in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:
                    agents, in_rules = [], False
                agents.append(value.lower())
                if len(agents) == 1:
                    groups.append((agents, []))
            elif key in ("allow", "disallow") and groups:
                in_rules = True
                groups[-1][1].append((key == "allow", value))
        for names, rules in groups:
            if "*" in names:
                for allow, path in rules:
                    if path:  # an empty Disallow allows everything
                        self.rules.append((len(path), allow, self._pattern(path), path))

    @staticmethod
    def _pattern(path: str) -> re.Pattern:
        end = path.endswith("$")
        body = re.escape(path.rstrip("$")).replace(r"\*", ".*")
        return re.compile(body + ("$" if end else ""))

    def rule_for(self, url: str) -> tuple[bool, str]:
        u = urlparse(url)
        target = (u.path or "/") + ("?" + u.query if u.query else "")
        best = None
        for length, allow, pattern, path in self.rules:
            if pattern.match(target) and (best is None or (length, allow) > (best[0], best[1])):
                best = (length, allow, path)
        return (True, "") if best is None else (best[1], ("Allow: " if best[1] else "Disallow: ") + best[2])

    def can_fetch(self, url: str) -> bool:
        return self.rule_for(url)[0]


async def _robots_via_reader(client: httpx.AsyncClient, robots_url: str) -> "Robots | None":
    """The store didn't answer us (timeout/blocked): read its robots.txt through the reader service instead."""
    try:
        r = await client.get("https://r.jina.ai/" + robots_url, timeout=6,
                             headers={"Accept": "text/plain", "X-Return-Format": "text"})
    except httpx.HTTPError:
        return None
    if r.status_code != 200 or "user-agent" not in r.text.lower():
        return None
    return Robots(r.text)


async def robots_allows(client: httpx.AsyncClient, url: str) -> bool:
    """Standard crawler rules: robots.txt missing (4xx) = allowed; unreachable (5xx/timeout) = read it through
    the reader service, and if that fails too = not allowed."""
    parts = urlparse(url)
    host = f"{parts.scheme}://{parts.netloc}"
    hit = _ROBOTS.get(host)
    if not hit or time.time() - hit[0] > ROBOTS_TTL:
        rp: Robots | None
        try:
            r = await client.get(host + "/robots.txt", timeout=4, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (compatible; price-agent)"})
            rp = None if r.status_code >= 500 else Robots(r.text if r.status_code < 400 else "")
        except httpx.HTTPError:
            rp = None
        if rp is None:
            rp = await _robots_via_reader(client, host + "/robots.txt")
        _ROBOTS[host] = hit = (time.time(), rp)
    rp = hit[1]
    return rp is not None and rp.can_fetch(url)


def robots_reason(url: str) -> str:
    """Why robots_allows said no (after it ran for this site): the rule that matched, or that we couldn't read
    the file at all."""
    parts = urlparse(url)
    hit = _ROBOTS.get(f"{parts.scheme}://{parts.netloc}")
    if not hit:
        return "robots.txt not read yet"
    if hit[1] is None:
        return "robots.txt unreachable"
    return hit[1].rule_for(url)[1] or "allowed"


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


async def read_pages(client: httpx.AsyncClient, urls: list[str], deadline: float = 12) -> tuple[dict[str, str], dict[str, str]]:
    """Returns ({url: page text}, {url: why it couldn't be read}) within `deadline` seconds.
    Both readers work on every page at the same time and the first good text wins, so one slow reader or one
    slow page doesn't cost the others; whatever is ready when time is up is used."""
    t0 = time.monotonic()
    why: dict[str, str] = {}

    async def check(u):
        try:
            return await asyncio.wait_for(robots_allows(client, u), timeout=max(2.0, deadline / 2))
        except asyncio.TimeoutError:
            return False

    allowed = []
    for u, ok in zip(urls, await asyncio.gather(*(check(u) for u in urls))):
        (allowed.append(u) if ok else why.__setitem__(u, "robots.txt disallows (or unreachable)"))
    if not allowed:
        return {}, why
    tavily = asyncio.ensure_future(_tavily_extract(client, allowed))
    jina = {u: asyncio.ensure_future(_jina(client, u)) for u in allowed}
    pending = {tavily, *jina.values()}
    texts: dict[str, str] = {}
    while pending and len(texts) < len(allowed):
        left = deadline - (time.monotonic() - t0)
        if left <= 0:
            break
        done, pending = await asyncio.wait(pending, timeout=left, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled() or task.exception() is not None:
                if task is tavily:
                    log.warning("tavily extract failed: %s", type(task.exception()).__name__)
                continue
            if task is tavily:
                texts.update({u: x for u, x in task.result().items() if u in jina and len(x) >= MIN_TEXT and u not in texts})
            else:
                u = next(k for k, v in jina.items() if v is task)
                if len(task.result()) >= MIN_TEXT and u not in texts:
                    texts[u] = task.result()
    for task in pending:
        task.cancel()
    for u in allowed:
        if u not in texts:
            why[u] = "reader couldn't open it" if time.monotonic() - t0 < deadline else "reader too slow"
    log.info("reader: opened %d of %d pages in %.1fs %s", len(texts), len(urls), time.monotonic() - t0,
             [urlparse(u).netloc for u in texts])
    return texts, why
