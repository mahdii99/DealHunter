"""Web search through free search APIs. The agent finds stores by itself instead of a fixed list.

Set one or more keys in .env (all free, no card). With several keys, searches are shared between them (more
free searches per month), and when one is out of quota or fails, the next one answers.
  TAVILY_API_KEY     https://app.tavily.com     1000 searches/month
  EXA_API_KEY        https://dashboard.exa.ai   free monthly credit (about 1000+ searches)
  FIRECRAWL_API_KEY  https://www.firecrawl.dev  1000 credits/month (2 credits per search)
  BRAVE_API_KEY      https://api.search.brave.com  (new accounts now need a card for the free credit)
"""
import logging
import os
import time

import httpx

from . import market

log = logging.getLogger("uvicorn.error")
PROVIDERS = ("tavily", "exa", "firecrawl", "brave")
KEYS = {"tavily": "TAVILY_API_KEY", "exa": "EXA_API_KEY", "firecrawl": "FIRECRAWL_API_KEY", "brave": "BRAVE_API_KEY"}
QUOTA_REST = 6 * 3600   # out of quota / bad key: skip it for a while
BUSY_REST = 60          # rate limited / server error
_REST: dict[str, float] = {}
_turn = [0]


class SearchError(Exception):
    pass


def configured() -> list[str]:
    return [p for p in PROVIDERS if os.getenv(KEYS[p])]


def available() -> bool:
    return bool(configured())


def _site_query(query: str, include: list[str] | None, exclude: list[str] | None) -> str:
    """For engines without domain filters: Google-style site: operators."""
    q = query + "".join(f" -site:{d}" for d in exclude or [])
    if include:
        q += " (" + " OR ".join(f"site:{d}" for d in include) + ")"
    return q


async def _tavily(client, key, query, n, include, exclude):
    r = await client.post("https://api.tavily.com/search", timeout=15, headers={"Authorization": f"Bearer {key}"},
                          json={"query": query, "max_results": n, "search_depth": "basic", "topic": "general",
                                "country": market.current().country_en.lower(),
                                **({"exclude_domains": exclude} if exclude else {}),
                                **({"include_domains": include} if include else {})})
    return r, lambda d: [{"url": x["url"], "title": x.get("title", ""), "snippet": x.get("content", "")}
                         for x in d.get("results", [])]


async def _exa(client, key, query, n, include, exclude):
    r = await client.post("https://api.exa.ai/search", timeout=15, headers={"x-api-key": key},
                          json={"query": query, "numResults": n, "type": "auto", "userLocation": market.current().code,
                                **({"excludeDomains": exclude} if exclude else {}),
                                **({"includeDomains": include} if include else {})})
    return r, lambda d: [{"url": x["url"], "title": x.get("title") or "", "snippet": x.get("text") or x.get("summary") or ""}
                         for x in d.get("results", [])]


async def _firecrawl(client, key, query, n, include, exclude):
    r = await client.post("https://api.firecrawl.dev/v2/search", timeout=20, headers={"Authorization": f"Bearer {key}"},
                          json={"query": _site_query(query, include, exclude), "limit": n,
                                "country": market.current().code})

    def parse(d):
        data = d.get("data")
        items = data.get("web", []) if isinstance(data, dict) else data or []
        return [{"url": x["url"], "title": x.get("title", ""), "snippet": x.get("description", "")} for x in items]
    return r, parse


async def _brave(client, key, query, n, include, exclude):
    r = await client.get("https://api.search.brave.com/res/v1/web/search", timeout=15,
                         headers={"X-Subscription-Token": key, "Accept": "application/json"},
                         params={"q": _site_query(query, include, exclude), "count": n, "country": market.current().code})
    return r, lambda d: [{"url": x["url"], "title": x.get("title", ""), "snippet": x.get("description", "")}
                         for x in d.get("web", {}).get("results", [])]


_CALLS = {"tavily": _tavily, "exa": _exa, "firecrawl": _firecrawl, "brave": _brave}


async def search(client: httpx.AsyncClient, query: str, max_results: int = 8,
                 exclude_domains: list[str] | None = None, include_domains: list[str] | None = None) -> list[dict]:
    """Returns [{"url", "title", "snippet"}]. include_domains = only these sites; exclude_domains = never these."""
    names = configured()
    if not names:
        raise SearchError("مفيش مفتاح بحث (TAVILY_API_KEY / EXA_API_KEY / FIRECRAWL_API_KEY) في .env")
    _turn[0] += 1  # take turns, so each free quota lasts longer
    k = _turn[0] % len(names)
    now = time.time()
    order = [p for p in names[k:] + names[:k] if _REST.get(p, 0) <= now] or names
    errors = []
    for p in order:
        try:
            r, parse = await _CALLS[p](client, os.environ[KEYS[p]], query, max_results, include_domains, exclude_domains)
        except httpx.HTTPError as e:
            errors.append(f"{p}: {type(e).__name__}")
            _REST[p] = time.time() + BUSY_REST
            continue
        if r.status_code == 200:
            try:
                return [{**x, "snippet": (x.get("snippet") or "")[:300]} for x in parse(r.json()) if x.get("url")]
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                errors.append(f"{p}: bad reply {type(e).__name__}")
                continue
        errors.append(f"{p} {r.status_code}")
        log.warning("search %s said %s: %s", p, r.status_code, r.text[:150].replace("\n", " "))
        # 401/402/403/432/433 = bad key or the free quota is used up; 429/5xx = busy for now
        _REST[p] = time.time() + (QUOTA_REST if r.status_code in (401, 402, 403, 432, 433) else BUSY_REST)
    raise SearchError("البحث فشل: " + ", ".join(errors))
