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
import threading
import time
from datetime import datetime, timedelta, timezone

import httpx

from . import market, problems

log = logging.getLogger("uvicorn.error")
PROVIDERS = ("tavily", "exa", "firecrawl", "brave")
KEYS = {"tavily": "TAVILY_API_KEY", "exa": "EXA_API_KEY", "firecrawl": "FIRECRAWL_API_KEY", "brave": "BRAVE_API_KEY"}
QUOTA_REST = 6 * 3600   # out of quota / bad key: skip it for a while
BUSY_REST = 60          # rate limited / server error
CONTENT_CHARS = 12000   # page text the search service already crawled: read it when we can't open the page
_REST: dict[str, float] = {}
_QUOTA_OUT: dict[str, float] = {}  # service -> until when we skip it because its free quota looked used up
_turn = [0]
# Free searches a month per service (one search = 1 Tavily/Exa credit, 2 Firecrawl credits). The services' own
# dashboards have the exact count: these are for the owner's /admin page. Change with SEARCH_MONTHLY_LIMITS
# ("tavily=1000, firecrawl=500").
MONTHLY_LIMITS = {"tavily": 1000, "exa": 1000, "firecrawl": 500, "brave": 2000}
PERSIST = True     # save the month's counts in the database (off in tests)
FLUSH_EVERY = 30   # seconds between saves
_MONTH: dict[tuple[str, str], int] = {}     # (day, service) -> searches, this month's (and the db's)
_UNSAVED: dict[tuple[str, str], int] = {}
_loaded_months: set[str] = set()
_last_flush = [0.0]


def monthly_limits() -> dict[str, int]:
    out = dict(MONTHLY_LIMITS)
    for part in os.getenv("SEARCH_MONTHLY_LIMITS", "").split(","):
        name, _, n = part.partition("=")
        if name.strip() in out and n.strip().isdigit():
            out[name.strip()] = int(n)
    return out


def _load_month(month: str) -> None:
    """This month's counts saved before a restart."""
    if month in _loaded_months or not PERSIST:
        return
    _loaded_months.add(month)
    try:
        from . import db
        for row in db.search_usage_month(month):
            key = (row["day"], row["service"])
            _MONTH[key] = _MONTH.get(key, 0) + row["calls"]
    except Exception as e:  # database down: count from zero
        log.warning("search usage: could not load %s: %s", month, e)


def _count(service: str) -> None:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _load_month(day[:7])
    for book in (_MONTH, _UNSAVED):
        book[(day, service)] = book.get((day, service), 0) + 1
    if PERSIST and time.time() - _last_flush[0] >= FLUSH_EVERY:
        _last_flush[0] = time.time()
        threading.Thread(target=flush_usage, daemon=True).start()


def flush_usage() -> None:
    """Adds the searches counted since the last save to the database."""
    if not PERSIST or not _UNSAVED:
        return
    rows = [(d, s, n) for (d, s), n in list(_UNSAVED.items())]
    _UNSAVED.clear()
    try:
        from . import db
        db.add_search_usage(rows)
    except Exception as e:  # kept in memory: saved with the next one
        log.warning("search usage: could not save: %s", e)
        for d, s, n in rows:
            _UNSAVED[(d, s)] = _UNSAVED.get((d, s), 0) + n


def quota_report() -> list[dict]:
    """Each configured search service this month, for the owner's /admin page: used of its free monthly searches,
    left, and whether it's skipped now (quota used up or key refused, or busy)."""
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    _load_month(month)
    nxt = (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=32)).replace(day=1)
    limits, out = monthly_limits(), []
    for p in configured():
        used = sum(n for (d, s), n in _MONTH.items() if s == p and d.startswith(month))
        lim = limits.get(p)
        out.append({"name": p, "used": used, "limit": lim, "left": None if lim is None else max(0, lim - used),
                    "resets_at": nxt.timestamp(), "quota_out_until": _QUOTA_OUT.get(p) if _QUOTA_OUT.get(p, 0) > time.time() else None,
                    "busy_until": _REST.get(p) if _REST.get(p, 0) > time.time() else None})
    return out


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
                                "include_raw_content": "markdown",  # page text from Tavily's own crawl, same cost
                                "country": market.current().country_en.lower(),
                                **({"exclude_domains": exclude} if exclude else {}),
                                **({"include_domains": include} if include else {})})
    return r, lambda d: [{"url": x["url"], "title": x.get("title", ""), "snippet": x.get("content", ""),
                          "content": x.get("raw_content") or ""} for x in d.get("results", [])]


async def _exa(client, key, query, n, include, exclude):
    r = await client.post("https://api.exa.ai/search", timeout=15, headers={"x-api-key": key},
                          json={"query": query, "numResults": n, "type": "auto", "userLocation": market.current().code,
                                "contents": {"text": {"maxCharacters": CONTENT_CHARS}},  # page text from Exa's crawl
                                **({"excludeDomains": exclude} if exclude else {}),
                                **({"includeDomains": include} if include else {})})
    return r, lambda d: [{"url": x["url"], "title": x.get("title") or "", "snippet": x.get("summary") or x.get("text") or "",
                          "content": x.get("text") or ""} for x in d.get("results", [])]


async def _firecrawl(client, key, query, n, include, exclude):
    r = await client.post("https://api.firecrawl.dev/v2/search", timeout=20, headers={"Authorization": f"Bearer {key}"},
                          json={"query": _site_query(query, include, exclude), "limit": min(n, 10),  # 2 credits per 10
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
                 exclude_domains: list[str] | None = None, include_domains: list[str] | None = None,
                 only: list[str] | None = None) -> list[dict]:
    """Returns [{"url", "title", "snippet"}]. include_domains = only these sites; exclude_domains = never these.
    only = just these services (e.g. the ones that return page text)."""
    names = [n for n in configured() if not only or n in only]
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
            _count(p)
            try:
                return [{**x, "snippet": (x.get("snippet") or "")[:300], "content": (x.get("content") or "")[:CONTENT_CHARS]}
                        for x in parse(r.json()) if x.get("url")]
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                errors.append(f"{p}: bad reply {type(e).__name__}")
                continue
        errors.append(f"{p} {r.status_code}")
        log.warning("search %s said %s: %s", p, r.status_code, r.text[:150].replace("\n", " "))
        # 401/402/403/432/433 = bad key or the free quota is used up; 429/5xx = busy for now
        quota = r.status_code in (401, 402, 403, 432, 433)
        _REST[p] = time.time() + (QUOTA_REST if quota else BUSY_REST)
        if quota:
            _QUOTA_OUT[p] = _REST[p]
            problems.note("search_error", f"{p} {r.status_code}: free quota used up or key refused")
    problems.note("search_error", "every search service failed: " + ", ".join(errors), request=query)
    raise SearchError("البحث فشل: " + ", ".join(errors))
