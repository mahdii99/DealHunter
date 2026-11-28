"""Web search through free search APIs. The agent finds stores by itself instead of a fixed list.

Free services come first, the ones with a small monthly quota only when those fail or find nothing:
  keenable           https://keenable.ai   no key needed (1000 searches/hour per server); with KEENABLE_API_KEY
                                           100,000 a month. Off with SEARCH_KEYLESS=0.
  LANGSEARCH_API_KEY https://langsearch.com  free plan, daily allowance
Then the monthly-quota ones (all free, no card). With several keys, searches are shared between them, and when one
is out of quota or fails, the next one answers.
  TAVILY_API_KEY     https://app.tavily.com     1000 searches/month
  EXA_API_KEY        https://dashboard.exa.ai   free monthly credit (about 1000+ searches)
  FIRECRAWL_API_KEY  https://www.firecrawl.dev  1000 credits/month (2 credits per search)
  BRAVE_API_KEY      https://api.search.brave.com  (new accounts now need a card for the free credit)
"""
import asyncio
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import httpx

from . import market, problems

log = logging.getLogger("uvicorn.error")
PROVIDERS = ("keenable", "langsearch", "tavily", "exa", "firecrawl", "brave")
KEYS = {"keenable": "KEENABLE_API_KEY", "langsearch": "LANGSEARCH_API_KEY", "tavily": "TAVILY_API_KEY",
        "exa": "EXA_API_KEY", "firecrawl": "FIRECRAWL_API_KEY", "brave": "BRAVE_API_KEY"}
# Free (no monthly quota worth counting): asked first. The monthly-quota services only answer when these fail or
# find nothing, so their few free searches a month last.
FREE_FIRST = ("keenable", "langsearch")
KEYLESS = os.getenv("SEARCH_KEYLESS", "1") != "0"  # keenable's keyless search (per server address, no sign-up)
KEYLESS_SERVICES = ("keenable",)
APP_TITLE = "DealHunter"
KEENABLE_TEXT = 3000          # characters of each result's page text
KEENABLE_FRESH = 3 * 86400    # its page copy is used (when we can't open the page) only if crawled this lately  # keenable's keyless search asks callers to name themselves
# The same search (words + sites) is answered from the database for a while: shoppers ask for the same products,
# and a chat's next turns repeat its searches. Store pages found are opened and their prices checked live anyway.
CACHE_TTL = float(os.getenv("SEARCH_CACHE_HOURS", "12")) * 3600
CACHE_PREFIX = "__web__:"
CACHE_TEXT = 6000      # page text kept per cached result (the database is small)
CACHE_PRUNE_EVERY = 3600
_CACHE: dict[str, tuple[float, list[dict]]] = {}
_last_prune = [0.0]
QUOTA_REST = 6 * 3600   # out of quota / bad key: skip it for a while
BUSY_REST = 60          # rate limited / server error
CONTENT_CHARS = 12000   # page text the search service already crawled: read it when we can't open the page
_REST: dict[str, float] = {}
_QUOTA_OUT: dict[str, float] = {}  # service -> until when we skip it because its free quota looked used up
_turn = [0]
# Free searches a month per service (one search = 1 Tavily/Exa credit, 2 Firecrawl credits). The services' own
# dashboards have the exact count: these are for the owner's /admin page. Change with SEARCH_MONTHLY_LIMITS
# ("tavily=1000, firecrawl=500").
MONTHLY_LIMITS = {"keenable": 100000, "tavily": 1000, "exa": 1000, "firecrawl": 500, "brave": 2000}
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


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _load_month(month: str) -> None:
    """This month's counts saved before a restart."""
    if month in _loaded_months or not PERSIST:
        return
    _loaded_months.add(month)
    if _on_event_loop():  # the database can be slow: never wait for it on the event loop (the server stops answering)
        threading.Thread(target=_load_month_now, args=(month,), daemon=True).start()
        return
    _load_month_now(month)


def _load_month_now(month: str) -> None:
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


# ---------- what the services themselves say ----------
# Our count misses searches made with the same key elsewhere (the owner's computer), so the services' own usage
# pages win where they have one: Tavily /usage, Firecrawl /v2/team/credit-usage. Exa has none: our count plus
# "used up" when it refuses. A service that said "used up" stays skipped across restarts (db quota_state).
LIVE_TTL = 600  # seconds the services' own numbers are kept before asking again
_LIVE: dict[str, tuple[float, dict | None]] = {}
_out_loaded = [False]


def _num(v) -> int | None:
    try:
        return None if v is None else int(float(v))
    except (TypeError, ValueError):
        return None


def _iso(v) -> float | None:
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


async def _tavily_usage(client, key) -> dict | None:
    r = await client.get("https://api.tavily.com/usage", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    if r.status_code != 200:
        return None
    d = r.json()
    acc, k = d.get("account") or {}, d.get("key") or {}
    used, lim = _num(acc.get("plan_usage")), _num(acc.get("plan_limit"))
    if used is None or not lim:
        used, lim = _num(k.get("usage")), _num(k.get("limit"))
    return {"used": used, "limit": lim} if used is not None and lim else None


async def _firecrawl_usage(client, key) -> dict | None:
    r = await client.get("https://api.firecrawl.dev/v2/team/credit-usage", headers={"Authorization": f"Bearer {key}"},
                         timeout=10)
    if r.status_code != 200:
        return None
    d = r.json().get("data") or {}
    left, plan = _num(d.get("remainingCredits", d.get("remaining_credits"))), _num(d.get("planCredits"))
    if left is None:
        return None
    out = {"left": left // 2}  # 2 credits per search
    if plan:
        out.update(limit=plan // 2, used=max(0, plan - left) // 2)
    if end := _iso(d.get("billingPeriodEnd")):
        out["resets_at"] = end
    return out


_USAGE_CALLS = {"tavily": _tavily_usage, "firecrawl": _firecrawl_usage}


async def provider_usage() -> dict[str, dict]:
    """{service: {used, limit, left, resets_at}} as each service reports it (only the ones that can), cached."""
    out, now = {}, time.time()
    async with httpx.AsyncClient() as client:
        for p in configured():
            if p not in _USAGE_CALLS:
                continue
            if (hit := _LIVE.get(p)) and now - hit[0] < LIVE_TTL:
                got = hit[1]
            else:
                try:
                    got = await _USAGE_CALLS[p](client, os.environ[KEYS[p]])
                except (httpx.HTTPError, ValueError, AttributeError) as e:
                    log.warning("search %s usage: %s", p, type(e).__name__)
                    got = None
                _LIVE[p] = (now, got)
            if got:
                out[p] = got
    return out


def _mark_out(p: str, until: float) -> None:
    _QUOTA_OUT[p] = until
    if PERSIST:
        def save():
            try:
                from . import db
                db.save_quota_state([(f"search:{p}", p, None, 0, until, until, time.time())])
            except Exception as e:
                log.warning("search: could not save %s used up: %s", p, e)
        threading.Thread(target=save, daemon=True).start()


def load_out() -> None:
    """Services that said "used up" before a restart stay skipped until then."""
    if _out_loaded[0] or not PERSIST:
        return
    _out_loaded[0] = True
    try:
        from . import db
        rows = db.quota_states()
    except Exception as e:
        log.warning("search: could not load used-up services: %s", e)
        return
    now = time.time()
    for row in rows:
        if row["name"].startswith("search:") and (until := row["out_until"] or 0) > now:
            p = row["name"][7:]
            _QUOTA_OUT[p] = max(_QUOTA_OUT.get(p, 0), until)
            _REST[p] = max(_REST.get(p, 0), until)


def quota_report(live: dict[str, dict] | None = None) -> list[dict]:
    """Each configured search service this month, for the owner's /admin page: used of its free monthly searches,
    left, and whether it's skipped now (quota used up or key refused, or busy). live = what the services said
    (provider_usage), which wins over our own count (source 'provider')."""
    load_out()
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    _load_month(month)
    nxt = (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=32)).replace(day=1)
    limits, out, t = monthly_limits(), [], time.time()
    for p in configured():
        used = sum(n for (d, s), n in _MONTH.items() if s == p and d.startswith(month))
        lim, source, resets = limits.get(p), "ours", nxt.timestamp()
        if p in KEYLESS_SERVICES and not os.getenv(KEYS[p]) or p == "langsearch":
            lim = None  # keyless: an hourly limit per server, not a monthly quota; langsearch: a daily one
        if got := (live or {}).get(p):
            lim, source, resets = got.get("limit") or lim, "provider", got.get("resets_at") or resets
            if got.get("used") is not None:
                used = got["used"]
            elif got.get("left") is not None and lim:
                used = max(0, lim - got["left"])
        left = None if lim is None else max(0, lim - used)
        if got and got.get("left") is not None:
            left = got["left"]
        out_until = _QUOTA_OUT.get(p) if _QUOTA_OUT.get(p, 0) > t else None
        if out_until:
            left = 0
            used = lim if lim and source == "ours" else used
        out.append({"name": p, "used": used, "limit": lim, "left": left, "resets_at": resets, "source": source,
                    "quota_out_until": out_until, "busy_until": _REST.get(p) if _REST.get(p, 0) > t else None})
    return out


class SearchError(Exception):
    pass


def configured() -> list[str]:
    return [p for p in PROVIDERS if os.getenv(KEYS[p]) or KEYLESS and p in KEYLESS_SERVICES]


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


async def _keenable(client, key, query, n, include, exclude):
    """Keyless (per server address) when there's no key. No country filter: the planner's words name the market."""
    q = query
    if include:
        q += " (" + " OR ".join(f"site:{d}" for d in include) + ")"
    if not include and not _names_market(q):  # no country filter: say which market, or it answers for the world
        m = market.current()
        q += " " + (m.country_ar if re.search(r"[\u0600-\u06FF]", q) else m.country_en)
    body = {"query": q, "max_results": min(n, 50), "snippet_max_length": KEENABLE_TEXT,
            **({"exclude_domains": exclude[:1000]} if exclude else {})}
    if key:
        r = await client.post("https://api.keenable.ai/v1/search", timeout=15, headers={"X-API-Key": key}, json=body)
    else:
        r = await client.post("https://api.keenable.ai/v1/search/public", timeout=15,
                              headers={"X-Keenable-Title": APP_TITLE}, json=body)
    def parse(d):
        out = []
        for x in d.get("results", []):
            if "keenable.ai" in (x.get("url") or ""):  # its own price summary cards, not a store page
                continue
            text = x.get("snippet") or ""
            # its copy of the page: read only when we can't open the page, and only when it was crawled lately
            fresh = (t := _iso(x.get("acquired_at"))) is not None and time.time() - t < KEENABLE_FRESH
            out.append({"url": x["url"], "title": x.get("title") or "", "snippet": text,
                        "content": text if fresh else ""})
        return out
    return r, parse


def _names_market(q: str) -> bool:
    m, low = market.current(), q.lower()
    return any(w.lower() in low for w in (m.country_en, m.country_ar, f".{m.code.lower()}") if w)


async def _langsearch(client, key, query, n, include, exclude):
    r = await client.post("https://api.langsearch.com/v1/web-search", timeout=15,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"query": query, "count": min(n, 50), "contents": {"text": {"maxCharacters": CONTENT_CHARS}},
                                **({"excludeDomains": exclude} if exclude else {}),
                                **({"includeDomains": include} if include else {})})

    def parse(d):
        items = ((d.get("data") or {}).get("webPages") or {}).get("value") or []
        return [{"url": x["url"], "title": x.get("name") or "", "snippet": x.get("snippet") or x.get("text") or "",
                 "content": x.get("text") or ""} for x in items]
    return r, parse


_CALLS = {"keenable": _keenable, "langsearch": _langsearch, "tavily": _tavily, "exa": _exa, "firecrawl": _firecrawl, "brave": _brave}


def _cache_key(query: str, n: int, include, exclude, only) -> str:
    from . import db
    return CACHE_PREFIX + db.cache_key("|".join([query, str(n), ",".join(sorted(include or [])),
                                                 ",".join(sorted(exclude or [])), ",".join(sorted(only or []))]),
                                       market.current().code)


def _cached(key: str) -> list[dict] | None:
    if (hit := _CACHE.get(key)) and time.time() - hit[0] < CACHE_TTL:
        return hit[1]
    if not PERSIST or CACHE_TTL <= 0:
        return None
    try:
        from . import db
        got = db.cached_search(key, CACHE_TTL)
    except Exception as e:  # database down: search again
        log.warning("search cache: could not read: %s", e)
        return None
    if got is not None:
        _CACHE[key] = (time.time(), got.get("hits") or [])
        return _CACHE[key][1]
    return None


def _remember(key: str, hits: list[dict]) -> None:
    if CACHE_TTL <= 0:
        return
    if len(_CACHE) > 2000:
        _CACHE.clear()
    _CACHE[key] = (time.time(), hits)
    if not PERSIST:
        return
    small = [{**h, "content": (h.get("content") or "")[:CACHE_TEXT]} for h in hits]
    prune = time.time() - _last_prune[0] >= CACHE_PRUNE_EVERY
    if prune:
        _last_prune[0] = time.time()

    def save():
        try:
            from . import db
            db.save_search(key, market.current().code, {"hits": small})
            if prune:
                db.drop_old_searches(CACHE_PREFIX, time.time() - CACHE_TTL)
        except Exception as e:
            log.warning("search cache: could not save: %s", e)
    threading.Thread(target=save, daemon=True).start()


def _order(names: list[str]) -> list[str]:
    """Free services first in FREE_FIRST order (Keenable's results for Egyptian stores are far better than
    LangSearch's, live 2026-11-27: LangSearch is only its backup), then the monthly-quota ones taking turns so
    every free quota lasts. Resting ones last (asked only when nothing else is left)."""
    _turn[0] += 1
    now = time.time()

    def turns(group):
        if not group:
            return []
        k = _turn[0] % len(group)
        return group[k:] + group[:k]
    order = [p for p in FREE_FIRST if p in names] + turns([p for p in names if p not in FREE_FIRST])
    return [p for p in order if _REST.get(p, 0) <= now] or order


async def search(client: httpx.AsyncClient, query: str, max_results: int = 8,
                 exclude_domains: list[str] | None = None, include_domains: list[str] | None = None,
                 only: list[str] | None = None) -> list[dict]:
    """Returns [{"url", "title", "snippet"}]. include_domains = only these sites; exclude_domains = never these.
    only = just these services (e.g. the ones that return page text)."""
    names = [n for n in configured() if not only or n in only]
    if not names:
        raise SearchError("مفيش مفتاح بحث (TAVILY_API_KEY / EXA_API_KEY / FIRECRAWL_API_KEY) في .env")
    key = _cache_key(query, max_results, include_domains, exclude_domains, only)
    if (hit := await asyncio.to_thread(_cached, key)) is not None:
        return hit
    errors, empty = [], None
    for p in _order(names):
        try:
            r, parse = await _CALLS[p](client, os.getenv(KEYS[p], ""), query, max_results, include_domains,
                                       exclude_domains)
        except httpx.HTTPError as e:
            errors.append(f"{p}: {type(e).__name__}")
            _REST[p] = time.time() + BUSY_REST
            continue
        if r.status_code == 200:
            _count(p)
            try:
                hits = [{**x, "snippet": (x.get("snippet") or "")[:300], "content": (x.get("content") or "")[:CONTENT_CHARS]}
                        for x in parse(r.json()) if x.get("url")]
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                errors.append(f"{p}: bad reply {type(e).__name__}")
                continue
            if not hits and p in FREE_FIRST:  # a free service found nothing: the next one may
                empty = hits
                continue
            _remember(key, hits)
            return hits
        errors.append(f"{p} {r.status_code}")
        log.warning("search %s said %s: %s", p, r.status_code, r.text[:150].replace("\n", " "))
        # 401/402/403/432/433 = bad key or the free quota is used up; 429/5xx = busy for now
        quota = r.status_code in (401, 402, 403, 432, 433)
        wait = BUSY_REST
        if r.status_code == 429:
            try:
                wait = max(BUSY_REST, min(float(r.headers.get("retry-after") or 0), QUOTA_REST))
            except ValueError:
                pass
        _REST[p] = time.time() + (QUOTA_REST if quota else wait)
        if quota:
            _mark_out(p, _REST[p])
            problems.note("search_error", f"{p} {r.status_code}: free quota used up or key refused")
    if empty is not None:  # nobody failed loudly: the free services just had nothing
        return empty
    problems.note("search_error", "every search service failed: " + ", ".join(errors), request=query)
    raise SearchError("البحث فشل: " + ", ".join(errors))


async def probe(paid: bool = False, query: str = "") -> dict:
    """Checks from this server that each search service answers (the free ones; the monthly-quota ones too with
    paid=True): status, results, seconds, the stores found, and whether a key was used. Then the cache: the same
    search twice, the second from the cache. For the owner's check workflow; one search per service."""
    query = query or "Samsung Galaxy A55 " + market.current().country_en
    out: dict = {"services": [], "configured": configured()}
    async with httpx.AsyncClient() as client:
        for p in configured():
            if p not in FREE_FIRST and not paid:
                continue
            row = {"name": p, "key": bool(os.getenv(KEYS[p]))}
            t = time.monotonic()
            try:
                r, parse = await _CALLS[p](client, os.getenv(KEYS[p], ""), query, 8, None, None)
                row["status"] = r.status_code
                if r.status_code == 200:
                    _count(p)
                    hits = [h for h in parse(r.json()) if h.get("url")]
                    row["results"] = len(hits)
                    row["sites"] = sorted({urlparse(h["url"]).netloc.removeprefix("www.") for h in hits})[:10]
                else:
                    row["error"] = r.text[:200]
            except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as e:
                row["error"] = type(e).__name__
            row["seconds"] = round(time.monotonic() - t, 1)
            out["services"].append(row)
        words = f"{query} {int(time.time())}"
        t = time.monotonic()
        try:
            first = await search(client, words)
            mid = time.monotonic()
            again = await search(client, words)
            out["cache"] = {"first_seconds": round(mid - t, 1), "again_seconds": round(time.monotonic() - mid, 2),
                            "same": again == first, "results": len(first)}
        except SearchError as e:
            out["cache"] = {"error": str(e)[:200]}
    return out
