import asyncio
import calendar
import hashlib
import hmac
import io
import ipaddress
import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as UrlPart
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path):
    """Read KEY=VALUE lines from .env (so the API key isn't typed every time)."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file(ROOT / ".env")

from . import accounts, admin, affiliate, agent, catalog, coupons, db, deals, fetchers, market, problems, trending  # noqa: E402  (after .env is loaded)
from .stores import ALL_STORES  # noqa: E402
from .stores.base import HEADERS  # noqa: E402

log = logging.getLogger("uvicorn.error")

DEMO = os.getenv("DEMO") == "1"
if DEMO:  # offline demo: stores read saved sample pages instead of the internet
    for _s in ALL_STORES:
        _page = (ROOT / "tests" / "fixtures" / f"{_s.name.lower().replace('.', '')}.html").read_text(encoding="utf-8")

        async def _fake(client, text, _s=_s, _page=_page):
            return _s.parse(_page)
        _s.search = _fake

@asynccontextmanager
async def _lifespan(app):
    try:  # a database that's down must not stop the site from starting
        if n := await asyncio.to_thread(accounts.drop_loose_price_alerts):
            log.info("removed %d old price-drop alerts", n)
    except Exception:
        log.exception("removing old price-drop alerts failed")
    yield


app = FastAPI(title="Easy Found", lifespan=_lifespan)
# Same request again within this time = answered from the price database (instant, saves the free quotas).
CACHE_TTL = float(os.getenv("CACHE_HOURS", "6")) * 3600
PLAN_TTL = 24 * 3600  # how long the plan of a first message is reused for the same words
# New searches per visitor per day (each costs ~7 web searches and many LLM calls from the shared free quotas).
# 0 = no limit (e.g. on your own PC). Searches answered from the cache don't count.
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "0"))
# Signed-in shoppers get their own (usually higher) limit, counted per account instead of per IP. 0 = DAILY_LIMIT.
USER_DAILY_LIMIT = int(os.getenv("USER_DAILY_LIMIT", "0"))
LIMIT_TEXT = ("وصلت لحد البحث النهارده ({n} بحث). الموقع مجاني وبيشتغل على حصص محدودة، "
              "فالحد بيرجع بكرة. تقدر لسه تسأل عن النتايج اللي ظهرتلك.")


def _public_ip(text: str) -> bool:
    try:
        return ipaddress.ip_address(text).is_global
    except ValueError:
        return False


def client_id(request: Request) -> str:
    """Who is asking, for the daily limit and reports: a hash of the IP (the IP itself is never stored).
    Behind hosting proxies the real IP is the last public one appended to X-Forwarded-For: entries further left
    can be faked by the visitor, and the hosts' own internal hops are private addresses."""
    forwarded = [x.strip() for x in request.headers.get("x-forwarded-for", "").split(",") if x.strip()]
    public = [x for x in forwarded if _public_ip(x)]
    ip = public[-1] if public else (request.client.host if request.client else "?")
    return hashlib.sha256((os.getenv("CLIENT_SALT", "dealhunter") + ip).encode()).hexdigest()[:16]


def _user_client(user: dict) -> str:
    return "u" + hashlib.sha256((os.getenv("CLIENT_SALT", "dealhunter") + user["id"]).encode()).hexdigest()[:15]


def current_user(request: Request) -> dict | None:
    """The signed-in shopper (from the signed cookie), or None. Blocking (the database): call it in a thread from
    async code."""
    if not accounts.enabled():
        return None
    try:
        return accounts.get_user(accounts.read_session(request.cookies.get(accounts.COOKIE)))
    except accounts.Unavailable:  # searching still works, as a visitor, while the accounts database is down
        return None


@app.get("/")
def index():
    # no-cache: the browser always picks up the latest page after an update
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/logo.svg")
@app.get("/logo-512.png")
@app.get("/logo-120.png")
@app.get("/favicon.ico")
def logo(request: Request):
    name = {"/favicon.ico": "logo-120.png"}.get(request.url.path, request.url.path.lstrip("/"))
    return FileResponse(ROOT / "static" / name, headers={"Cache-Control": "public, max-age=86400"})


def _site(request: Request) -> str:
    proto = "https" if _secure(request) else request.url.scheme
    return f"{proto}://{request.url.netloc}"


@app.get("/robots.txt")
def robots(request: Request):
    """Search engines may list the pages, not the API (search results are made per visitor)."""
    return Response(f"User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /img\nDisallow: /admin\nSitemap: {_site(request)}/sitemap.xml\n",
                    media_type="text/plain")


@app.get("/sitemap.xml")
def sitemap(request: Request):
    urls = "".join(f"<url><loc>{_site(request)}{p}</loc></url>" for p in ("/", "/deals", "/coupons", "/privacy"))
    return Response('<?xml version="1.0" encoding="UTF-8"?>'
                    f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>',
                    media_type="application/xml")


@app.get("/{name}.html")
def site_verification(name: str):
    """Google Search Console's "HTML file" check: set GOOGLE_SITE_VERIFICATION to the file name it gives
    (e.g. google1a2b3c.html) on the server, no code change or upload needed."""
    want = os.getenv("GOOGLE_SITE_VERIFICATION", "").strip().removesuffix(".html")
    if not want or name != want or not name.startswith("google"):
        raise HTTPException(404)
    return Response(f"google-site-verification: {name}.html", media_type="text/html")


@app.get("/privacy")
def privacy():
    """Privacy policy: Google requires its link before sign-in can be opened to everyone."""
    return FileResponse(ROOT / "static" / "privacy.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/search")
async def search(request: Request, q: str = Query(..., min_length=2, max_length=300)):
    """One-shot search (no chat). Same daily limit as the chat, or it would be a free way around it."""
    log.info("=" * 70)
    log.info("NEW SEARCH: %s", q)
    client = client_id(request)
    if DAILY_LIMIT and not await _db(db.cached_search, db.cache_key(q, market.current().code), CACHE_TTL):
        if await _db(db.searches_today, client, default=0) >= DAILY_LIMIT:
            raise HTTPException(429, LIMIT_TEXT.format(n=DAILY_LIMIT))
        await _db(db.count_search, client)
    d = await _search(q)
    await _track_turn(request, q, {**d, "action": "search"})
    return d


async def _db(fn, *args, default=None, **kwargs):
    """A database call off the event loop (a slow or reconnecting database must not freeze every other visitor).
    Bookkeeping errors are logged, not shown: a search still works without the cache or price history."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except Exception:
        log.exception("database call %s failed", fn.__name__)
        return default


async def _search(q: str, plan: dict | None = None, budget_text: str | None = None) -> dict:
    mk = market.current().code
    key = db.cache_key(q, mk)
    if not DEMO and (hit := await _db(db.cached_search, key, CACHE_TTL)):
        log.info("served from cache (prices checked again on the store pages)")
        log.info("=" * 70 + "\n")
        return {**await agent.recheck(hit), "cached": True}
    t = time.monotonic()
    payload = await agent.run(q, plan=plan, budget_text=budget_text)
    log.info("DONE in %.1fs: %d results (%s)", time.monotonic() - t, len(payload["results"]), payload.get("source"))
    log.info("=" * 70 + "\n")
    payload.update(query=q, count=len(payload["results"]), cached=False, demo=DEMO, market=mk)
    if not DEMO and payload["results"]:
        await _db(db.add_price_context, payload["results"])  # before saving today's prices
        await _db(db.record_prices, payload["results"], mk)
        await _db(catalog.remember_found, payload["results"])  # the next shopper gets them from the index first
        # no price-drop alerts from a search: only the scheduled check reads a followed page for its own product
        if payload["mode"] == "llm":
            await _db(db.save_search, key, mk, payload)
    return payload


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(max_length=2000)


class ShownResult(BaseModel):
    """One result on the shopper's screen, sent back with the next message so questions about it get real answers."""
    title: str = Field("", max_length=300)
    price: float | None = None
    old_price: float | None = None
    discount_pct: int | None = None
    store: str = Field("", max_length=100)
    trusted: bool = False
    trust_score: int | None = None
    verified: bool | None = None
    condition: str | None = Field(None, max_length=30)
    unconfirmed: bool = False
    lowest_seen: float | None = None
    previous_price: float | None = None
    note: str | None = Field(None, max_length=300)


class ChatTurn(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    request: str = Field("", max_length=600)  # the standalone request of the previous search, "" at the start
    shown: list[ShownResult] = Field(default_factory=list, max_length=20)  # the last search's results, cheapest first
    price_min: float | None = Field(None, ge=0, le=10_000_000)  # the price range set in the search box, if any
    price_max: float | None = Field(None, gt=0, le=10_000_000)


@app.post("/api/chat")
async def chat(turn: ChatTurn, request: Request):
    """A chat turn: the newest message is read together with the earlier ones, so a follow-up like
    'سامسونج بس' narrows the previous request instead of starting a new search."""
    user = await asyncio.to_thread(current_user, request)
    d = await _chat_turn(turn, client_id(request), user)
    await _track_turn(request, turn.messages[-1].text, d)
    return d


KEEPALIVE_SECONDS = 8
SEARCH_CAP_SECONDS = 25  # the shopper gets checked results by then; the other stores' search goes on and updates them


def _interim(partial: dict) -> dict:
    """The answer sent when the search passes SEARCH_CAP_SECONDS: the offers already checked on their own pages,
    marked pending (the full answer follows as an "update" line when the rest of the search ends)."""
    return {**partial, "action": "search", "pending": True, "query": partial.get("request") or "", "cached": False,
            "demo": DEMO, "mode": "llm", "suggestions": [], "candidates_checked": partial.get("count") or 0}


@app.post("/api/chat/stream")
async def chat_stream(turn: ChatTurn, request: Request):
    """Same as /api/chat, but sends progress lines while it works (NDJSON: one JSON object per line):
    {"type": "status", "text": ...} and {"type": "reply", "text": <the reply so far, while the LLM writes it>},
    {"type": "partial", "data": {"results": [...], ...}} (the first checked offers while the search goes on) ...
    then {"type": "done", "data": <the /api/chat answer>}.
    A search still going after SEARCH_CAP_SECONDS with checked offers already found answers with those right away:
    {"type": "done", "data": {..., "pending": true}}, then {"type": "update", "data": <the full answer>} at the end."""
    queue: asyncio.Queue = asyncio.Queue()
    client = client_id(request)
    user = await asyncio.to_thread(current_user, request)

    async def work():
        token = agent.PROGRESS.set(lambda text: queue.put_nowait({"type": "status", "text": text}))
        said = agent.REPLY.set(lambda text: queue.put_nowait({"type": "reply", "text": text}))  # typed as it comes
        early = agent.PARTIAL.set(lambda data: queue.put_nowait({"type": "partial", "data": data}))  # first results
        try:
            d = await _chat_turn(turn, client, user)
            queue.put_nowait({"type": "done", "data": d})
            # on its own: the stream ends right after "done" and cancels this task, which would lose the count
            _later(_track_turn(request, turn.messages[-1].text, d))
        except HTTPException as e:
            queue.put_nowait({"type": "error", "status": e.status_code})
        except Exception as e:  # the page shows a friendly error; the details are in the server log and /admin
            log.exception("chat stream failed")
            problems.note("server_error", f"chat: {type(e).__name__}: {e}"[:300], request=turn.messages[-1].text)
            queue.put_nowait({"type": "error", "status": 500})
        finally:
            agent.PARTIAL.reset(early)
            agent.REPLY.reset(said)
            agent.PROGRESS.reset(token)

    async def lines():
        task = asyncio.create_task(work())
        t0, early, answered = time.monotonic(), None, False

        def line(event: dict) -> str:
            return json.dumps(event, ensure_ascii=False) + "\n"
        try:
            while True:
                left = SEARCH_CAP_SECONDS - (time.monotonic() - t0)
                wait = min(KEEPALIVE_SECONDS, left) if not answered and left > 0 else KEEPALIVE_SECONDS
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=wait)
                except asyncio.TimeoutError:
                    if not answered and early and time.monotonic() - t0 >= SEARCH_CAP_SECONDS:
                        answered = True
                        yield line({"type": "done", "data": _interim(early)})
                    else:
                        # nothing new for a while (a slow model): a small line keeps proxies and phones from
                        # closing a connection that looks idle, which the page showed as an error
                        yield '{"type": "ping"}\n'
                    continue
                if event["type"] == "done" and answered:
                    event = {**event, "type": "update"}
                yield line(event)
                if event["type"] in ("done", "update", "error"):
                    break
                if event["type"] == "partial":
                    early = event["data"]
                    if not answered and time.monotonic() - t0 >= SEARCH_CAP_SECONDS:
                        answered = True
                        yield line({"type": "done", "data": _interim(early)})
        finally:
            if not task.done():  # the shopper closed the page: stop working for nobody
                task.cancel()

    return StreamingResponse(lines(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


async def _chat_turn(turn: ChatTurn, client: str | None = None, user: dict | None = None) -> dict:
    """client = who is asking (None = no daily limit, e.g. scripts); user = the signed-in shopper, if any:
    searches in their region (when the agent covers it), uses their limit and preferences."""
    if user:
        client = client and _user_client(user)
    token = market.use(user and user["market"])
    try:
        with agent.llm.live():  # background LLM work (cron jobs) waits while a shopper waits
            d = await _chat_turn_in_market(turn, client, USER_DAILY_LIMIT if user and USER_DAILY_LIMIT else DAILY_LIMIT)
    finally:
        market.reset(token)
    used = agent.llm.usage()  # this turn's own calls (a saved answer served again carries its first run's count)
    d["llm_usage"] = {"calls": used["calls"], "tokens": used["tokens"],
                      "by_step": {k: dict(v) for k, v in used["by_step"].items()}}
    if user:
        await asyncio.to_thread(_personalize, d, user)
    return d


async def _track_turn(request: Request, message: str, d: dict) -> None:
    """Counts a chat message for the owner's /admin page: what was asked, what the agent did, how many results."""
    action = "limit" if d.get("limit") else d.get("action") or "search"
    results = d.get("results") or []
    await _track(request, "turn", detail=message, request=d.get("query") or d.get("request") or "", action=action,
                 results=len(results) if action == "search" else None, cached=d.get("cached"))
    await _track_llm(request, d)


async def _track_llm(http: Request, d: dict) -> None:
    """The turn's LLM calls per step, for the owner's /admin page (how the free daily quotas are spent). The owner's
    own searches count too: they use the same quotas."""
    used = d.get("llm_usage") or {}
    if DEMO or not used.get("calls"):
        return
    client, req = client_id(http), d.get("query") or d.get("request") or ""
    action = "limit" if d.get("limit") else d.get("action") or "search"
    await _db(db.add_event, "llm", client, action="turn", results=used["calls"], price=used.get("tokens") or 0,
              detail=action, request=req)
    for step, s in (used.get("by_step") or {}).items():
        await _db(db.add_event, "llm", client, action=step[:20], results=s.get("calls") or 0,
                  price=s.get("tokens") or 0, detail=s.get("model") or "", request=req)


_background: set[asyncio.Task] = set()


def _later(coro) -> None:
    """Runs bookkeeping after the answer is sent, kept referenced until it ends (a lone task can be dropped)."""
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _track(http: Request, kind: str, **fields) -> None:
    """One usage event (db.events). The owner's own visits (signed in to /admin) aren't counted."""
    if DEMO or admin.is_admin(http.cookies.get(admin.COOKIE)):
        return
    member = bool(accounts.read_session(http.cookies.get(accounts.COOKIE)))
    await _db(db.add_event, kind, client_id(http), member, **fields)


def _personalize(d: dict, user: dict) -> None:
    """Applies the shopper's saved preferences to an answer and marks the products they already follow."""
    if user["prefs"].get("hide_unsure_shops") and d.get("results"):
        d["results"] = [r for r in d["results"] if r.get("trusted") or (r.get("trust_score") or 0) >= 60]
        d["count"] = len(d["results"])
    followed = {w["url"] for w in accounts.list_watches(user["id"])}
    for r in d.get("results") or []:
        r["watched"] = r.get("url") in followed
    mk = user["market"]
    if mk and mk not in market.live() and d.get("action") == "search":  # searched in MARKET instead
        d["region_note"] = (f"البحث في {market.MARKETS[mk].country_ar} لسه بنجهزه، فالنتايج دي من متاجر "
                            f"{market.current().country_ar}. هنبعتلك إشعار أول ما يشتغل.")


async def _chat_turn_in_market(turn: ChatTurn, client: str | None, limit: int) -> dict:
    messages = [m.model_dump() for m in turn.messages]
    lo, hi = turn.price_min or None, turn.price_max or None
    if lo and hi and lo > hi:
        lo, hi = hi, lo
    rng = agent.price_range_text(lo, hi)
    said = " | ".join([m["text"] for m in messages if m["role"] == "user"] + ([rng] if rng else []))
    newest = next((m["text"] for m in reversed(messages) if m["role"] == "user"), "").strip()
    if len(newest) < 2:
        raise HTTPException(422, "empty message")
    log.info("=" * 70)
    log.info("CHAT: %s (current request: %s)", newest, turn.request or "-")
    plan, failed = None, None
    agent.llm.reset_usage()
    # a first message with nothing else around it (no chat, no results, no price range) is planned once a day: the
    # next shopper writing the same words gets the same plan with no LLM call and no web search
    users = [m for m in messages if m["role"] == "user"]
    plan_key = ("__plan__:" + db.cache_key(newest, market.current().code)
                if len(users) == len(messages) == 1 and not turn.shown and not turn.request and not rng else "")
    if plan_key and not DEMO and (plan := await _db(db.cached_search, plan_key, PLAN_TTL)):
        log.info("chat: the same first message was planned today: reusing its plan")
        if plan.get("reply") and (live := agent.REPLY.get()):
            live(plan["reply"])
    elif agent.llm.available():
        hits, words = [], ""
        if not turn.shown:  # a new request: the planner checks the product against the web in the same call
            words = agent.plain_request([next(m["text"] for m in messages if m["role"] == "user")])
            hits = await agent.shopper_hits(words)
        try:
            plan = await agent.plan_turn(messages, turn.request, [r.model_dump() for r in turn.shown], (lo, hi),
                                         hits=hits, hits_for=words)
            if plan_key and not DEMO:
                await _db(db.save_search, plan_key, market.current().code, plan)
        except agent.llm.LLMError as e:
            log.warning("chat planner failed: %s", e.details or e)
            failed = e
        except Exception as e:  # a provider's odd reply must not end the turn with an error: search without it
            log.exception("chat planner crashed")
            problems.note("server_error", f"planner: {type(e).__name__}: {e}"[:300], request=newest)
            failed = e
    if plan and plan["action"] == "ask":  # like a salesperson: one question first, to recommend the right thing
        log.info("=" * 70 + "\n")
        return {"action": "ask", "question": plan["question"], "options": plan.get("options") or [],
                "request": turn.request, "results": [], "demo": DEMO}
    if plan and plan["action"] == "recommend":
        # products that fit the need: found in the stores right away, so the shopper sees real offers (checked price,
        # picture, store), not just names
        names = [r["name"] for r in plan.get("recommendations") or []]
        plan["need_based"] = True
        plan["suggested_products"] = plan.get("suggested_products") or names
        plan["understood"] = plan.get("reply") or plan.get("understood")
    shown = [r.model_dump() for r in turn.shown]
    if plan and plan["action"] == "answer" and agent.unchecked_prices(
            " ".join([plan["answer"]] + [" ".join(map(str, r)) for r in (plan.get("table") or {}).get("rows", [])]), shown):
        log.info("chat: the answer names prices no result shows: searching instead")
        plan["action"] = "search"
    if plan and plan["action"] == "answer":  # a question about the shown results: answer it, no new search
        log.info("=" * 70 + "\n")
        return {"action": "answer", "answer": plan["answer"], "table": plan.get("table"),
                "suggestions": plan["suggestions"], "request": turn.request or plan["request"],
                "results": [], "demo": DEMO}
    if failed and turn.request and newest.rstrip().endswith(("?", "؟")):
        # a question about the results needs the LLM; gluing it onto the old request would search for nonsense
        log.info("=" * 70 + "\n")
        return {"action": "answer", "request": turn.request, "results": [], "demo": DEMO,
                "answer": "معلش، الموديلات المجانية مشغولة دلوقتي فمقدرتش أرد على سؤالك. ابعته تاني كمان دقيقة."}
    # no LLM: the earlier request plus the new message; at the start, all their messages (the product is often in
    # the first one and the newest is a tapped option like 'دورلي على طول')
    request = plan["request"] if plan else " ".join(filter(None, [turn.request, newest])) if turn.request else \
        agent.plain_request([m["text"] for m in messages if m["role"] == "user"])
    if rng:  # the range set in the box is the budget: always applied, and part of the request (and its cache key)
        if plan:
            plan["max_price"], plan["min_price"] = hi, lo
        if not agent._states_budget(request):
            request = f"{request} ({rng})"
    if (client and limit and await _db(db.searches_today, client, default=0) >= limit
            and not await _db(db.cached_search, db.cache_key(request, market.current().code), CACHE_TTL)):
        log.info("daily limit reached for %s", client)
        log.info("=" * 70 + "\n")
        return {"action": "answer", "limit": True, "answer": LIMIT_TEXT.format(n=limit),
                "request": turn.request, "results": [], "demo": DEMO}
    if plan and plan.get("understood"):
        agent._progress(plan["understood"])
    if failed:  # don't wait for the LLM a second time: simple rules for this turn
        payload = await agent.run_rules(request, note=agent.BUSY_NOTE)
        payload.update(query=request, count=len(payload["results"]), cached=False, demo=DEMO)
    else:
        cached = await _db(db.cached_search, db.cache_key(request, market.current().code), CACHE_TTL)
        if client and not cached:  # counted before searching, so many searches sent at once can't pass the limit
            await _db(db.count_search, client)
        payload = await _search(request, plan=plan, budget_text=said)
    if plan and not payload.get("suggestions"):  # next steps the shopper can tap (the judge's, from the results, first)
        payload["suggestions"] = plan.get("suggestions") or []
    return {**payload, "action": "search"}


class Report(BaseModel):
    reason: str = Field(pattern="^(wrong_product|wrong_price|bad_store|sold_out|other)$")
    request: str = Field("", max_length=600)
    url: str = Field(max_length=1000)
    store: str = Field("", max_length=100)
    title: str = Field("", max_length=300)
    price: float | None = None


# Reports and feedback messages per visitor per day: enough for any real shopper, and stops one script from filling
# the database. Kept in memory (a restart forgets it, which is fine for this).
SPAM_LIMITS = {"report": 60, "feedback": 20, "event": 600, "admin": 600}
_sent: dict[tuple[str, str, str], int] = {}


def _too_many(kind: str, client: str) -> bool:
    key = (time.strftime("%Y-%m-%d", time.gmtime()), kind, client)
    if len(_sent) > 50_000:
        _sent.clear()
    _sent[key] = _sent.get(key, 0) + 1
    return _sent[key] > SPAM_LIMITS[kind]


@app.post("/api/report")
def report(r: Report, request: Request):
    """The "this result is wrong" button: saved for review (python -m scripts.reports)."""
    if _too_many("report", client_id(request)):
        raise HTTPException(429, "too many reports today")
    db.add_report(r.reason, r.request, r.url, r.store, r.title, r.price, client_id(request))
    log.info("REPORT %s: %s @ %s (%s)", r.reason, r.title[:60], r.store, r.request[:60])
    return {"ok": True}


class Feedback(BaseModel):
    kind: str = Field(pattern="^(complaint|suggestion|result|question)$")
    message: str = Field(min_length=3, max_length=2000)
    contact: str = Field("", max_length=200)
    lang: str = Field("ar", pattern="^(ar|en)$")


@app.post("/api/feedback")
def feedback(f: Feedback, request: Request):
    """The complaints & suggestions box: saved for review (python -m scripts.feedback)."""
    if _too_many("feedback", client_id(request)):
        raise HTTPException(429, "too many messages today")
    db.add_feedback(f.kind, f.message.strip(), f.contact.strip(), f.lang, client_id(request))
    log.info("FEEDBACK %s: %s", f.kind, f.message[:80])
    return {"ok": True}


class Event(BaseModel):
    """Sent by the page: a visit (once per page load) or a click on a store link. Generous sizes: long text is cut
    when saved (db.add_event), a count must never be lost because a store's address or title was long."""
    kind: str = Field(pattern="^(visit|click|save|share)$")  # save / share: a result kept or shared
    source: str = Field("", max_length=4000)     # visit: the site the visitor came from (its host name only)
    mobile: bool = False                         # visit: a phone or a tablet
    url: str = Field("", max_length=8000)        # click: the product page opened
    store: str = Field("", max_length=1000)
    title: str = Field("", max_length=4000)
    price: float | None = None
    request: str = Field("", max_length=4000)    # click: the search it came from
    rank: int | None = Field(None, ge=0)         # click: its place in the list as shown (0 = first)


@app.post("/api/event")
async def event(e: Event, request: Request):
    """Usage counting for the owner's /admin page. Only what's listed in Event is kept, with a hash of the IP."""
    if _too_many("event", client_id(request)):
        raise HTTPException(429, "too many events today")
    if e.kind == "visit":
        source = urlparse(e.source if "//" in e.source else "//" + e.source).hostname or ""
        await _track(request, "visit", detail=source.removeprefix("www."), action="mobile" if e.mobile else "desktop")
    else:
        if not e.url.startswith(("https://", "http://")):
            raise HTTPException(422, "not a web address")
        await _track(request, e.kind, detail=e.url, store=e.store, title=e.title, price=e.price, request=e.request,
                     results=min(e.rank, 100) if e.rank is not None else None)
    return {"ok": True}


# ---------- the owner's /admin page (app/admin.py): off unless ADMIN_PASSWORD is set ----------
NO_INDEX = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow", "Referrer-Policy": "no-referrer"}


@app.get("/admin")
def admin_page():
    if not admin.enabled():
        raise HTTPException(404)
    return FileResponse(ROOT / "static" / "admin.html", headers=NO_INDEX)


class AdminLogin(BaseModel):
    password: str = Field(min_length=1, max_length=200)


@app.post("/api/admin/login")
def admin_login(body: AdminLogin, request: Request, response: Response):
    if not admin.enabled():
        raise HTTPException(404)
    client = client_id(request)
    if admin.locked(client):
        raise HTTPException(429, "too many wrong passwords, try again in an hour")
    if not admin.check(body.password, client):
        log.warning("admin: wrong password")
        raise HTTPException(401, "wrong password")
    response.set_cookie(admin.COOKIE, admin.make_cookie(), max_age=admin.ADMIN_DAYS * 86400, httponly=True,
                        samesite="strict", secure=_secure(request))
    return {"ok": True}


@app.post("/api/admin/logout")
def admin_logout(response: Response):
    response.delete_cookie(admin.COOKIE)
    return {"ok": True}


@app.get("/api/admin/stats")
async def admin_stats(request: Request, days: int = Query(7, ge=1, le=db.EVENTS_DAYS),
                      tz: int = Query(0, ge=-840, le=840)):
    """Visitors, chat messages, searches and store clicks of the last `days` days. tz = the viewer's offset from UTC
    in minutes (the page sends it), so the days start at the owner's midnight."""
    if not admin.is_admin(request.cookies.get(admin.COOKIE)):
        raise HTTPException(401, "sign in")
    try:
        stats = await asyncio.to_thread(db.usage_stats, days, tz)
    except Exception:
        log.exception("admin stats failed")
        raise HTTPException(503, "database unavailable")
    for row in stats.get("stores", []):  # which stores can pay us for these clicks (AFFILIATE_TAGS)
        row["affiliate"] = affiliate.store_has_code(row.get("store") or "")
    return JSONResponse(stats, headers=NO_INDEX)


class CouponIn(BaseModel):
    store: str = Field(min_length=3, max_length=120)          # the store's domain or any page of it
    code: str = Field("", max_length=60)                      # '' = the link gives the discount by itself
    title: str = Field(min_length=3, max_length=200)
    link: str = Field(min_length=10, max_length=1000)
    where: str = Field(min_length=2, max_length=80)           # where the code came from: "ArabClicks", "صفحة نون"
    expires: str = Field("", max_length=10)                   # YYYY-MM-DD, last day it works (Cairo)


def _need_admin(request: Request) -> None:
    if not admin.is_admin(request.cookies.get(admin.COOKIE)):
        raise HTTPException(401, "sign in")


@app.get("/api/admin/coupons")
def admin_coupons(request: Request):
    _need_admin(request)
    return JSONResponse({"coupons": db.all_coupons(market.current().code), "now": time.time()}, headers=NO_INDEX)


@app.post("/api/admin/coupons")
def admin_add_coupon(body: CouponIn, request: Request):
    """The owner adds a code from a legitimate source (a network's offer, the store's own page)."""
    _need_admin(request)
    store = coupons.domain_of(body.store)
    if "." not in store:
        raise HTTPException(400, "store: write its address, like noon.com")
    if not body.link.startswith("https://") or not urlparse(body.link).hostname:
        raise HTTPException(400, "link must start with https://")
    ends = None
    if body.expires:
        try:
            day = time.strptime(body.expires, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "expires: YYYY-MM-DD")
        ends = calendar.timegm(day) + 86400 - 2 * 3600  # end of that day in Cairo (UTC+2)
        if ends <= time.time():
            raise HTTPException(400, "that date has passed")
    db.add_coupon(market.current().code, store, body.code.strip(), " ".join(body.title.split()), body.link.strip(),
                  "owner:" + " ".join(body.where.split()), ends)
    return {"ok": True}


@app.delete("/api/admin/coupons/{coupon_id}")
def admin_delete_coupon(coupon_id: int, request: Request):
    _need_admin(request)
    if not db.delete_coupon(coupon_id):
        raise HTTPException(404)
    return {"ok": True}


# Store images are loaded through us, shrunk to a thumbnail: some stores block their images on other sites, and the
# originals can be 3000 px photos that make phones slow or even reload the page (each one takes ~30 MB to show).
# Only public addresses are fetched (never this server's own network), and thumbnails are kept in memory for a while.
THUMB_PX = 320
MAX_IMAGE_BYTES = 8_000_000
_img_client = httpx.AsyncClient(timeout=10, follow_redirects=False)
_thumbs: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()


async def _resolve(host: str) -> list:
    return await asyncio.get_running_loop().getaddrinfo(host, 443)


async def _public_host(host: str) -> bool:
    try:
        infos = await _resolve(host)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def _thumbnail(data: bytes) -> tuple[bytes, str]:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 50_000_000  # bigger than this is a decompression bomb, not a product photo
    with Image.open(io.BytesIO(data)) as im:
        im.draft("RGB", (THUMB_PX * 2, THUMB_PX * 2))  # JPEGs decode straight at a smaller size: much faster
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        im.thumbnail((THUMB_PX, THUMB_PX))
        out = io.BytesIO()
        im.save(out, "WEBP", quality=78, method=4)
    return out.getvalue(), "image/webp"


@app.get("/img")
async def image(u: str = Query(..., max_length=1000)):
    if u in _thumbs:
        _thumbs.move_to_end(u)
        body, ctype = _thumbs[u]
        return Response(body, media_type=ctype, headers={"Cache-Control": "public, max-age=604800"})
    url = u
    for _ in range(3):  # follow at most 2 redirects, each checked like the first address
        host = urlparse(url).hostname or ""
        if urlparse(url).scheme != "https" or not await _public_host(host):
            raise HTTPException(400, "image address not allowed")
        headers = {**HEADERS, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
                   "Referer": f"https://{host.split('.', 1)[-1]}/"}
        try:
            r = await _img_client.get(url, headers=headers)
        except httpx.HTTPError as e:
            log.warning("img %s failed: %s", url, type(e).__name__)
            raise HTTPException(502, "image fetch failed")
        if r.is_redirect and r.headers.get("location"):
            url = str(r.url.join(r.headers["location"]))
            continue
        break
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or not ctype.startswith("image/") or len(r.content) > MAX_IMAGE_BYTES:
        log.warning("img %s -> %s %s", url, r.status_code, ctype)
        raise HTTPException(404, "image not found")
    try:
        body, ctype = await asyncio.to_thread(_thumbnail, r.content)
    except Exception as e:  # SVG, a broken file, a format Pillow can't read: send it as it is
        log.info("img %s not shrunk: %s", url, type(e).__name__)
        body = r.content
    _thumbs[u] = (body, ctype)
    while len(_thumbs) > 400:
        _thumbs.popitem(last=False)
    return Response(body, media_type=ctype, headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/history")
def price_history(url: str = Query(..., max_length=1000)):
    """A product's saved prices, oldest first (for the price chart / 'is this discount real?')."""
    points = db.history(url)
    return {"url": url, "points": points, "lowest": min((p["price"] for p in points), default=None)}


# ---------- accounts: Google sign-in, preferences, followed products, notifications ----------

@app.exception_handler(accounts.Unavailable)
def _accounts_down(request: Request, exc: accounts.Unavailable):
    log.error("accounts request while the database is unreachable: %s", request.url.path)
    return JSONResponse({"detail": "الحسابات مش متاحة دلوقتي، جرب كمان شوية."}, status_code=503)


def _need_user(request: Request) -> dict:
    user = accounts.enabled() and accounts.get_user(accounts.read_session(request.cookies.get(accounts.COOKIE)))
    if not user:
        raise HTTPException(401, "sign in first")
    return user


@app.get("/api/suggestions")
async def suggestions():
    """Request ideas for the empty page, made from what's trending now ({} = the page keeps its own examples)."""
    if DEMO:
        return {}
    return await trending.suggestions()


@app.get("/api/config")
def config():
    """What the page needs: the region menu (the market searched now, and which ones work today), whether Google
    sign-in and email alerts are on, and the regions a shopper can pick at sign-up."""
    cur, live = market.current(), market.live()
    return {"market": cur.code, "demo": DEMO,
            "markets": [{"code": m.code, "name_ar": m.country_ar, "name_en": m.country_en, "currency_ar": m.currency_ar,
                         "currency": m.currency, "enabled": m.code in live} for m in market.MARKETS.values()],
            "google_client_id": accounts.google_client_id(), "email_alerts": accounts.email_enabled(),
            "affiliate": affiliate.rules(),
            "regions": [{"code": c, "name": market.MARKETS[c].country_ar, "currency": market.MARKETS[c].currency_ar,
                         "live": c in live} for c in accounts.REGIONS]}


class GoogleLogin(BaseModel):
    credential: str = Field(min_length=20, max_length=4096)


def _secure(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


@app.post("/api/auth/google")
async def google_login(body: GoogleLogin, request: Request, response: Response):
    if not accounts.enabled():
        raise HTTPException(404, "sign-in is off (no GOOGLE_CLIENT_ID)")
    try:
        info = await accounts.verify_google(body.credential)
    except accounts.AuthError as e:
        log.warning("google sign-in refused: %s", e)
        raise HTTPException(401, "sign-in failed")
    user = await asyncio.to_thread(accounts.upsert_user, info)
    response.set_cookie(accounts.COOKIE, accounts.make_session(user["id"]), max_age=accounts.SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=_secure(request))
    log.info("sign-in: %s (new: %s)", user["id"][:8], user["market"] is None)
    return {"user": user, "unseen": await asyncio.to_thread(accounts.unseen_count, user["id"])}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(accounts.COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    user = current_user(request)
    return {"user": user, "unseen": accounts.unseen_count(user["id"]) if user else 0}


class Settings(BaseModel):
    market: str | None = Field(None, pattern="^(EG|SA|AE)$")
    email_alerts: bool | None = None
    prefs: dict[str, bool] | None = None


@app.patch("/api/me")
def update_me(body: Settings, request: Request):
    """Sign-up's region choice and the settings panel."""
    user = _need_user(request)
    new = accounts.update_user(user["id"], body.market, body.email_alerts, body.prefs)
    if body.market and body.market != user["market"]:
        name = market.MARKETS[body.market].country_ar
        if body.market in market.live():
            accounts.notify(user["id"], "info", f"أهلاً بيك! هندورلك في متاجر {name}",
                            "تقدر تتابع سعر أي منتج من زرار «تابع السعر» وهنبلغك هنا أول ما ينزل.")
        else:
            accounts.notify(user["id"], "info", f"البحث في {name} لسه بنجهزه",
                            f"لحد ما يشتغل النتايج هتبقى من متاجر {market.current().country_ar}، "
                            "وهنبعتلك إشعار هنا أول ما يبقى جاهز.")
    return {"user": new, "unseen": accounts.unseen_count(user["id"])}


@app.delete("/api/me")
def delete_me(request: Request, response: Response):
    accounts.delete_user(_need_user(request)["id"])
    response.delete_cookie(accounts.COOKIE)
    return {"ok": True}


MAX_CHAT_BYTES = 400_000  # one saved conversation (6 turns with their results are ~60-150 KB)
CHAT_ID = r"^[A-Za-z0-9_-]{8,40}$"


class SavedChat(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    data: dict


@app.get("/api/chats")
def chats(request: Request):
    """The signed-in shopper's saved conversations, newest first (any device they sign in on)."""
    return {"chats": accounts.list_chats(_need_user(request)["id"])}


@app.get("/api/chats/{chat_id}")
def get_chat(request: Request, chat_id: str = UrlPart(pattern=CHAT_ID)):
    c = accounts.get_chat(_need_user(request)["id"], chat_id)
    if not c:
        raise HTTPException(404, "not found")
    return c


@app.put("/api/chats/{chat_id}")
def save_chat(body: SavedChat, request: Request, chat_id: str = UrlPart(pattern=CHAT_ID)):
    if len(json.dumps(body.data, ensure_ascii=False).encode()) > MAX_CHAT_BYTES:
        raise HTTPException(413, "conversation too big")
    accounts.save_chat(_need_user(request)["id"], chat_id, body.title.strip(), body.data)
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
def delete_chat(request: Request, chat_id: str = UrlPart(pattern=CHAT_ID)):
    if not accounts.delete_chat(_need_user(request)["id"], chat_id):
        raise HTTPException(404, "not found")
    return {"ok": True}


class Watch(BaseModel):
    url: str = Field(pattern="^https://", max_length=1000)
    title: str = Field(max_length=300)
    store: str = Field("", max_length=100)
    image: str = Field("", max_length=1000)
    price: float = Field(gt=0)
    target_price: float | None = Field(None, gt=0)
    option: str = Field("", max_length=80)  # the size/color... the shopper searched for (the result's `option`)


@app.get("/api/watches")
def watches(request: Request):
    return {"watches": accounts.list_watches(_need_user(request)["id"])}


_watch_client = httpx.AsyncClient(timeout=10)


async def _may_open(url: str) -> bool:
    """The store's robots.txt lets us open this page (unknown = yes: the scheduled check tries it anyway)."""
    try:
        return await asyncio.wait_for(fetchers.robots_allows(_watch_client, url), timeout=10)
    except Exception:
        return True


@app.post("/api/watches")
async def add_watch(w: Watch, request: Request):
    """"تابع السعر" on a result: we tell the shopper when this product's price drops."""
    if not await _public_host(urlparse(w.url).hostname or ""):  # we open this page later: never our own network
        raise HTTPException(400, "not a store page")
    if not await _may_open(w.url):  # we must open the page again later: a copy kept elsewhere can be old
        raise HTTPException(422, "المتجر ده مش بيسمح لنا نفتح صفحات منتجاته، فمنقدرش نتابع سعره ونبلغك لما ينزل.")
    d = await asyncio.to_thread(_add_watch, w, request)
    await _track(request, "follow", detail=w.url, store=w.store, title=w.title, price=w.price)
    return d


def _add_watch(w: Watch, request: Request) -> dict:
    user = _need_user(request)
    try:
        mk = user["market"] if user["market"] in market.live() else market.current().code  # where it was found
        watch = accounts.add_watch(user["id"], w.url, w.title, w.store, w.price, mk, w.image, w.target_price,
                                   wanted=w.option)
    except ValueError:
        raise HTTPException(409, f"وصلت لأقصى عدد منتجات تتابعها ({accounts.MAX_WATCHES})")
    return {"watch": watch}


@app.delete("/api/watches/{watch_id}")
def remove_watch(watch_id: int, request: Request):
    if not accounts.remove_watch(_need_user(request)["id"], watch_id):
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.get("/api/notifications")
def notifications(request: Request):
    user = _need_user(request)
    return {"notifications": accounts.list_notifications(user["id"]), "unseen": accounts.unseen_count(user["id"])}


@app.post("/api/notifications/seen")
def notifications_seen(request: Request):
    accounts.mark_seen(_need_user(request)["id"])
    return {"ok": True}


_mail_client = httpx.AsyncClient(timeout=15)


async def _alert_followers(results: list[dict]) -> list[dict]:
    """New prices just seen: notify followers of products that got cheaper (and email the ones who allowed it)."""
    try:
        made = await asyncio.to_thread(accounts.prices_seen, results)
    except Exception:  # alerts must never break a search
        log.exception("price alerts failed")
        return []
    for n in made:
        log.info("price drop alert: %s", n["title"][:70])
        if n["email_alerts"] and accounts.email_enabled():
            await accounts.send_email(_mail_client, n["email"], n["name"], n["title"], n["body"], n["url"])
    return made


CHECK_HOURS = float(os.getenv("WATCH_CHECK_HOURS", "6"))  # how often each followed product page is re-checked


def _same_page(a: str, b: str) -> bool:
    def norm(u):
        u = urlparse(u)
        return (u.hostname or "").removeprefix("www.") + u.path.rstrip("/")
    return norm(a) == norm(b)


READ_AGAIN_KEY = "__followed_read__:"  # a followed page's last LLM-read price, kept in `searches`
READ_AGAIN_DAYS = 2  # an older read doesn't confirm a new one


async def _read_twice(url: str, price: float) -> bool:
    """True when the previous check of this page read the same price; otherwise this read is kept for the next one."""
    before = await _db(db.cached_search, READ_AGAIN_KEY + url, READ_AGAIN_DAYS * 86400)
    if before and abs(before.get("price", 0) - price) < 1:
        return True
    await _db(db.save_search, READ_AGAIN_KEY + url, market.current().code, {"price": price})
    return False


@app.post("/api/cron/check-watches")
async def check_watches(request: Request, limit: int = Query(30, ge=1, le=100), every: bool = False):
    """Opens followed product pages again to catch price drops nobody searched for. Called on a schedule by
    .github/workflows/check-watches.yml (free), with the CRON_SECRET header: the free server sleeps, so it can't
    run its own timer."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret or not hmac.compare_digest(request.headers.get("x-cron-secret", ""), secret):
        raise HTTPException(403, "forbidden")
    agent.llm.background()  # waits for shoppers' searches (this request's own context: nothing to undo)
    # every=1 (a run started by hand): all followed pages now, not only those not checked for a while
    urls = await asyncio.to_thread(accounts.urls_to_check, 0 if every else CHECK_HOURS * 3600 * 0.8, limit)
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient() as client:
        async def read(url: str, html: str, title: str, wanted: str, last: set[float]):
            """The LLM reads the page's own product (and the option its followers want): (product, problem)."""
            try:
                p = await agent._read_main_price(url, html, title, wanted=wanted)
            except Exception as e:
                return None, f"LLM read failed: {type(e).__name__}"
            if not p:
                return None, "no single price of this product on its page"
            if p.in_stock is False:
                return None, "sold out (every option)"
            if wanted and p.option_ok is not True:  # an alert only when the page says their size/color is in stock
                return None, (f"{wanted}: not offered or sold out" if p.option_ok is False
                              else f"{wanted}: the page doesn't say it's in stock")
            # a read can still be wrong: a new price counts only once a later check reads the same number
            if p.price not in last and not await _read_twice(url + ("#" + wanted if wanted else ""), p.price):
                return None, f"read {p.price:g}: waiting for the next check to read it again"
            return p, None

        async def one(url: str) -> list[tuple]:
            """[(url, option wanted, product or None, problem)] for each option this page's followers want."""
            if not url.startswith("https://") or not await _public_host(urlparse(url).hostname or ""):
                return [(url, "", None, "address not allowed")]
            try:
                products, html = await asyncio.wait_for(agent._fetch_page(client, sem, url), timeout=25)
            except Exception as e:  # a slow / blocked page: tried again next time
                return [(url, "", None, f"{type(e).__name__}: {e}"[:200])]
            title, wants = await asyncio.to_thread(accounts.followed, url)
            out = []
            for wanted, last in (wants or {"": set()}).items():
                # no product data in the page (e.g. Amazon), or a size/color to check: the LLM reads it
                if wanted or not products:
                    if not agent.llm.available():
                        out.append((url, wanted, None, f"no product read on the page ({len(html)} chars, "
                                                       f"title: {_page_title(html)})"))
                        continue
                    p, err = await read(url, html, title, wanted, last)
                    out.append((url, wanted, p, err))
                    continue
                # only a price the page itself gives for this product: its own product data, or the one product on
                # it when the page shows that price (never a copy kept elsewhere, which can be old)
                shown = agent._shown_numbers(html)
                same = [p for p in products if _same_page(p.url, url)]
                single = products[0] if len(products) == 1 and agent.shows(shown, products[0].price) else None
                p = same[0] if same else single
                found = ", ".join(f"{q.price:g}" for q in products[:5])
                out.append((url, wanted, p, None if p else f"page doesn't show the price read ({found})"))
            return out
        done = [row for rows in await asyncio.gather(*(one(u) for u in urls)) for row in rows]
    seen = [{"url": url, "price": p.price, "title": p.title, "store": p.store, "in_stock": p.in_stock,
             "old_price": p.old_price, "wanted": wanted} for url, wanted, p, _ in done if p]
    for url in {url for url, _, p, _ in done if not p} - {r["url"] for r in seen}:
        await asyncio.to_thread(accounts.checked, url)
    for url, wanted, p, err in done:
        if not p:
            log.info("watch check %s%s: %s", url[:80], f" ({wanted})" if wanted else "", err)
    if general := [r for r in seen if not r["wanted"]]:  # one size's price isn't the page's price history
        await asyncio.to_thread(db.record_prices, general, market.current().code)
    made = await _alert_followers(seen)
    return {"checked": len(urls), "prices": len(seen), "alerts": len(made),
            "failed": {url + (f" ({wanted})" if wanted else ""): err for url, wanted, p, err in done if not p}}


@app.post("/api/cron/scrape")
async def scrape_catalog(request: Request, store: str = Query("", max_length=40), category: str = Query("", max_length=40),
                         pages: int = Query(0, ge=0, le=20)):
    """Reads the big stores' category pages into our own price list (app/catalog.py). Called once or twice a day by
    .github/workflows/scrape-prices.yml with the CRON_SECRET header, one store per call (short requests suit the
    free server)."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret or not hmac.compare_digest(request.headers.get("x-cron-secret", ""), secret):
        raise HTTPException(403, "forbidden")
    stores = [d for d in store.split(",") if d] or None
    if stores and not set(stores) <= set(catalog.STORES):
        raise HTTPException(400, f"unknown store, one of: {', '.join(catalog.STORES)}")
    report = await catalog.scrape(stores, [c for c in category.split(",") if c] or None, pages or None)
    return {"report": report, "total": sum(r["products"] for r in report.values())}


@app.post("/api/cron/understand")
async def understand_catalog(request: Request, limit: int = Query(catalog.UNDERSTAND_MAX, ge=1, le=3000)):
    """The LLM reads the titles of new products in our price index (brand, model, kind in English and Arabic, size),
    so searches find them by any of those words. Run by scrape-prices.yml after the stores are read."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret or not hmac.compare_digest(request.headers.get("x-cron-secret", ""), secret):
        raise HTTPException(403, "forbidden")
    token = agent.llm.background()  # waits for shoppers' searches
    try:
        return await catalog.understand(limit)
    finally:
        agent.llm.background_end(token)


@app.post("/api/cron/deals")
async def build_deals(request: Request):
    """Makes today's deals list from the price list, every deal opened on its store page first (app/deals.py).
    Run by .github/workflows/scrape-prices.yml after the stores are read."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret or not hmac.compare_digest(request.headers.get("x-cron-secret", ""), secret):
        raise HTTPException(403, "forbidden")
    token = agent.llm.background()  # waits for shoppers' searches
    try:
        got = await deals.build()
    finally:
        agent.llm.background_end(token)
    return {k: v for k, v in got.items() if k != "deals"} | {"deals": len(got["deals"])}


@app.post("/api/cron/coupons")
async def refresh_coupons(request: Request):
    """Reads the affiliate networks' coupon feeds again (app/coupons.py). Run by scrape-prices.yml."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret or not hmac.compare_digest(request.headers.get("x-cron-secret", ""), secret):
        raise HTTPException(403, "forbidden")
    token = agent.llm.background()  # waits for shoppers' searches
    try:
        return await coupons.refresh()
    finally:
        agent.llm.background_end(token)


@app.get("/api/coupons")
async def coupons_api():
    return {"coupons": await _db(coupons.current, default=[])}


@app.get("/coupons")
async def coupons_page(request: Request):
    """Store promo codes, only from a network or the store itself (app/coupons.py)."""
    got = await _db(coupons.current, default=[])
    return Response(_coupons_html(got or [], _site(request)), media_type="text/html; charset=utf-8",
                    headers={"Cache-Control": "public, max-age=600"})


@app.get("/api/deals")
async def deals_api():
    return await _db(deals.current, default={"checked_at": None, "deals": []})


@app.get("/deals")
async def deals_page(request: Request):
    """Today's best deals, as a plain page search engines can read (people find it from Google)."""
    got = await _db(deals.current, default={"checked_at": None, "deals": []})
    return Response(_deals_html(got, _site(request)), media_type="text/html; charset=utf-8",
                    headers={"Cache-Control": "public, max-age=600"})


def _money(n: float) -> str:
    return f"{n:,.0f} ج.م"


def _chips(name: str, label: str, options: list[tuple[str, str]]) -> str:
    """A row of filter buttons (the page works without them: every card is shown until one is pressed)."""
    from html import escape as e
    if len(options) < 2:
        return ""
    buttons = "".join(f'<button type="button" data-v="{e(v)}" aria-pressed="false">{e(t)}</button>' for v, t in options)
    return (f'<div class="chips" data-f="{name}" role="group" aria-label="{e(label)}">'
            f'<button type="button" data-v="" aria-pressed="true">الكل</button>{buttons}</div>')


def _deals_html(got: dict, site: str) -> str:
    from html import escape as e
    from urllib.parse import quote
    cards, stores, cats = [], {}, {}
    for d in got.get("deals") or []:
        dom = coupons.domain_of(d["url"])
        stores[dom] = coupons.store_name(dom) if dom in coupons.STORE_NAMES else d["store"]
        cats[d["category"]] = deals.category_name(d["category"])
        why = ("أرخص سعر ليه في 30 يوم" if d["kind"] == "drop" else "خصم المتجر نفسه")
        img = (f'<img src="/img?u={quote(d["image"], safe="")}" alt="" loading="lazy" referrerpolicy="no-referrer">'
               if d.get("image") else "")
        cards.append(
            f'<a class="deal" data-store="{e(dom)}" data-cat="{e(d["category"])}" href="{e(affiliate.link(d["url"]))}" '
            f'target="_blank" rel="noopener nofollow sponsored"><span class="pic">{img}'
            f'<b class="pct">-{d["pct"]}%</b><span class="ok">✓ السعر متأكد منه</span></span>'
            f'<span class="t" dir="auto">{e(d["title"])}</span>'
            f'<span class="p">{_money(d["price"])}</span><s>{_money(d["was"])}</s>'
            f'<span class="why{" real" if d["kind"] == "drop" else ""}">{e(why)}</span>'
            f'<span class="st">{e(stores[dom])}</span></a>')
    when = got.get("checked_at")
    body = (f'<div class="grid">{"".join(cards)}</div>' if cards else
            '<p class="empty">لسه مفيش عروض متأكدين منها النهارده. بنراجع الأسعار مرتين في اليوم، ارجع كمان شوية '
            'أو <a href="/">دوّر على اللي انت عايزه</a>.</p>')
    filters = (_chips("store", "المتجر", sorted(stores.items(), key=lambda x: x[1])) +
               _chips("cat", "القسم", sorted(cats.items(), key=lambda x: x[1])))
    stamp = (f'<p class="when">فتحنا صفحة كل عرض في المتجر وتأكدنا من سعره وإنه متوفر <span data-ts="{when:.0f}"></span>. '
             'الأسعار بتتغير، فبص على السعر في صفحة المتجر قبل ما تطلب.</p>' if when else "")
    return (_shell("deals", site).replace("{TITLE}", "أحسن العروض النهارده في مصر: موبايلات وأجهزة بخصم حقيقي | Easy Found")
            .replace("{DESC}", "عروض النهارده من أمازون وجوميا: موبايلات وشاشات وأجهزة نزل سعرها فعلاً عن آخر 30 يوم، "
                               "واتأكدنا من سعر كل واحد في صفحة المتجر.")
            .replace("{H1}", "أحسن العروض النهارده")
            .replace("{LEAD}", "منتجات نزل سعرها فعلاً عن أقل سعر شفناه في آخر 30 يوم، وبعدها خصومات المتاجر الكبيرة.")
            .replace("{BODY}", stamp + filters + body))


def _coupons_html(rows: list[dict], site: str) -> str:
    from html import escape as e
    cards, stores = [], {}
    for c in rows:
        stores[c["store"]] = c["store_name"]
        ends = (f'<span data-day="{c["expires_at"]:.0f}"></span>' if c.get("expires_at") else "")
        code = (f'<button type="button" class="code" data-code="{e(c["code"])}" aria-label="انسخ الكود {e(c["code"])}">'
                f'<span dir="ltr">{e(c["code"])}</span><small>انسخ</small></button>' if c["code"] else
                '<span class="nocode">من غير كود: الخصم بيتطبق من اللينك</span>')
        cards.append(
            f'<div class="coupon" data-store="{e(c["store"])}"><div class="ch"><b>{e(c["store_name"])}</b>{ends}</div>'
            f'<p class="t" dir="auto">{e(c["title"])}</p>{code}'
            f'<a class="go" href="{e(c["link"])}" target="_blank" rel="noopener nofollow sponsored">افتح {e(c["store_name"])}</a>'
            f'<span class="src">{e(c["from"])}</span></div>')
    body = (f'<div class="cgrid">{"".join(cards)}</div>' if cards else
            '<p class="empty">مفيش أكواد خصم شغالة دلوقتي. بنحط هنا بس الأكواد اللي جاية من المتجر نفسه أو من شبكة '
            'التسويق الرسمية بتاعته، عشان كل كود تلاقيه يكون شغال فعلاً. بص على <a href="/deals">عروض النهارده</a>.</p>')
    note = ('<p class="when">كل كود هنا جاي من المتجر نفسه أو من شبكة التسويق الرسمية بتاعته (مكتوب تحت كل كود). '
            'المتجر ممكن يوقف الكود قبل ميعاده أو يحط له شروط، فاقرا الشروط في صفحة الدفع.</p>' if cards else "")
    return (_shell("coupons", site).replace("{TITLE}", "أكواد خصم شغالة لنون وأمازون وجوميا | Easy Found")
            .replace("{DESC}", "أكواد خصم وكوبونات من المتاجر نفسها أو شبكاتها الرسمية: نون وأمازون مصر وجوميا وغيرهم. "
                               "مفيش كود منتهي.")
            .replace("{H1}", "أكواد الخصم")
            .replace("{LEAD}", "انسخ الكود وافتح المتجر، وحطه في صفحة الدفع.")
            .replace("{BODY}", note + _chips("store", "المتجر", sorted(stores.items(), key=lambda x: x[1])) + body))


def _shell(page: str, site: str) -> str:
    tabs = "".join(f'<a href="/{k}"{" aria-current=page" if k == page else ""}>{t}</a>'
                   for k, t in (("deals", "🔥 العروض"), ("coupons", "🏷️ أكواد الخصم")))
    return DEALS_PAGE.replace("{SITE}", site).replace("{PAGE}", page).replace("{TABS}", tabs)


DEALS_PAGE = """<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{DESC}">
<link rel="canonical" href="{SITE}/{PAGE}"><link rel="icon" href="/logo.svg?v=2" type="image/svg+xml">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#F4F7F6;--card:#fff;--text:#13221F;--muted:#56665F;--border:#DCE4E1;--accent:#0F766E;--on:#fff;--deal:#B93A0A;--deal-bg:#C2410C;--soft:#DDF1EC;--ink:#0B5F58;--ok:#047857}
@media (prefers-color-scheme:dark){:root{--bg:#0C1312;--card:#141F1D;--text:#E6EEEC;--muted:#9DAEA9;--border:#263633;--accent:#2DD4BF;--on:#04221F;--deal:#FDBA74;--deal-bg:#C2410C;--soft:#103A35;--ink:#5EEAD4;--ok:#047857}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 "IBM Plex Sans Arabic",system-ui,Tahoma,sans-serif}
main{max-width:1080px;margin:0 auto;padding:16px 16px 48px}header{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px;margin-bottom:6px}
header>a{color:inherit;text-decoration:none;display:flex;align-items:center;gap:8px;font-weight:700}
nav.tabs{display:flex;gap:6px;margin-inline-start:auto}nav.tabs a{padding:6px 14px;border-radius:999px;text-decoration:none;color:var(--text);border:1px solid var(--border);background:var(--card);font-weight:600;font-size:14px}
nav.tabs a[aria-current]{background:var(--accent);color:var(--on);border-color:var(--accent)}
h1{font-size:24px;margin:10px 0 2px}.lead,.when{color:var(--muted);margin:0 0 12px}.when{font-size:13px}
.chips{display:flex;gap:6px;overflow-x:auto;padding:2px 0 8px;scrollbar-width:none}.chips button{flex:none;height:34px;padding:0 14px;border-radius:999px;border:1px solid var(--border);background:var(--card);color:var(--text);font:inherit;font-size:14px;cursor:pointer}
.chips button[aria-pressed=true]{background:var(--soft);border-color:var(--accent);color:var(--ink);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-top:4px}
.deal{display:flex;flex-direction:column;gap:2px;background:var(--card);border:1px solid var(--border);border-radius:14px;padding:8px;color:inherit;text-decoration:none;min-width:0}
.deal:hover{border-color:var(--accent)}.pic{position:relative;display:block;aspect-ratio:1;background:#fff;border-radius:10px;overflow:hidden;margin-bottom:6px}
.pic img{width:100%;height:100%;object-fit:contain}.pct{position:absolute;top:6px;inset-inline-start:6px;background:var(--deal-bg);color:#fff;font-size:13px;padding:1px 8px;border-radius:8px}
.ok{position:absolute;bottom:6px;inset-inline-start:6px;background:var(--ok);color:#fff;font-size:11.5px;padding:1px 8px;border-radius:8px}
.t{font-weight:600;font-size:14px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical}
.p{font-size:17px;font-weight:700}.deal s{font-size:13px;color:var(--muted)}
.why{font-size:12.5px;color:var(--muted)}.why.real{color:var(--deal);font-weight:600}.st{font-size:12.5px;color:var(--ink)}
.cgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.coupon{display:flex;flex-direction:column;gap:8px;background:var(--card);border:1px dashed var(--accent);border-radius:14px;padding:14px}
.ch{display:flex;justify-content:space-between;gap:8px;font-size:13px;color:var(--muted)}.ch b{color:var(--text);font-size:16px}
.coupon .t{margin:0;-webkit-line-clamp:3}.code{display:flex;justify-content:space-between;align-items:center;gap:8px;border:1px solid var(--border);border-radius:10px;background:var(--soft);color:var(--ink);padding:8px 12px;font:700 17px/1.2 ui-monospace,monospace;cursor:pointer}
.code small{font:600 13px "IBM Plex Sans Arabic",system-ui,sans-serif}.nocode{font-size:13px;color:var(--muted)}
.go{text-align:center;background:var(--accent);color:var(--on);padding:8px;border-radius:10px;text-decoration:none;font-weight:600}.src{font-size:12px;color:var(--muted)}
.empty{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px}a{color:var(--accent)}[hidden]{display:none!important}
.cta{display:inline-block;margin:16px 0 0;color:var(--accent);font-weight:600}
</style></head><body><main>
<header><a href="/"><img src="/logo.svg?v=2" width="32" height="32" alt="">Easy Found</a><nav class="tabs">{TABS}</nav></header>
<h1>{H1}</h1><p class="lead">{LEAD}</p>
{BODY}
<a class="cta" href="/">مش لاقي اللي عايزه؟ دوّر على أرخص سعر لأي حاجة ←</a>
</main><script>
document.querySelectorAll('[data-ts]').forEach(function(el){var d=new Date(el.dataset.ts*1000);
el.textContent='الساعة '+d.toLocaleTimeString('ar-EG',{hour:'numeric',minute:'2-digit'})+' يوم '+d.toLocaleDateString('ar-EG',{weekday:'long'});});
document.querySelectorAll('[data-day]').forEach(function(el){var d=new Date(el.dataset.day*1000-1);
el.textContent='لحد '+d.toLocaleDateString('ar-EG',{day:'numeric',month:'long',timeZone:'Africa/Cairo'});});
var picked={};
document.querySelectorAll('.chips').forEach(function(row){row.addEventListener('click',function(e){var b=e.target.closest('button');if(!b)return;
row.querySelectorAll('button').forEach(function(x){x.setAttribute('aria-pressed',x===b?'true':'false');});picked[row.dataset.f]=b.dataset.v;
document.querySelectorAll('.deal,.coupon').forEach(function(c){c.hidden=Object.keys(picked).some(function(k){return picked[k]&&c.dataset[k]!==picked[k];});});});});
document.addEventListener('click',function(e){var b=e.target.closest('.code');if(!b)return;var s=b.querySelector('small');
(navigator.clipboard?navigator.clipboard.writeText(b.dataset.code):Promise.reject()).then(function(){s.textContent='اتنسخ ✓';},function(){s.textContent='انسخه بإيدك';});});
</script></body></html>"""


@app.get("/bot")
def bot_info():
    """Who our crawler is (its name points here, as stores ask crawlers to say who runs them)."""
    return Response(BOT_PAGE, media_type="text/html; charset=utf-8")


BOT_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DealHunterBot</title><body style="font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;
line-height:1.6"><h1>DealHunterBot</h1><p>DealHunterBot is the crawler of <a href="/">Easy Found</a>, a price-comparison
site for shoppers in Egypt. Once or twice a day it reads a few public category pages of large online stores (for
example the mobile phones category) to know their current prices, and it sends shoppers to the store's own page to
buy.</p><ul><li>It obeys robots.txt.</li><li>It opens one page at a time per store, with a pause between pages (a
few dozen pages per store per day).</li><li>It never signs in, never adds to cart and never reads personal data.</li>
</ul><p>To ask us to slow down or stop, write to us through the feedback box on <a href="/">the site</a>.</p>"""


def _page_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    return " ".join(m.group(1).split())[:60] if m else "-"


@app.api_route("/health", methods=["GET", "HEAD"])  # uptime monitors (UptimeRobot) ask with HEAD
def health():
    return {"ok": True}
