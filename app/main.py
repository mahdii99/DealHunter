import asyncio
import hashlib
import hmac
import io
import ipaddress
import json
import logging
import os
import time
from collections import OrderedDict
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

from . import accounts, agent, db, market, trending  # noqa: E402  (after .env is loaded)
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

app = FastAPI(title="Easy Found")
# Same request again within this time = answered from the price database (instant, saves the free quotas).
CACHE_TTL = float(os.getenv("CACHE_HOURS", "6")) * 3600
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
    return Response(f"User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /img\nSitemap: {_site(request)}/sitemap.xml\n",
                    media_type="text/plain")


@app.get("/sitemap.xml")
def sitemap(request: Request):
    urls = "".join(f"<url><loc>{_site(request)}{p}</loc></url>" for p in ("/", "/privacy"))
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
    return await _search(q)


async def _db(fn, *args, default=None):
    """A database call off the event loop (a slow or reconnecting database must not freeze every other visitor).
    Bookkeeping errors are logged, not shown: a search still works without the cache or price history."""
    try:
        return await asyncio.to_thread(fn, *args)
    except Exception:
        log.exception("database call %s failed", fn.__name__)
        return default


async def _search(q: str, plan: dict | None = None, budget_text: str | None = None) -> dict:
    mk = market.current().code
    key = db.cache_key(q, mk)
    if not DEMO and (hit := await _db(db.cached_search, key, CACHE_TTL)):
        log.info("served from cache")
        log.info("=" * 70 + "\n")
        return {**hit, "cached": True}
    t = time.monotonic()
    payload = await agent.run(q, plan=plan, budget_text=budget_text)
    log.info("DONE in %.1fs: %d results (%s)", time.monotonic() - t, len(payload["results"]), payload.get("source"))
    log.info("=" * 70 + "\n")
    payload.update(query=q, count=len(payload["results"]), cached=False, demo=DEMO, market=mk)
    if not DEMO and payload["results"]:
        await _db(db.add_price_context, payload["results"])  # before saving today's prices
        await _db(db.record_prices, payload["results"], mk)
        await _alert_followers(payload["results"])
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
    return await _chat_turn(turn, client_id(request), user)


@app.post("/api/chat/stream")
async def chat_stream(turn: ChatTurn, request: Request):
    """Same as /api/chat, but sends progress lines while it works (NDJSON: one JSON object per line):
    {"type": "status", "text": ...} and {"type": "reply", "text": <the reply so far, while the LLM writes it>} ...
    then {"type": "done", "data": <the /api/chat answer>}."""
    queue: asyncio.Queue = asyncio.Queue()
    client = client_id(request)
    user = await asyncio.to_thread(current_user, request)

    async def work():
        token = agent.PROGRESS.set(lambda text: queue.put_nowait({"type": "status", "text": text}))
        said = agent.REPLY.set(lambda text: queue.put_nowait({"type": "reply", "text": text}))  # typed as it comes
        try:
            queue.put_nowait({"type": "done", "data": await _chat_turn(turn, client, user)})
        except HTTPException as e:
            queue.put_nowait({"type": "error", "status": e.status_code})
        except Exception:  # the page shows a friendly error; the details are in the server log
            log.exception("chat stream failed")
            queue.put_nowait({"type": "error", "status": 500})
        finally:
            agent.REPLY.reset(said)
            agent.PROGRESS.reset(token)

    async def lines():
        task = asyncio.create_task(work())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in ("done", "error"):
                    break
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
        d = await _chat_turn_in_market(turn, client, USER_DAILY_LIMIT if user and USER_DAILY_LIMIT else DAILY_LIMIT)
    finally:
        market.reset(token)
    if user:
        await asyncio.to_thread(_personalize, d, user)
    return d


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
    if agent.llm.available():
        try:
            plan = await agent.plan_turn(messages, turn.request, [r.model_dump() for r in turn.shown], (lo, hi))
        except agent.llm.LLMError as e:
            log.warning("chat planner failed: %s", e)
            failed = e
    if plan and plan["action"] == "ask":  # like a salesperson: one question first, to recommend the right thing
        log.info("=" * 70 + "\n")
        return {"action": "ask", "question": plan["question"], "options": plan.get("options") or [],
                "request": turn.request, "results": [], "demo": DEMO}
    if plan and plan["action"] == "recommend":  # products that fit, from the LLM's knowledge, before any search
        await agent.price_recommendations(plan, lo, hi)  # ...with today's prices, so none is over the budget
        log.info("=" * 70 + "\n")
        return {"action": "recommend", "answer": plan["reply"], "recommendations": plan["recommendations"],
                "over_budget": plan.get("over_budget") or [],
                "suggestions": plan["suggestions"], "request": turn.request, "results": [], "demo": DEMO}
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
    request = plan["request"] if plan else " ".join(filter(None, [turn.request, newest]))  # no LLM: just append
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
        payload = await agent.run_rules(request, note=f"الـ LLM مردش ({failed})، فاستخدمنا الفهم البسيط.")
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
SPAM_LIMITS = {"report": 60, "feedback": 20}
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


@app.get("/api/watches")
def watches(request: Request):
    return {"watches": accounts.list_watches(_need_user(request)["id"])}


@app.post("/api/watches")
async def add_watch(w: Watch, request: Request):
    """"تابع السعر" on a result: we tell the shopper when this product's price drops."""
    if not await _public_host(urlparse(w.url).hostname or ""):  # we open this page later: never our own network
        raise HTTPException(400, "not a store page")
    return await asyncio.to_thread(_add_watch, w, request)


def _add_watch(w: Watch, request: Request) -> dict:
    user = _need_user(request)
    try:
        mk = user["market"] if user["market"] in market.live() else market.current().code  # where it was found
        watch = accounts.add_watch(user["id"], w.url, w.title, w.store, w.price, mk, w.image, w.target_price)
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


SEARCH_RECHECKS = 10  # followed pages per re-check read through the search service (each = 1 search + 1 LLM read)
CHECK_HOURS = float(os.getenv("WATCH_CHECK_HOURS", "6"))  # how often each followed product page is re-checked


def _same_page(a: str, b: str) -> bool:
    def norm(u):
        u = urlparse(u)
        return (u.hostname or "").removeprefix("www.") + u.path.rstrip("/")
    return norm(a) == norm(b)


@app.post("/api/cron/check-watches")
async def check_watches(request: Request, limit: int = Query(30, ge=1, le=100), every: bool = False):
    """Opens followed product pages again to catch price drops nobody searched for. Called on a schedule by
    .github/workflows/check-watches.yml (free), with the CRON_SECRET header: the free server sleeps, so it can't
    run its own timer."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret or not hmac.compare_digest(request.headers.get("x-cron-secret", ""), secret):
        raise HTTPException(403, "forbidden")
    # every=1 (a run started by hand): all followed pages now, not only those not checked for a while
    urls = await asyncio.to_thread(accounts.urls_to_check, 0 if every else CHECK_HOURS * 3600 * 0.8, limit)
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient() as client:
        async def one(url: str):
            if not url.startswith("https://") or not await _public_host(urlparse(url).hostname or ""):
                return url, None, "address not allowed"
            try:
                products, _ = await asyncio.wait_for(agent._fetch_page(client, sem, url), timeout=25)
            except Exception as e:  # a slow / blocked page: tried again next time
                return url, None, type(e).__name__
            same = [p for p in products if _same_page(p.url, url)]
            p = same[0] if same else products[0] if len(products) == 1 else None
            return url, p, None if p else "no price on page"
        done = await asyncio.gather(*(one(u) for u in urls))
        # pages we couldn't open or read: the search service's copy of them (robots.txt / bot walls, e.g. Amazon)
        tries = [i for i, (url, p, err) in enumerate(done) if not p and err != "address not allowed"][:SEARCH_RECHECKS]

        async def by_search(i):
            url, _, err = done[i]
            try:
                p, why = await agent.price_from_search(client, url, await asyncio.to_thread(accounts.watch_title, url))
            except Exception as e:
                return url, None, f"{err} / search: {type(e).__name__}: {str(e)[:120]}"
            return url, p, None if p else f"{err} / {why}"
        for i, res in zip(tries, await asyncio.gather(*(by_search(i) for i in tries))):
            done[i] = res
    seen = [{"url": url, "price": p.price, "title": p.title, "store": p.store, "in_stock": p.in_stock,
             "old_price": p.old_price} for url, p, _ in done if p]
    for url, p, err in done:
        if not p:
            log.info("watch check %s: %s", url[:80], err)
            await asyncio.to_thread(accounts.checked, url)
    if seen:
        await asyncio.to_thread(db.record_prices, seen, market.current().code)
    made = await _alert_followers(seen)
    return {"checked": len(urls), "prices": len(seen), "alerts": len(made),
            "failed": {url: err for url, p, err in done if not p}}


@app.get("/health")
def health():
    return {"ok": True}
