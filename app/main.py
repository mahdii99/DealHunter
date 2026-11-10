import asyncio
import json
import logging
import os
import time
from pathlib import Path

from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
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

from . import agent, db, market  # noqa: E402  (after .env is loaded)
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

app = FastAPI(title="DealHunter")
# Same request again within this time = answered from the price database (instant, saves the free quotas).
CACHE_TTL = float(os.getenv("CACHE_HOURS", "6")) * 3600


@app.get("/")
def index():
    # no-cache: the browser always picks up the latest page after an update
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/search")
async def search(q: str = Query(..., min_length=2, max_length=300)):
    log.info("=" * 70)
    log.info("NEW SEARCH: %s", q)
    return await _search(q)


async def _search(q: str, plan: dict | None = None, budget_text: str | None = None) -> dict:
    mk = market.current().code
    key = db.cache_key(q, mk)
    if not DEMO and (hit := db.cached_search(key, CACHE_TTL)):
        log.info("served from cache")
        log.info("=" * 70 + "\n")
        return {**hit, "cached": True}
    t = time.monotonic()
    payload = await agent.run(q, plan=plan, budget_text=budget_text)
    log.info("DONE in %.1fs: %d results (%s)", time.monotonic() - t, len(payload["results"]), payload.get("source"))
    log.info("=" * 70 + "\n")
    for r in payload["results"]:
        if r.get("image"):
            IMAGE_HOSTS.add(urlparse(r["image"]).hostname or "")
    payload.update(query=q, count=len(payload["results"]), cached=False, demo=DEMO, market=mk)
    if not DEMO and payload["results"]:
        db.add_price_context(payload["results"])  # before saving today's prices: compare with what we saw earlier
        db.record_prices(payload["results"], mk)
        if payload["mode"] == "llm":
            db.save_search(key, mk, payload)
    return payload


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(max_length=2000)


class ChatTurn(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)
    request: str = Field("", max_length=600)  # the standalone request of the previous search, "" at the start


@app.post("/api/chat")
async def chat(turn: ChatTurn):
    """A chat turn: the newest message is read together with the earlier ones, so a follow-up like
    'سامسونج بس' narrows the previous request instead of starting a new search."""
    return await _chat_turn(turn)


@app.post("/api/chat/stream")
async def chat_stream(turn: ChatTurn):
    """Same as /api/chat, but sends progress lines while it works (NDJSON: one JSON object per line):
    {"type": "status", "text": ...} ... then {"type": "done", "data": <the /api/chat answer>}."""
    queue: asyncio.Queue = asyncio.Queue()

    async def work():
        token = agent.PROGRESS.set(lambda text: queue.put_nowait({"type": "status", "text": text}))
        try:
            queue.put_nowait({"type": "done", "data": await _chat_turn(turn)})
        except HTTPException as e:
            queue.put_nowait({"type": "error", "status": e.status_code})
        except Exception:  # the page shows a friendly error; the details are in the server log
            log.exception("chat stream failed")
            queue.put_nowait({"type": "error", "status": 500})
        finally:
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


async def _chat_turn(turn: ChatTurn) -> dict:
    messages = [m.model_dump() for m in turn.messages]
    said = " | ".join(m["text"] for m in messages if m["role"] == "user")
    newest = next((m["text"] for m in reversed(messages) if m["role"] == "user"), "").strip()
    if len(newest) < 2:
        raise HTTPException(422, "empty message")
    log.info("=" * 70)
    log.info("CHAT: %s (current request: %s)", newest, turn.request or "-")
    plan, failed = None, None
    if agent.llm.available():
        try:
            plan = await agent.plan_turn(messages, turn.request)
        except agent.llm.LLMError as e:
            log.warning("chat planner failed: %s", e)
            failed = e
    if plan and plan["action"] == "ask":
        log.info("=" * 70 + "\n")
        return {"action": "ask", "question": plan["question"],
                "options": [o for o in plan.get("options") or [] if isinstance(o, str)][:4],
                "request": turn.request, "understood": plan.get("understood", ""), "results": [], "demo": DEMO}
    if plan and plan["action"] == "answer":  # a question about the shown results: answer it, no new search
        log.info("=" * 70 + "\n")
        return {"action": "answer", "answer": plan["answer"], "request": turn.request or plan["request"],
                "results": [], "demo": DEMO}
    if plan and plan.get("understood"):
        agent._progress(plan["understood"])
    request = plan["request"] if plan else " ".join(filter(None, [turn.request, newest]))  # no LLM: just append
    if failed:  # don't wait for the LLM a second time: simple rules for this turn
        payload = await agent.run_rules(request, note=f"الـ LLM مردش ({failed})، فاستخدمنا الفهم البسيط.")
        payload.update(query=request, count=len(payload["results"]), cached=False, demo=DEMO)
    else:
        payload = await _search(request, plan=plan, budget_text=said)
    return {**payload, "action": "search"}


# Some stores block their images when shown on another site, so the page loads them through us.
# Only hosts of images that appeared in our own results are allowed, so it can't be used as an open proxy.
IMAGE_HOSTS = {"media.btech.com", "eg.jumia.is"}
_img_client = httpx.AsyncClient(timeout=10, follow_redirects=True)


@app.get("/img")
async def image(u: str = Query(..., max_length=1000)):
    host = urlparse(u).hostname or ""
    if urlparse(u).scheme != "https" or host not in IMAGE_HOSTS:
        raise HTTPException(400, "image host not allowed")
    headers = {**HEADERS, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
               "Referer": f"https://{host.split('.', 1)[1]}/"}
    try:
        r = await _img_client.get(u, headers=headers)
    except httpx.HTTPError as e:
        log.warning("img %s failed: %s", u, type(e).__name__)
        raise HTTPException(502, "image fetch failed")
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200 or not ctype.startswith("image/"):
        log.warning("img %s -> %s %s", u, r.status_code, ctype)
        raise HTTPException(404, "image not found")
    return Response(r.content, media_type=ctype, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/history")
def price_history(url: str = Query(..., max_length=1000)):
    """A product's saved prices, oldest first (for the price chart / 'is this discount real?')."""
    points = db.history(url)
    return {"url": url, "points": points, "lowest": min((p["price"] for p in points), default=None)}


@app.get("/health")
def health():
    return {"ok": True}
