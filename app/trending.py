"""The request ideas under the search box ("هدية لأمي في حدود ٥٠٠"...): made fresh twice a day from what people
search on the site lately, what's trending in the market's stores this week, and the occasions coming up
(Black Friday, Ramadan, back to school...). One web search + one LLM call per refresh, kept in the database."""
import asyncio
import json
import logging
import time

import httpx

from . import db, llm, market, websearch

log = logging.getLogger("uvicorn.error")
REFRESH = 12 * 3600   # new ideas twice a day
SHOWN = 6
MAX_LEN = 45          # a chip, not a sentence
_memory: dict[str, dict] = {}  # market -> {"ar", "en", "at"}: no database read on every page load
_running: dict[str, asyncio.Task] = {}

PROMPT = """You pick the {n} example requests shown under the search box of a price-comparison site for shoppers
in {country}. Today is {today}. A shopper taps one to search it. Make them what people want to buy RIGHT NOW:
- what shoppers searched on the site lately (most repeated themes first):
{recent}
- what the web says is trending / best-selling in {country} this week:
{web}
- occasions coming up soon in {country} (sales seasons like White/Black Friday, Ramadan, Eid, back to school,
  Mother's Day, Valentine's, summer, winter...), only if they are within about a month of today.
Rules: each one is a short natural request in Egyptian Arabic, like a shopper would type it (e.g. "هدية لأمي في
حدود ٥٠٠", "سخان كهربا ٥٠ لتر رخيص"), at most 40 characters, with a budget only when it helps. Mix different
kinds of products. Never copy anything personal, rude or odd from the searches; generic themes only.
Reply JSON: {{"ar": ["..."], "en": ["the same requests in short English, same order"]}}"""


def _ok(x) -> bool:
    return isinstance(x, str) and 2 <= len(x.strip()) <= MAX_LEN


def _pairs(ar, en) -> tuple[list[str], list[str]]:
    """Arabic ideas that fit on a chip, with their English twins (same position); [] English if any is missing."""
    ar = ar if isinstance(ar, list) else []
    en = en if isinstance(en, list) and len(en) == len(ar) else [None] * len(ar)
    out_ar, out_en = [], []
    for a, e in zip(ar, en):
        if _ok(a) and a.strip() not in out_ar and len(out_ar) < SHOWN:
            out_ar.append(a.strip())
            out_en.append(e.strip() if _ok(e) else None)
    return out_ar, out_en if all(out_en) else []


async def _make(mk: str) -> dict | None:
    m = market.MARKETS[mk]
    recent = await asyncio.to_thread(db.recent_requests, mk)
    web = []
    if websearch.available():
        month = time.strftime("%B %Y")
        try:
            async with httpx.AsyncClient() as client:
                web = await websearch.search(client, f"الأكثر مبيعا والأكثر طلبا أونلاين في {m.country_ar} {month}",
                                             max_results=8)
        except (websearch.SearchError, httpx.HTTPError) as e:
            log.info("trending: web search failed: %s", e)
    prompt = PROMPT.format(
        n=SHOWN, country=m.country_en, today=time.strftime("%A %d %B %Y"),
        recent="\n".join(f"  {r}" for r in recent[:60]) or "  (none yet)",
        web="\n".join(f"  {h.get('title', '')[:100]} | {h.get('snippet', '')[:160]}" for h in web) or "  (none)")
    data = await llm.ask_json(prompt, timeout=15, budget=25, prefer="strong", step="trending")
    ar, en = _pairs(data.get("ar"), data.get("en"))
    if len(ar) < 3:
        return None
    return {"ar": ar, "en": en, "at": time.time()}


async def _refresh(mk: str) -> dict | None:
    try:
        fresh = await _make(mk)
    except Exception as e:  # LLM busy / no key / database down: keep the ideas we have
        log.info("trending: refresh failed: %s", e)
        return None
    if fresh:
        _memory[mk] = fresh
        try:
            await asyncio.to_thread(db.save_search, db.cache_key(db.TRENDING_KEY, mk), mk, fresh)
        except Exception:
            log.exception("trending: saving failed")
        log.info("trending: %s", fresh["ar"])
    return fresh


async def suggestions(wait: float = 20) -> dict:
    """{"ar": [...], "en": [...]} or {} (the page then keeps its built-in examples). Old ideas are returned at once
    while new ones are made in the background; with none at all, waits up to `wait` seconds for them."""
    mk = market.current().code
    have = _memory.get(mk)
    if have is None:
        try:
            have = await asyncio.to_thread(db.cached_search, db.cache_key(db.TRENDING_KEY, mk), 30 * 24 * 3600)
        except Exception:
            have = None
        if have:
            _memory[mk] = have
    if not llm.available():
        return _shown(have)
    if (not have or time.time() - have.get("at", 0) > REFRESH) and mk not in _running:
        task = asyncio.ensure_future(_refresh(mk))
        _running[mk] = task
        task.add_done_callback(lambda _: _running.pop(mk, None))
    if not have and mk in _running:
        done, _ = await asyncio.wait({_running[mk]}, timeout=wait)
        have = _memory.get(mk)
    return _shown(have)


def _shown(d: dict | None) -> dict:
    return {"ar": d["ar"], "en": d.get("en") or []} if d and d.get("ar") else {}
