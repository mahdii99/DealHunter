"""LLM client, free tiers only.

Tries, in order, until one answers:
  1. Google Gemini (GEMINI_API_KEY, https://aistudio.google.com/apikey), several models
  2. Groq (GROQ_API_KEY, https://console.groq.com/keys), optional backup
  (and the other OpenAI-style free tiers in COMPAT; Gemma models come free with the Gemini key, with a much bigger
  daily quota than the Gemini models themselves, so bulk work goes there)
Each model's calls are counted per day (saved in the database: a restart doesn't forget them). Background work (the
deals list, reading new titles, learning store pages) stops at BACKGROUND_SHARE of a model's free daily requests, so
shoppers' searches always find quota left.
A model that is busy (429/5xx) or times out is skipped for a few minutes (longer each time it keeps timing out); one whose free DAILY quota is used up is
skipped until the quota resets (midnight US Pacific time). The last model that worked is tried first.
Both are saved to debug/llm_state.json, so a server restart (--reload) doesn't retry dead models again.
"""
import asyncio
import contextvars
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from . import problems

RETRY_STATUS = {429, 500, 502, 503, 504}
log = logging.getLogger("uvicorn.error")
TOTAL_BUDGET = float(os.getenv("LLM_BUDGET_SECONDS", "25"))  # give up and use the simple fallback after this


_TEMPERATURE: contextvars.ContextVar = contextvars.ContextVar("temperature", default=0.0)  # set per ask_json call
# Live searches first: the free models' per-minute limits are shared by everything. Work nobody is waiting for (the
# deals list, reading new titles, checking followed products, learning a store's page) waits while a shopper's search
# is running, so twice a day the cron jobs don't make shoppers' searches slow or empty (live test 2026-11-24).
_LIVE = 0
_BACKGROUND: contextvars.ContextVar = contextvars.ContextVar("llm_background", default=False)
BACKGROUND_WAIT = 90  # seconds a background call waits at most for the live searches to end


class _Live:
    def __enter__(self):
        global _LIVE
        _LIVE += 1

    def __exit__(self, *exc):
        global _LIVE
        _LIVE -= 1


def live() -> _Live:
    """`with llm.live():` around a shopper's search: background LLM work waits meanwhile."""
    return _Live()


def background() -> contextvars.Token:
    """Marks this task's LLM calls (and the tasks it starts) as background work; undo with background_end(token)."""
    return _BACKGROUND.set(True)


def background_end(token: contextvars.Token) -> None:
    _BACKGROUND.reset(token)


def live_searches() -> int:
    return _LIVE


BUSY_TEXT = "الموديلات المجانية مشغولة دلوقتي"


class LLMError(Exception):
    """No model answered. str(e) is safe to show a shopper; e.details (which models failed, how) is for the log."""
    def __init__(self, message: str = BUSY_TEXT, details: str = ""):
        super().__init__(message)
        self.details = details


_USAGE: contextvars.ContextVar = contextvars.ContextVar("llm_usage", default=None)
_IDLE_USAGE = {"calls": 0, "tokens": 0, "by_step": {}}  # calls made outside any search (tests, scripts)


def usage() -> dict:
    """LLM use of the current search (its own count: searches running at the same time don't mix)."""
    return _USAGE.get() or _IDLE_USAGE


class _UsageView:
    """llm.USAGE["calls"] reads the current search's count (kept for older callers)."""
    def __getitem__(self, key):
        return usage()[key]

    def get(self, key, default=None):
        return usage().get(key, default)


USAGE = _UsageView()


def reset_usage() -> None:
    """A new count for this search. Tasks it starts later share it (they copy the context, not the dict)."""
    _USAGE.set({"calls": 0, "tokens": 0, "by_step": {}})


def _failed(step: str, model: str, why: str) -> None:
    """A model that didn't answer this step, kept with the search's count: a slow step can be traced to its models
    (the live search check prints them)."""
    s = usage()["by_step"].setdefault(step, {"calls": 0, "tokens": 0})
    fails = s.setdefault("fails", {})
    key = f"{model}: {why}"
    fails[key] = fails.get(key, 0) + 1


def _count(provider: str, r: httpx.Response, step: str, model: str = "") -> None:
    try:
        data = r.json()
        u = data.get("usageMetadata", {}).get("totalTokenCount") if provider == "gemini" else \
            (data.get("usage") or {}).get("total_tokens")
    except ValueError:
        u = None
    now = usage()
    now["calls"] += 1
    now["tokens"] += int(u or 0)
    s = now["by_step"].setdefault(step, {"calls": 0, "tokens": 0})
    s["calls"] += 1
    s["tokens"] += int(u or 0)
    if model:
        s["model"] = model  # which model answered this step last (e.g. who judged)
        _tally(provider, model, step, int(u or 0))


def last_model() -> str:
    return _LAST_GOOD[0] if _LAST_GOOD else "?"


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or any(_configured(p) for p in COMPAT))


# Gemma (open models, free on the same Gemini key): ~14,400 requests a day each, against 20-500 for the Gemini
# models (free limits as of 2026-11). Slower and weaker than Gemini Flash, so they read pages and titles; planning
# and judging keep using the strong models. They don't take JSON mode or a thinking setting.
GEMMA_MIN_SIZE = 12  # billions of parameters: the small Gemma models get product names and prices wrong

# lite first: it answers in ~3s; the full flash model often times out on the free tier (mahdy's check, 2026-11-10)
GEMINI_PREFERRED = ("gemini-flash-lite-latest", "gemini-flash-latest")
# (each Groq model has its own per-minute and daily limits; llama-4-scout and llama-3.1-8b have the biggest ones,
# so page reading spreads over them while gpt-oss-120b is kept for planning and judging)
GROQ_PREFERRED = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
                  "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b", "llama-3.1-8b-instant",
                  "meta-llama/llama-4-maverick-17b-128e-instruct")
FAST_BULK = ("groq", "cerebras", "cloudflare")  # answer in ~1-3 s: page reading goes here first, spread over their models
# Other free tiers with the same (OpenAI-style) API. Each one added = more requests per minute/day for free.
# provider -> (base url, key in .env, preferred models; the rest are discovered from the provider's list)
COMPAT = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", GROQ_PREFERRED),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                ("mistral-small-latest", "mistral-medium-latest", "mistral-large-latest")),
    "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1", "CLOUDFLARE_API_KEY",
                   ("@cf/openai/gpt-oss-120b", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/qwen/qwen3-30b-a3b-fp8")),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
               ("openai/gpt-oss-120b", "meta/llama-3.3-70b-instruct", "qwen/qwen3-next-80b-a3b-instruct",
                "deepseek-ai/deepseek-v3.1")),
    # only free models (50/day); "openrouter/free" lets OpenRouter route to whichever free model is up
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", ("openrouter/free",)),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", ("gpt-oss-120b", "llama-3.3-70b", "qwen-3-32b")),
}


def _base(provider: str) -> str:
    return COMPAT[provider][0].replace("{CLOUDFLARE_ACCOUNT_ID}", os.getenv("CLOUDFLARE_ACCOUNT_ID", ""))


def _configured(provider: str) -> bool:
    return bool(os.getenv(COMPAT[provider][1])) and (provider != "cloudflare" or bool(os.getenv("CLOUDFLARE_ACCOUNT_ID")))
_MODEL_CACHE: dict[str, tuple[float, list[str]]] = {}
_COOLDOWN: dict[str, float] = {}  # model -> time until which we skip it (it was busy / timed out)
_LAST_GOOD: list[str] = []        # the model that answered last time is tried first
_SLOW_COUNT: dict[str, int] = {}  # model -> timeouts in a row
COOLDOWN_SECONDS = 180   # timed out (busy right now, or the prompt was big): try others for a few minutes
GONE_SECONDS = 30 * 86400  # 410: the provider retired the model
UNPAID_SECONDS = 1800   # 402 (no active plan on the key): ask again after this
BUSY_SECONDS = 60        # 429 per-minute limit / 503 overloaded: usually fine again within a minute
SLOW_MAX = 3600         # a model that keeps timing out rests longer each time (3, 6, 12... min), up to an hour
HEDGE_MIN = 4           # a model that hasn't answered after this long (or 40% of its timeout): ask the next one too,
HEDGE_SHARE = 0.4       # and take whichever answers first, so one slow model can't use up the whole budget
MAX_PARALLEL = 3        # at most this many models asked at once for one call
STRONG_WAIT = 15         # the judge waits up to this for a strong model to free up before using a weaker one (planning: 3 s)
STATE_FILE = Path(__file__).resolve().parent.parent / "debug" / "llm_state.json"


def _load_state() -> None:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    now = time.time()
    _COOLDOWN.update({m: t for m, t in data.get("cooldown", {}).items() if isinstance(t, (int, float)) and t > now})
    if isinstance(data.get("last_good"), str):
        _LAST_GOOD[:] = [data["last_good"]]


def _save_state() -> None:
    now = time.time()
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"cooldown": {m: t for m, t in _COOLDOWN.items() if t > now},
                                          "last_good": last_model() if _LAST_GOOD else None}), encoding="utf-8")
    except OSError:
        pass


# ---------- daily budget ----------
# Free requests per day (RPD) of each model, from the providers' free-tier pages (2026-11). The first matching
# (provider, part of the model name) wins; a model not listed has no known daily count (it rests when it says so).
# "*" counts every model of that provider together (OpenRouter: 50 free requests a day per account).
DAILY_LIMITS = [
    ("gemini", "gemma", 14400), ("gemini", "flash-lite", 500), ("gemini", "flash", 20),
    ("groq", "llama-3.1-8b", 14400), ("groq", "", 1000),
    ("openrouter", "*", 50),
]
BACKGROUND_SHARE = float(os.getenv("LLM_BACKGROUND_SHARE", "0.6"))  # the rest of each daily quota is kept for shoppers
PERSIST = True        # save the day's counts in the database (off in tests)
FLUSH_EVERY = 30      # seconds between saves
_DAY: dict[tuple[str, str, str], list[int]] = {}    # (day, model, step) -> [calls, tokens], today's (and the db's)
_UNSAVED: dict[tuple[str, str, str], list[int]] = {}
_PROVIDER: dict[str, str] = {}  # model -> provider, for the limits and /admin
_loaded_days: set[str] = set()
_last_flush = [0.0]


def quota_day(model: str = "") -> str:
    """The free-quota day a call counts on: it starts at midnight US Pacific (08:00 UTC), Cloudflare's at 00:00 UTC."""
    shift = 0 if model.startswith("@cf/") else 8
    return (datetime.now(timezone.utc) - timedelta(hours=shift)).strftime("%Y-%m-%d")


def _limit_entry(provider: str, model: str) -> tuple[str, str, int] | None:
    """The DAILY_LIMITS line this model counts on (the first that matches), or None."""
    name = model.lower()
    for prov, part, rpd in DAILY_LIMITS:
        if prov == provider and (part == "*" or part in name):
            return prov, part, rpd
    return None


def daily_limit(provider: str, model: str) -> tuple[str, int] | None:
    """(what is counted: the model or 'provider:*', its free requests a day), or None when not known."""
    if not (entry := _limit_entry(provider, model)):
        return None
    return (f"{provider}:*" if entry[1] == "*" else model), entry[2]


def _load_day(day: str) -> None:
    """Today's counts saved before a restart (Render's free service sleeps and restarts often)."""
    if day in _loaded_days or not PERSIST:
        return
    _loaded_days.add(day)
    try:
        from . import db
        for row in db.llm_usage_day(day):
            key = (day, row["model"], row["step"])
            got = _DAY.setdefault(key, [0, 0])
            got[0] += row["calls"]
            got[1] += row["tokens"]
            if row.get("provider"):
                _PROVIDER[row["model"]] = row["provider"]
    except Exception as e:  # database down: count from zero, the providers' own 429s still stop us
        log.warning("llm usage: could not load %s: %s", day, e)


def calls_today(provider: str, model: str) -> tuple[int, int | None]:
    """(requests used today, free requests a day or None) for what this model's daily limit counts."""
    day = quota_day(model)
    _load_day(day)
    lim = daily_limit(provider, model)
    if lim and lim[0].endswith(":*"):
        used = sum(v[0] for (d, m, _), v in _DAY.items() if d == day and _PROVIDER.get(m) == provider)
    else:
        used = sum(v[0] for (d, m, _), v in _DAY.items() if d == day and m == model)
    return used, (lim[1] if lim else None)


def over_budget(provider: str, model: str, background: bool) -> bool:
    """True when this call should skip the model: its free daily requests are used up, or (background work) the
    share kept for shoppers is all that's left."""
    used, rpd = calls_today(provider, model)
    if rpd is None:
        return False
    return used >= (rpd * BACKGROUND_SHARE if background else rpd)


def _tally(provider: str, model: str, step: str, tokens: int) -> None:
    day = quota_day(model)
    _load_day(day)
    _PROVIDER[model] = provider
    for book in (_DAY, _UNSAVED):
        got = book.setdefault((day, model, step), [0, 0])
        got[0] += 1
        got[1] += tokens
    if PERSIST and time.time() - _last_flush[0] >= FLUSH_EVERY:
        _last_flush[0] = time.time()
        threading.Thread(target=flush_usage, daemon=True).start()


def flush_usage() -> None:
    """Adds the calls counted since the last save to the database."""
    if not PERSIST or not _UNSAVED:
        return
    rows = [(d, m, s, _PROVIDER.get(m, ""), v[0], v[1]) for (d, m, s), v in list(_UNSAVED.items())]
    _UNSAVED.clear()
    try:
        from . import db
        db.add_llm_usage(rows)
    except Exception as e:  # kept in memory: counted again on the next save
        log.warning("llm usage: could not save: %s", e)
        for d, m, s, _, c, t in rows:
            got = _UNSAVED.setdefault((d, m, s), [0, 0])
            got[0] += c
            got[1] += t


def today_report() -> list[dict]:
    """Today's use per model, for the owner's /admin page: calls, tokens, by step, free daily limit, resting."""
    now, out = time.time(), {}
    days = {quota_day(), quota_day("@cf/")}
    for d in days:
        _load_day(d)
    for (d, m, step), (c, t) in _DAY.items():
        if d != quota_day(m):
            continue
        row = out.setdefault(m, {"model": m, "provider": _PROVIDER.get(m, ""), "calls": 0, "tokens": 0, "steps": {}})
        row["calls"] += c
        row["tokens"] += t
        row["steps"][step] = row["steps"].get(step, 0) + c
    for row in out.values():
        used, rpd = calls_today(row["provider"], row["model"])
        row["limit"] = rpd
        row["used_of_limit"] = used
        row["resting_until"] = _COOLDOWN.get(row["model"]) if _COOLDOWN.get(row["model"], 0) > now else None
    return sorted(out.values(), key=lambda r: -r["calls"])


def quota_report() -> list[dict]:
    """Each free daily limit, for the owner's /admin page: what it counts, used today of its limit, left, when it
    renews, and whether we skip the model now. Known limits of a configured provider show up even unused; a model
    with no known daily limit shows up only once it was used today."""
    now, rows = time.time(), {}
    for r in today_report():
        lim = daily_limit(r["provider"], r["model"])
        key = lim[0] if lim else r["model"]
        row = rows.setdefault(key, {"name": key, "provider": r["provider"], "used": 0, "limit": lim[1] if lim else None,
                                    "resting_until": None})
        row["used"] = r["used_of_limit"] if lim else row["used"] + r["calls"]
        if r["resting_until"] and (row["resting_until"] or 0) < r["resting_until"]:
            row["resting_until"] = r["resting_until"]
    for provider, part, rpd in DAILY_LIMITS:
        configured = bool(os.getenv("GEMINI_API_KEY")) if provider == "gemini" else _configured(provider)
        if not configured:
            continue
        entry = (provider, part, rpd)
        if not any(r["provider"] == provider and _limit_entry(provider, r["name"]) == entry for r in rows.values()):  # nothing used yet: still show the limit, named by what it covers
            name = f"{provider}:*" if part == "*" else f"{provider} {part or 'other models'}"
            rows[name] = {"name": name, "provider": provider, "used": 0, "limit": rpd, "resting_until": None}
    for row in rows.values():
        row["left"] = None if row["limit"] is None else max(0, row["limit"] - row["used"])
        row["resets_at"] = _daily_quota_reset(row["name"].startswith("@cf/"))
        if row["resting_until"] and row["resting_until"] <= now:
            row["resting_until"] = None
    return sorted(rows.values(), key=lambda r: (r["limit"] is None, -(r["used"] / r["limit"] if r["limit"] else 0),
                                                -r["used"], r["name"]))


def _daily_quota_reset(utc_midnight: bool = False) -> float:
    """Free daily quotas reset at midnight US Pacific time (08:00 UTC; no time-zone database needed on Windows);
    Cloudflare's at midnight UTC."""
    now = datetime.now(timezone.utc)
    reset = now.replace(hour=0 if utc_midnight else 8, minute=1, second=0, microsecond=0)
    return (reset if reset > now else reset + timedelta(days=1)).timestamp()


def retry_hint(r: httpx.Response) -> float | None:
    """Seconds the provider says to wait before the next request (per-minute limits often free up in a few
    seconds): Retry-After header, Groq/OpenAI 'try again in 1m2.5s / 450ms', Gemini 'retryDelay'."""
    try:
        if (h := r.headers.get("retry-after")) and float(h) >= 0:
            return float(h)
    except ValueError:
        pass
    text = r.text
    if m := re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)(ms|s)\b", text):
        secs = float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
        return secs + 60 * float(m.group(1) or 0)
    if m := re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', text):
        return float(m.group(1))
    return None


def _rest(model: str, r: httpx.Response | None = None, slow: bool = False) -> None:
    """Skip a model for a while: until tomorrow if its daily quota is gone, else a few minutes (or as long as the
    provider says, when it says). slow=True: it timed out with its full time; each time in a row it rests twice
    as long, so a model that is always too slow on the free tier stops taking time from every request."""
    body = r.text.lower() if r is not None else ""
    daily = r is not None and r.status_code == 429 and any(x in body for x in (
        "perday", "per day", "per-day", "(rpd)", "(tpd)",
        "daily free allocation", "daily limit", "daily quota"))  # Cloudflare: "used up your daily free allocation"
    busy = r is not None and r.status_code in (429, 503)
    hint = retry_hint(r) if busy and not daily else None
    refused = r is not None and r.status_code in (401, 403, 404)  # not available to us: skip it until tomorrow
    # 402: the key has no active plan yet (e.g. Mistral before its free Experiment plan is turned on). The owner may
    # turn it on any minute, so it is asked again after half an hour, not tomorrow
    unpaid = r is not None and r.status_code == 402
    if r is not None and r.status_code == 410:  # the provider retired this model: don't try it again for a month
        _COOLDOWN[model] = time.time() + GONE_SECONDS
        log.warning("LLM %s was retired by its provider, using other models", model)
    else:
        _COOLDOWN[model] = (time.time() + UNPAID_SECONDS if unpaid
                            else _daily_quota_reset(utc_midnight=model.startswith("@cf/")) if daily or refused
                            else time.time() + min(BUSY_SECONDS, hint + 0.5) if hint is not None
                            else time.time() + (BUSY_SECONDS if busy else COOLDOWN_SECONDS))
    if slow:
        _SLOW_COUNT[model] = n = _SLOW_COUNT.get(model, 0) + 1
        _COOLDOWN[model] = time.time() + min(SLOW_MAX, COOLDOWN_SECONDS * 2 ** (n - 1))
    if daily:
        log.warning("LLM %s: daily free quota used up, skipping it until it resets", model)
        problems.note("llm_quota", f"{model}: daily free quota used up (429)")
    _save_state()


_load_state()


async def _list_models(client: httpx.AsyncClient, provider: str) -> list[str]:
    """Ask the provider which models exist right now (names change often), cached for 6 hours."""
    if (hit := _MODEL_CACHE.get(provider)) and time.time() - hit[0] < 6 * 3600:
        return hit[1]
    names: list[str] = []
    try:
        if provider == "gemini":
            r = await client.get("https://generativelanguage.googleapis.com/v1beta/models",
                                 headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, params={"pageSize": 200})
            names = [m["name"].removeprefix("models/") for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
        elif provider == "cloudflare":  # no OpenAI-style model list there: our preferred names are used
            return []
        else:
            key = COMPAT[provider][1]
            r = await client.get(_base(provider) + "/models", headers={"Authorization": f"Bearer {os.environ[key]}"})
            names = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
            if provider == "openrouter":
                names = [n for n in names if n.endswith(":free") or n == "openrouter/free"]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("could not list %s models: %s", provider, e)
    if names:
        _MODEL_CACHE[provider] = (time.time(), names)
    return names


def _size(model: str) -> float:
    """Parameter count in billions from the name (e.g. 'llama-3.3-70b' -> 70), 0 if not written."""
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b\b", model.lower())]
    return max(sizes) if sizes else 0


CHAT_FAMILIES = ("gpt-oss", "deepseek", "llama", "kimi", "glm-4", "qwen", "mistral", "mixtral", "gemma", "nemotron")


def _big_model(model: str) -> bool:
    """A big general chat model, from a provider's long list. Lists also hold special-purpose models with big sizes
    in their names (live test 2026-11-24: NVIDIA's 'ising-calibration-1.5-31b' was picked to read prices), so the
    name must also belong to a known chat model family or say it follows instructions."""
    name = model.lower()
    if any(x in name for x in ("vision", "-vl", "coder", "embed", "guard", "audio", "image", "reasoning", "thinking",
                               "omni", "nano", "reward", "creative", "calibrat", "retriev", "rerank", "safety",
                               "parse", "translate", "math")):  # slow "thinking" models time out on our 10-20 s budget
        return False
    general = any(x in name for x in CHAT_FAMILIES) or "instruct" in name or "chat" in name
    return general and (_size(name) >= 24 or any(x in name for x in ("gpt-oss", "deepseek", "llama-4", "kimi", "glm-4",
                                                                      "qwen3-235")))


def gemma(model: str) -> bool:
    return model.lower().startswith("gemma")


def _gemma_models(available: list[str]) -> list[str]:
    """The biggest Gemma models the Gemini key offers (names change with each Gemma release)."""
    big = [m for m in available if gemma(m) and m.endswith("-it") and _size(m) >= GEMMA_MIN_SIZE and not _gone(m)]
    return sorted(big, key=lambda m: (_size(m), m), reverse=True)[:2]


def strong(model: str, provider: str = "") -> bool:
    """Models good enough to plan and judge (they decide what the shopper sees). Page reading only copies names
    and prices, so it goes to the smaller ones and leaves the strong models' free quota for the judge."""
    name = model.lower()
    return ((provider == "gemini" and not gemma(name)) or "gpt-oss-120b" in name or _size(name) >= 100
            or any(x in name for x in ("deepseek", "kimi")))


def _pick(available: list[str], preferred: tuple[str, ...], custom: str | None, provider: str,
          limit: int = 6) -> list[str]:
    order = ([custom] if custom else []) + list(preferred)
    if not available:  # listing failed: just try our guesses
        return list(dict.fromkeys(order))
    picked = [m for m in order if m in available]
    if provider == "gemini":  # newest flash models first, skip special-purpose ones
        extra = sorted((m for m in available if "flash" in m and not any(
            x in m for x in ("image", "tts", "audio", "live", "embedding", "thinking", "exp", "preview"))), reverse=True)
    elif provider == "nvidia":  # models get retired often (410): fall back to the other big models it lists
        extra = sorted((m for m in available if _big_model(m)), key=_size, reverse=True)
    elif provider in ("mistral", "cloudflare", "cerebras"):  # big catalogs: only our known-good picks
        extra = []
        if not picked:
            picked = list(order)
    elif provider == "openrouter":  # dozens of free models, many tiny: only big general ones
        extra = sorted((m for m in available if _big_model(m)), key=_size, reverse=True)
        if "openrouter/free" not in available:
            picked = ["openrouter/free"] + picked
    else:
        extra = [m for m in available if not any(x in m for x in ("whisper", "guard", "tts", "prompt", "compound", "orpheus",
                                                                  "allam", "gemma", "distil", "safeguard"))]
    return [m for m in dict.fromkeys(picked + extra) if not _gone(m)][:limit]


def _gone(model: str) -> bool:
    """Retired by its provider (410): skipped when choosing models, so live ones take its place."""
    return _COOLDOWN.get(model, 0) > time.time() + 2 * 86400


async def _gemini(client: httpx.AsyncClient, model: str, prompt: str) -> httpx.Response:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    if gemma(model):  # no JSON mode and no thinking setting for Gemma: the prompt asks for JSON
        body = {"contents": [{"parts": [{"text": prompt + "\nReply with JSON only."}]}],
                "generationConfig": {"temperature": _TEMPERATURE.get()}}
        return await client.post(url, headers=headers, json=body)
    config = {"responseMimeType": "application/json", "temperature": _TEMPERATURE.get(),
              "thinkingConfig": {"thinkingBudget": 0}}  # no "thinking": much faster, enough for this task
    r = await client.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config})
    if r.status_code == 400:  # e.g. a model that can't turn thinking off: retry with defaults
        config.pop("thinkingConfig")
        r = await client.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config})
    return r


async def _compat(provider: str, client: httpx.AsyncClient, model: str, prompt: str, on_text=None) -> httpx.Response:
    url = _base(provider) + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ[COMPAT[provider][1]]}"}
    body = {"model": model, "temperature": _TEMPERATURE.get(),
            "messages": [{"role": "user", "content": prompt + "\nReply with JSON only."}]}
    if provider != "openrouter":  # not every free OpenRouter model accepts it; the prompt asks for JSON anyway
        body["response_format"] = {"type": "json_object"}
    if provider == "cloudflare":  # its default is 256 tokens: a reasoning model (gpt-oss) uses them up thinking and
        body["max_tokens"] = 4096  # returns no answer at all
    if on_text:
        r = await _stream(client, url, headers, body, on_text)
        if r.status_code not in (400, 422):
            return r
        # this model can't stream (or not with these settings): one normal request instead
    r = await client.post(url, headers=headers, json=body)
    if r.status_code in (400, 422) and "response_format" in body:  # model without JSON mode: ask in the prompt only
        body.pop("response_format")
        r = await client.post(url, headers=headers, json=body)
    return r


async def _stream(client: httpx.AsyncClient, url: str, headers: dict, body: dict, on_text) -> httpx.Response:
    """The same request, with the answer sent piece by piece (server-sent events): on_text(text so far) is called
    as it grows, so the page can show the reply while it's written. JSON mode is left out because some providers
    (Groq) can't stream with it; the prompt asks for JSON anyway. Returns a normal-looking response with the whole
    answer, so the caller doesn't care that it streamed."""
    body = {k: v for k, v in body.items() if k != "response_format"}
    body.update(stream=True, stream_options={"include_usage": True})
    text, usage = "", None
    async with client.stream("POST", url, headers=headers, json=body) as r:
        if r.status_code != 200:
            return httpx.Response(r.status_code, headers=r.headers, content=await r.aread())
        async for line in r.aiter_lines():
            if not line.startswith("data:") or (data := line[5:].strip()) == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            usage = event.get("usage") or (event.get("x_groq") or {}).get("usage") or usage
            for choice in event.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    text += piece
                    on_text(text)
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": usage or {}})


async def _groq(client: httpx.AsyncClient, model: str, prompt: str, on_text=None) -> httpx.Response:
    return await _compat("groq", client, model, prompt, on_text)


def _caller(provider: str):
    if provider == "groq":
        return _groq

    async def call(client, model, prompt, on_text=None):
        return await _compat(provider, client, model, prompt, on_text)
    return call


def _json(text: str):
    """The reply as JSON, also when a model wraps it in ```json fences or adds a sentence around it."""
    try:
        return json.loads(text)
    except ValueError:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise
        return json.loads(text[a:b + 1])


def _text(provider: str, r: httpx.Response) -> str:
    data = r.json()
    if provider == "gemini":
        return data["candidates"][0]["content"]["parts"][0]["text"]
    return data["choices"][0]["message"]["content"]


async def _attempts(client: httpx.AsyncClient) -> list[tuple]:
    """Every (provider, model, caller) this server could ask, in no particular order."""
    attempts = []
    if os.getenv("GEMINI_API_KEY"):
        listed = await _list_models(client, "gemini")
        models = _pick(listed, GEMINI_PREFERRED, os.getenv("GEMINI_MODEL"), "gemini")
        attempts += [("gemini", m, _gemini) for m in models[:3] + _gemma_models(listed)]
    for provider, (_, key, preferred) in COMPAT.items():
        if _configured(provider):
            models = _pick(await _list_models(client, provider), preferred,
                           os.getenv(f"{provider.upper()}_MODEL"), provider)
            attempts += [(provider, m, _caller(provider)) for m in models]
    return attempts


async def free_in() -> float | None:
    """Seconds until some model can answer a shopper again (0 = one can now); None = none today (every model is at
    its free daily limit, retired, or there is no key). No model is asked: the cooldowns and daily counts we keep."""
    if not available():
        return None
    async with httpx.AsyncClient(timeout=8) as client:
        listed = await _attempts(client)
    if not listed:  # the providers' model lists couldn't be read: unknown, so the normal search tries
        return 0.0
    attempts = [a for a in listed if not over_budget(a[0], a[1], False) and not _gone(a[1])]
    if not attempts:
        return None
    now = time.time()
    return max(0.0, min(_COOLDOWN.get(a[1], 0) - now for a in attempts))


async def ask_json(prompt: str, timeout: float = 10, prefer: str | None = None, spread: int = 0,
                   budget: float | None = None, step: str = "other", on_text=None, temperature: float = 0) -> dict:
    """prefer="bulk": try the fast non-Gemini providers' smaller models first (bulk work like reading pages, so the
    strong models' daily quota is kept for understanding the request and judging the results);
    prefer="strong": the strongest models first (the judge), never just whichever model answered last;
    prefer="groq" etc.: that provider.
    spread=n: start from the n-th model of that provider, so calls running at the same time go to different
    models (each model has its own per-minute limit).
    on_text(text so far): called while the answer arrives (OpenAI-style providers stream it; Gemini answers at once).
    temperature: 0 (default) = the same answer every time, for planning searches and judging; a bit more for talk."""
    if not available():
        raise LLMError(details="no LLM key in .env")
    if _BACKGROUND.get():
        waited = 0.0
        while _LIVE > 0 and waited < BACKGROUND_WAIT:
            await asyncio.sleep(1)
            waited += 1
    token = _TEMPERATURE.set(temperature)
    try:
        return await _ask_json(prompt, timeout, prefer, spread, budget, step, on_text)
    finally:
        _TEMPERATURE.reset(token)


async def _ask_json(prompt: str, timeout: float, prefer: str | None, spread: int, budget: float | None, step: str,
                    on_text) -> dict:
    errors, start = [], time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        # a model whose free daily requests are used up (or, for background work, all but the shoppers' share) is
        # not asked: a 429 would only cost time, and background work must leave the rest of the day to shoppers
        behind = _BACKGROUND.get()
        attempts = [a for a in await _attempts(client) if not over_budget(a[0], a[1], behind)]
        if not attempts:
            _failed(step, "every model", "daily budget used")
            problems.note("llm_budget", f"step {step}: every model is at its daily "
                          + ("background share" if behind else "free limit"))
            raise LLMError(BUSY_TEXT, details="daily budget used" + (" (background share)" if behind else ""))
        # last model that worked goes first; models that just failed go last (or are skipped)
        now = time.time()
        bulk = prefer == "bulk"

        def tier(a) -> int:
            """Bulk work (reading pages, light steps): the fast providers' smaller models (the biggest daily
            quotas), then the smaller models elsewhere, then Gemini (fast, but a small daily quota), then the
            fast strong models, then the rest. The strong models' daily quota is kept for planning and judging,
            which decide what the shopper sees."""
            small = not strong(a[1], a[0])
            if gemma(a[1]):  # the biggest daily quota of all, but slower: first for background work
                return 0 if behind else 1
            if small:
                return 0 if a[0] in FAST_BULK else 1
            return 2 if a[0] == "gemini" else 3 if a[0] in FAST_BULK else 4

        def preferred(a) -> bool:
            if bulk:
                return tier(a) == 0
            if prefer == "strong":
                return strong(a[1], a[0])
            return bool(prefer) and a[0] == prefer
        attempts.sort(key=lambda a: (_COOLDOWN.get(a[1], 0) > now, tier(a) if bulk else 0,
                                     bulk and behind and not gemma(a[1]),
                                     not bulk and gemma(a[1]),  # planning/talk: Gemma only when nothing else answers
                                     not preferred(a),
                                     prefer == "strong" and "gpt-oss-120b" not in a[1],  # fast + strong judge
                                     a[1] not in _LAST_GOOD))
        ready = [a for a in attempts if _COOLDOWN.get(a[1], 0) <= now]
        if not ready:
            # every model is resting: only the ones free again within this call's time are worth waiting for. A model
            # retired (410) or out of its daily quota can't answer now; asking it again only adds another failure
            # (live 2026-11-25: one search asked a retired model 8 times while the shopper waited)
            ready = [a for a in attempts if _COOLDOWN.get(a[1], 0) - now < (budget or TOTAL_BUDGET)]
            if not ready:
                _failed(step, "every model", "resting")
                problems.note("llm_busy", f"step {step}: every model is resting")
                raise LLMError(BUSY_TEXT, details="every model is resting")
        if spread and prefer:
            mine = [a for a in ready if preferred(a)]
            if mine:
                k = spread % len(mine)
                ready = mine[k:] + mine[:k] + [a for a in ready if a not in mine]

        queue, tries, busy = list(ready), {}, set()
        budget_s = budget or TOTAL_BUDGET
        hedge = max(HEDGE_MIN, timeout * HEDGE_SHARE)
        leader: list = [None]  # with two models answering at once, only one of them types the live reply

        def left_now() -> float:
            return budget_s - (time.monotonic() - start)

        async def choose(idle: bool):
            """The next model to try, or None. Waits for a resting model only when nothing else is running."""
            left = left_now()
            if queue:
                a = queue.pop(0)
                if prefer == "strong" and not preferred(a):
                    # the judge: a strong model that is only resting a few seconds (per-minute limit) beats a weak
                    # model right now, which lets wrong or no-name items through
                    now = time.time()
                    soon = [x for x in attempts if preferred(x) and tries.get(x[1], 0) < 2
                            and 0 < _COOLDOWN.get(x[1], 0) - now < min(STRONG_WAIT if step == "judge" else 3, left - 5)]
                    if soon:
                        queue.insert(0, a)
                        if not idle:
                            return None
                        a = min(soon, key=lambda x: _COOLDOWN.get(x[1], 0))
                        wait = _COOLDOWN.get(a[1], 0) - now
                        log.info("LLM: waiting %.1fs for strong model %s instead of a weaker one", wait, a[1])
                        await asyncio.sleep(wait)
                return a
            if not idle:
                return None
            # Every model is resting. Per-minute limits free up within seconds, so wait for the one that is
            # free soonest (if it is within our time) instead of giving up while there's time left.
            now = time.time()
            soon = [x for x in attempts if (x[1] not in tries or (x[1] in busy and tries[x[1]] < 2))
                    and _COOLDOWN.get(x[1], 0) - now < left - 3]
            if not soon:
                return None
            a = min(soon, key=lambda x: (_COOLDOWN.get(x[1], 0), not preferred(x)))
            wait = _COOLDOWN.get(a[1], 0) - now
            if wait > 0:
                log.info("LLM: all models busy, waiting %.1fs for %s", wait, a[1])
                await asyncio.sleep(wait)
            return a

        async def attempt(provider, model, call):
            """One model, one request. Never raises (except when cancelled): the result says what happened."""
            def typed(text: str) -> None:
                if leader[0] in (None, model):
                    leader[0] = model
                    on_text(text)
            limit = max(1.0, min(timeout, left_now()))
            t = time.monotonic()
            try:
                r = await asyncio.wait_for(call(client, model, prompt, typed) if on_text and provider != "gemini"
                                           else call(client, model, prompt), timeout=limit)
                return provider, model, r, None, time.monotonic() - t, limit
            except Exception as e:  # timeout, network, a provider's odd reply: just try another model
                return provider, model, None, e, time.monotonic() - t, limit

        running: dict[asyncio.Task, tuple[str, float]] = {}
        last_launch, exhausted = -1e9, False
        try:
            while True:
                left = left_now()
                if left <= 0.5 or (not running and (left < 2 or exhausted)):
                    break
                can_launch = (not running or (len(running) < MAX_PARALLEL and not exhausted
                                              and time.monotonic() - last_launch >= hedge)) and left >= 2
                if can_launch:
                    nxt = await choose(idle=not running)
                    if nxt is not None:
                        provider, model, call = nxt
                        tries[model] = tries.get(model, 0) + 1
                        if running:
                            log.info("LLM: %s is slow, also asking %s", ", ".join(m for m, _ in running.values()), model)
                        last_launch = time.monotonic()
                        running[asyncio.ensure_future(attempt(provider, model, call))] = (model, last_launch)
                        continue
                    if not running:
                        break
                    exhausted = not queue  # nothing more to start for now: just wait for the ones running
                wait = left if len(running) >= MAX_PARALLEL or exhausted else max(0.05, last_launch + hedge - time.monotonic())
                done, _ = await asyncio.wait(list(running), timeout=min(wait, left_now()),
                                             return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    running.pop(task)
                    provider, model, r, err, took, limit = task.result()
                    if err is not None:
                        if leader[0] == model:
                            leader[0] = None
                        errors.append(f"{model}: {type(err).__name__}")
                        _failed(step, model, type(err).__name__)
                        timed_out = isinstance(err, (asyncio.TimeoutError, httpx.TimeoutException))
                        log.warning("LLM %s failed after %.1fs: %s", model, took, type(err).__name__)
                        _rest(model, slow=timed_out and limit >= timeout - 0.5)
                        exhausted, last_launch = False, -1e9  # failed fast: ask the next one right away
                        continue
                    log.info("LLM %s -> %s in %.1fs", model, r.status_code, took)
                    if r.status_code == 200:
                        _count(provider, r, step, model)
                        try:
                            data = _json(_text(provider, r))
                        except (KeyError, IndexError, ValueError, TypeError):  # TypeError: no text in the reply (cut off)
                            errors.append(f"{model}: bad reply")
                            _failed(step, model, "bad reply")
                            exhausted, last_launch = False, -1e9
                            continue
                        if not isinstance(data, dict):  # e.g. a bare list: callers expect an object
                            data = {"products": data} if isinstance(data, list) else {}
                        _SLOW_COUNT.pop(model, None)
                        if _LAST_GOOD != [model] and not bulk:  # page reads don't decide who judges next
                            _LAST_GOOD[:] = [model]
                            _save_state()
                        return data
                    errors.append(f"{model}: {r.status_code}")
                    _failed(step, model, str(r.status_code))
                    log.warning("LLM %s said: %s", model, r.text[:200].replace("\n", " "))
                    if r.status_code in RETRY_STATUS or r.status_code in (401, 402, 403, 404, 410):  # busy/quota/not allowed/gone
                        _rest(model, r)
                        if r.status_code in (429, 503):
                            busy.add(model)
                    exhausted, last_launch = False, -1e9
        finally:
            for task, (model, since) in running.items():
                task.cancel()
                errors.append(f"{model}: out of time")
                _failed(step, model, "out of time")
                if time.monotonic() - since >= hedge:  # it was slow too: let the others go first next time
                    _rest(model)
            if running:
                await asyncio.gather(*running, return_exceptions=True)  # let them stop before the client closes
    problems.note("llm_busy", f"step {step}: " + ", ".join(errors[-8:] or ["timeout"]))
    raise LLMError(BUSY_TEXT, details=", ".join(errors[-8:] or ["timeout"]))
