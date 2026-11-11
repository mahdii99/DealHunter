"""LLM client, free tiers only.

Tries, in order, until one answers:
  1. Google Gemini (GEMINI_API_KEY, https://aistudio.google.com/apikey), several models
  2. Groq (GROQ_API_KEY, https://console.groq.com/keys), optional backup
A model that is busy (429/5xx) or times out is skipped for 3 minutes; one whose free DAILY quota is used up is
skipped until the quota resets (midnight US Pacific time). The last model that worked is tried first.
Both are saved to debug/llm_state.json, so a server restart (--reload) doesn't retry dead models again.
"""
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}
log = logging.getLogger("uvicorn.error")
TOTAL_BUDGET = float(os.getenv("LLM_BUDGET_SECONDS", "25"))  # give up and use the simple fallback after this


class LLMError(Exception):
    pass


USAGE = {"calls": 0, "tokens": 0, "by_step": {}}  # LLM use since the last reset (the agent resets it per search)


def reset_usage() -> None:
    USAGE.update(calls=0, tokens=0, by_step={})


def _count(provider: str, r: httpx.Response, step: str, model: str = "") -> None:
    try:
        data = r.json()
        u = data.get("usageMetadata", {}).get("totalTokenCount") if provider == "gemini" else             (data.get("usage") or {}).get("total_tokens")
    except ValueError:
        u = None
    USAGE["calls"] += 1
    USAGE["tokens"] += int(u or 0)
    s = USAGE["by_step"].setdefault(step, {"calls": 0, "tokens": 0})
    s["calls"] += 1
    s["tokens"] += int(u or 0)
    if model:
        s["model"] = model  # which model answered this step last (e.g. who judged)


def last_model() -> str:
    return _LAST_GOOD[0] if _LAST_GOOD else "?"


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or any(_configured(p) for p in COMPAT))


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
COOLDOWN_SECONDS = 180   # timed out (busy right now, or the prompt was big): try others for a few minutes
GONE_SECONDS = 30 * 86400  # 410: the provider retired the model
BUSY_SECONDS = 60        # 429 per-minute limit / 503 overloaded: usually fine again within a minute
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


def _rest(model: str, r: httpx.Response | None = None) -> None:
    """Skip a model for a while: until tomorrow if its daily quota is gone, else a few minutes (or as long as the
    provider says, when it says)."""
    body = r.text.lower() if r is not None else ""
    daily = r is not None and r.status_code == 429 and any(x in body for x in (
        "perday", "per day", "per-day", "(rpd)", "(tpd)",
        "daily free allocation", "daily limit", "daily quota"))  # Cloudflare: "used up your daily free allocation"
    busy = r is not None and r.status_code in (429, 503)
    hint = retry_hint(r) if busy and not daily else None
    refused = r is not None and r.status_code in (401, 403, 404)  # not available to us: skip it until tomorrow
    if r is not None and r.status_code == 410:  # the provider retired this model: don't try it again for a month
        _COOLDOWN[model] = time.time() + GONE_SECONDS
        log.warning("LLM %s was retired by its provider, using other models", model)
    else:
        _COOLDOWN[model] = (_daily_quota_reset(utc_midnight=model.startswith("@cf/")) if daily or refused
                            else time.time() + min(BUSY_SECONDS, hint + 0.5) if hint is not None
                            else time.time() + (BUSY_SECONDS if busy else COOLDOWN_SECONDS))
    if daily:
        log.warning("LLM %s: daily free quota used up, skipping it until it resets", model)
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


def _big_model(model: str) -> bool:
    name = model.lower()
    if any(x in name for x in ("vision", "-vl", "coder", "embed", "guard", "audio", "image", "reasoning", "thinking",
                               "omni", "nano", "reward", "creative")):  # slow "thinking" models time out on our 10-20 s budget
        return False
    return _size(name) >= 24 or any(x in name for x in ("gpt-oss", "deepseek", "llama-4", "kimi", "glm-4", "qwen3-235"))


def strong(model: str, provider: str = "") -> bool:
    """Models good enough to plan and judge (they decide what the shopper sees). Page reading only copies names
    and prices, so it goes to the smaller ones and leaves the strong models' free quota for the judge."""
    name = model.lower()
    return (provider == "gemini" or "gpt-oss-120b" in name or _size(name) >= 100
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
    config = {"responseMimeType": "application/json", "temperature": 0,
              "thinkingConfig": {"thinkingBudget": 0}}  # no "thinking": much faster, enough for this task
    r = await client.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config})
    if r.status_code == 400:  # e.g. a model that can't turn thinking off: retry with defaults
        config.pop("thinkingConfig")
        r = await client.post(url, headers=headers, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config})
    return r


async def _compat(provider: str, client: httpx.AsyncClient, model: str, prompt: str) -> httpx.Response:
    url = _base(provider) + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ[COMPAT[provider][1]]}"}
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt + "\nReply with JSON only."}]}
    if provider != "openrouter":  # not every free OpenRouter model accepts it; the prompt asks for JSON anyway
        body["response_format"] = {"type": "json_object"}
    if provider == "cloudflare":  # its default is 256 tokens: a reasoning model (gpt-oss) uses them up thinking and
        body["max_tokens"] = 4096  # returns no answer at all
    r = await client.post(url, headers=headers, json=body)
    if r.status_code in (400, 422) and "response_format" in body:  # model without JSON mode: ask in the prompt only
        body.pop("response_format")
        r = await client.post(url, headers=headers, json=body)
    return r


async def _groq(client: httpx.AsyncClient, model: str, prompt: str) -> httpx.Response:
    return await _compat("groq", client, model, prompt)


def _caller(provider: str):
    if provider == "groq":
        return _groq

    async def call(client, model, prompt):
        return await _compat(provider, client, model, prompt)
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


async def ask_json(prompt: str, timeout: float = 10, prefer: str | None = None, spread: int = 0,
                   budget: float | None = None, step: str = "other") -> dict:
    """prefer="bulk": try the fast non-Gemini providers' smaller models first (bulk work like reading pages, so the
    strong models' daily quota is kept for understanding the request and judging the results);
    prefer="strong": the strongest models first (the judge), never just whichever model answered last;
    prefer="groq" etc.: that provider.
    spread=n: start from the n-th model of that provider, so calls running at the same time go to different
    models (each model has its own per-minute limit)."""
    if not available():
        raise LLMError("مفيش مفتاح LLM في ملف .env")
    errors, start = [], time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        attempts = []
        if os.getenv("GEMINI_API_KEY"):
            models = _pick(await _list_models(client, "gemini"), GEMINI_PREFERRED, os.getenv("GEMINI_MODEL"), "gemini")
            attempts += [("gemini", m, _gemini) for m in models[:3]]
        for provider, (_, key, preferred) in COMPAT.items():
            if _configured(provider):
                models = _pick(await _list_models(client, provider), preferred,
                               os.getenv(f"{provider.upper()}_MODEL"), provider)
                attempts += [(provider, m, _caller(provider)) for m in models]
        # last model that worked goes first; models that just failed go last (or are skipped)
        now = time.time()
        bulk = prefer == "bulk"

        def tier(a) -> int:
            """Bulk work (reading pages): the fast providers' smaller models, then Gemini (fast, big per-minute
            limits, its own daily quota), then the fast strong models (kept for planning and judging), then the
            rest (free models elsewhere often take 10 s+)."""
            if a[0] in FAST_BULK and not strong(a[1], a[0]):
                return 0
            return 1 if a[0] == "gemini" else 2 if a[0] in FAST_BULK else 3

        def preferred(a) -> bool:
            if bulk:
                return tier(a) == 0
            if prefer == "strong":
                return strong(a[1], a[0])
            return bool(prefer) and a[0] == prefer
        attempts.sort(key=lambda a: (_COOLDOWN.get(a[1], 0) > now, tier(a) if bulk else 0, not preferred(a),
                                     prefer == "strong" and "gpt-oss-120b" not in a[1],  # fast + strong judge
                                     a[1] not in _LAST_GOOD))
        ready = [a for a in attempts if _COOLDOWN.get(a[1], 0) <= now] or attempts
        if spread and prefer:
            mine = [a for a in ready if preferred(a)]
            if mine:
                k = spread % len(mine)
                ready = mine[k:] + mine[:k] + [a for a in ready if a not in mine]

        queue, tries, busy = list(ready), {}, set()
        while True:
            left = (budget or TOTAL_BUDGET) - (time.monotonic() - start)
            if left < 2:
                break
            if queue:
                provider, model, call = queue.pop(0)
                if prefer == "strong" and not preferred((provider, model, call)):
                    # the judge: a strong model that is only resting a few seconds (per-minute limit) beats a weak
                    # model right now, which lets wrong or no-name items through
                    now = time.time()
                    soon = [a for a in attempts if preferred(a) and tries.get(a[1], 0) < 2
                            and 0 < _COOLDOWN.get(a[1], 0) - now < min(STRONG_WAIT if step == "judge" else 3, left - 5)]
                    if soon:
                        queue.insert(0, (provider, model, call))
                        provider, model, call = min(soon, key=lambda a: _COOLDOWN.get(a[1], 0))
                        wait = _COOLDOWN.get(model, 0) - now
                        log.info("LLM: waiting %.1fs for strong model %s instead of a weaker one", wait, model)
                        await asyncio.sleep(wait)
                        left = (budget or TOTAL_BUDGET) - (time.monotonic() - start)
            else:
                # Every model is resting. Per-minute limits free up within seconds, so wait for the one that is
                # free soonest (if it is within our time) instead of giving up while there's time left.
                now = time.time()
                soon = [a for a in attempts if (a[1] not in tries or (a[1] in busy and tries[a[1]] < 2))
                        and _COOLDOWN.get(a[1], 0) - now < left - 3]
                if not soon:
                    break
                provider, model, call = min(soon, key=lambda a: (_COOLDOWN.get(a[1], 0), not preferred(a)))
                wait = _COOLDOWN.get(model, 0) - now
                if wait > 0:
                    log.info("LLM: all models busy, waiting %.1fs for %s", wait, model)
                    await asyncio.sleep(wait)
                left = (budget or TOTAL_BUDGET) - (time.monotonic() - start)
            tries[model] = tries.get(model, 0) + 1
            t = time.monotonic()
            try:
                r = await asyncio.wait_for(call(client, model, prompt), timeout=min(timeout, left))
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                errors.append(f"{model}: {type(e).__name__}")
                log.warning("LLM %s failed after %.1fs: %s (skipping it for %ds)",
                            model, time.monotonic() - t, type(e).__name__, COOLDOWN_SECONDS)
                _rest(model)
                continue
            log.info("LLM %s -> %s in %.1fs", model, r.status_code, time.monotonic() - t)
            if r.status_code == 200:
                _count(provider, r, step, model)
                try:
                    data = _json(_text(provider, r))
                except (KeyError, IndexError, ValueError, TypeError):  # TypeError: no text in the reply (cut off)
                    errors.append(f"{model}: bad reply")
                    continue
                if not isinstance(data, dict):  # e.g. a bare list: callers expect an object
                    data = {"products": data} if isinstance(data, list) else {}
                if _LAST_GOOD != [model] and not bulk:  # page reads don't decide who judges next
                    _LAST_GOOD[:] = [model]
                    _save_state()
                return data
            errors.append(f"{model}: {r.status_code}")
            log.warning("LLM %s said: %s", model, r.text[:200].replace("\n", " "))
            if r.status_code in RETRY_STATUS or r.status_code in (401, 403, 404, 410):  # busy/quota/not allowed/gone
                _rest(model, r)
                if r.status_code in (429, 503):
                    busy.add(model)
    raise LLMError("مفيش موديل رد في الوقت: " + ", ".join(errors[-4:] or ["timeout"]))
