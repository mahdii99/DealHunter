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


def last_model() -> str:
    return _LAST_GOOD[0] if _LAST_GOOD else "?"


def available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or any(_configured(p) for p in COMPAT))


# lite first: it answers in ~3s; the full flash model often times out on the free tier (mahdy's check, 2026-11-10)
GEMINI_PREFERRED = ("gemini-flash-lite-latest", "gemini-flash-latest")
GROQ_PREFERRED = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "qwen/qwen3-32b",
                  "meta-llama/llama-4-maverick-17b-128e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct",
                  "llama-3.1-8b-instant")
# Other free tiers with the same (OpenAI-style) API. Each one added = more requests per minute/day for free.
# provider -> (base url, key in .env, preferred models; the rest are discovered from the provider's list)
COMPAT = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", GROQ_PREFERRED),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY",
                ("mistral-small-latest", "mistral-medium-latest", "mistral-large-latest")),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
               ("openai/gpt-oss-120b", "meta/llama-3.3-70b-instruct", "qwen/qwen3-next-80b-a3b-instruct",
                "deepseek-ai/deepseek-v3.1")),
    "cloudflare": ("https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1", "CLOUDFLARE_API_KEY",
                   ("@cf/openai/gpt-oss-120b", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/qwen/qwen3-30b-a3b-fp8")),
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
COOLDOWN_SECONDS = 600   # timed out: a slow model costs every search 10s, so rest it longer
BUSY_SECONDS = 60        # 429 per-minute limit / 503 overloaded: usually fine again within a minute
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


def _daily_quota_reset() -> float:
    """Free daily quotas reset at midnight US Pacific time (08:00 UTC; no time-zone database needed on Windows)."""
    now = datetime.now(timezone.utc)
    reset = now.replace(hour=8, minute=1, second=0, microsecond=0)
    return (reset if reset > now else reset + timedelta(days=1)).timestamp()


def _rest(model: str, r: httpx.Response | None = None) -> None:
    """Skip a model for a while: until tomorrow if its daily quota is gone, else a few minutes."""
    body = r.text.lower() if r is not None else ""
    daily = r is not None and r.status_code == 429 and ("perday" in body or "per day" in body or "requests per day" in body
                                                         or "(rpd)" in body or "tokens per day" in body)
    busy = r is not None and r.status_code in (429, 503)
    refused = r is not None and r.status_code in (401, 403, 404)  # not available to us: skip it until tomorrow
    _COOLDOWN[model] = (_daily_quota_reset() if daily or refused
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
    if any(x in name for x in ("vision", "-vl", "coder", "embed", "guard", "audio", "image")):
        return False
    return _size(name) >= 24 or any(x in name for x in ("gpt-oss", "deepseek", "llama-4", "kimi", "glm-4", "qwen3-235"))


def _pick(available: list[str], preferred: tuple[str, ...], custom: str | None, provider: str) -> list[str]:
    order = ([custom] if custom else []) + list(preferred)
    if not available:  # listing failed: just try our guesses
        return list(dict.fromkeys(order))
    picked = [m for m in order if m in available]
    if provider == "gemini":  # newest flash models first, skip special-purpose ones
        extra = sorted((m for m in available if "flash" in m and not any(
            x in m for x in ("image", "tts", "audio", "live", "embedding", "thinking", "exp", "preview"))), reverse=True)
    elif provider in ("mistral", "nvidia", "cloudflare", "cerebras"):  # big catalogs: only our known-good picks
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
    return list(dict.fromkeys(picked + extra))[:4]


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


async def ask_json(prompt: str, timeout: float = 10, prefer: str | None = None, spread: int = 0) -> dict:
    """prefer="bulk": try the non-Gemini providers first (bulk work like reading pages, so Gemini's small daily
    quota is kept for understanding the request and judging the results); prefer="groq" etc.: that provider.
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
                attempts += [(provider, m, _caller(provider)) for m in models[:4]]
        # last model that worked goes first; models that just failed go last (or are skipped)
        now = time.time()
        def preferred(a) -> bool:
            return bool(prefer) and (a[0] != "gemini" if prefer == "bulk" else a[0] == prefer)
        attempts.sort(key=lambda a: (_COOLDOWN.get(a[1], 0) > now, not preferred(a), a[1] not in _LAST_GOOD))
        ready = [a for a in attempts if _COOLDOWN.get(a[1], 0) <= now] or attempts
        if spread and prefer:
            mine = [a for a in ready if preferred(a)]
            if mine:
                k = spread % len(mine)
                ready = mine[k:] + mine[:k] + [a for a in ready if not preferred(a)]

        for provider, model, call in ready:
            left = TOTAL_BUDGET - (time.monotonic() - start)
            if left < 2:
                break
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
                try:
                    data = _json(_text(provider, r))
                except (KeyError, IndexError, ValueError):
                    errors.append(f"{model}: bad reply")
                    continue
                if not isinstance(data, dict):  # e.g. a bare list: callers expect an object
                    data = {"products": data} if isinstance(data, list) else {}
                if _LAST_GOOD != [model]:
                    _LAST_GOOD[:] = [model]
                    _save_state()
                return data
            errors.append(f"{model}: {r.status_code}")
            log.warning("LLM %s said: %s", model, r.text[:200].replace("\n", " "))
            if r.status_code in RETRY_STATUS or r.status_code in (401, 403, 404):  # busy / quota / not allowed / gone
                _rest(model, r)
    raise LLMError("مفيش موديل رد في الوقت: " + ", ".join(errors[-4:] or ["timeout"]))
