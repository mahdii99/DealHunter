"""Checks every key in .env with one tiny real request each, and says which ones work.

    python -m scripts.check_keys

Uses about 1 search from each search service and 1 small LLM request per provider.
"""
import asyncio
import os
import time

import httpx

from app import main  # noqa: F401  (loads .env)
from app import llm, websearch


def show(ok: bool, name: str, detail: str) -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<12} {detail}")


async def check_search(client: httpx.AsyncClient) -> None:
    print("Search:")
    for p in websearch.PROVIDERS:
        if not os.getenv(websearch.KEYS[p]):
            continue
        t = time.monotonic()
        try:
            r, parse = await websearch._CALLS[p](client, os.environ[websearch.KEYS[p]], "سخان اوليمبك هيرو 50 لتر",
                                                 5, None, None)
            if r.status_code != 200:
                show(False, p, f"{r.status_code}: {r.text[:120]}")
                continue
            hits = parse(r.json())
            show(bool(hits), p, f"{len(hits)} results in {time.monotonic() - t:.1f}s, e.g. {hits[0]['url'] if hits else '-'}")
        except Exception as e:  # report and keep checking the others
            show(False, p, f"{type(e).__name__}: {e}")


async def check_llm(client: httpx.AsyncClient) -> None:
    print("LLM:")
    providers = (["gemini"] if os.getenv("GEMINI_API_KEY") else []) + [p for p in llm.COMPAT if llm._configured(p)]
    for p in providers:
        models = await llm._list_models(client, p)
        if p == "gemini":
            picks = llm._pick(models, llm.GEMINI_PREFERRED, os.getenv("GEMINI_MODEL"), p)
            call = llm._gemini
        else:
            picks = llm._pick(models, llm.COMPAT[p][2], os.getenv(f"{p.upper()}_MODEL"), p)
            call = llm._caller(p)
        if not picks:
            show(False, p, "no usable model found")
            continue
        for model in picks[:3]:
            t = time.monotonic()
            try:
                r = await asyncio.wait_for(call(client, model, 'Reply with this JSON: {"ok": true}'), 20)
                if r.status_code == 200:
                    llm._json(llm._text(p, r))
                    show(True, p, f"{model} answered in {time.monotonic() - t:.1f}s")
                    break
                show(False, p, f"{model} -> {r.status_code}: {r.text[:120]}")
            except Exception as e:  # report and try the next model
                show(False, p, f"{model} -> {type(e).__name__}")


async def run() -> None:
    async with httpx.AsyncClient(timeout=20) as client:
        await check_search(client)
        await check_llm(client)


if __name__ == "__main__":
    asyncio.run(run())
