"""Free daily quotas: Gemma models on the Gemini key take the bulk work, each model's calls are counted per day (and
saved), and background work leaves the shoppers' share of every daily quota alone."""
import asyncio
import time

import httpx
import pytest

from app import db, llm


def _setup(monkeypatch, models: dict[str, list[str]], env=("GROQ_API_KEY",)):
    for k in env:
        monkeypatch.setenv(k, "k")
    llm._MODEL_CACHE.clear(); llm._COOLDOWN.clear(); llm._LAST_GOOD.clear(); llm._SLOW_COUNT.clear()

    async def fake_list(client, provider):
        return models.get(provider, [])
    monkeypatch.setattr(llm, "_list_models", fake_list)
    used = []

    async def answer(client, model, prompt, on_text=None):
        used.append(model)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}],
                                         "usage": {"total_tokens": 10},
                                         "candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}],
                                         "usageMetadata": {"totalTokenCount": 10}})
    monkeypatch.setattr(llm, "_gemini", answer)
    monkeypatch.setattr(llm, "_caller", lambda p: answer)
    return used


def _background(coro):
    async def run():
        token = llm.background()
        try:
            return await coro
        finally:
            llm.background_end(token)
    return asyncio.run(run())


GEMINI_LIST = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it",
               "gemma-3n-e4b-it"]


def test_only_the_big_gemma_models_are_used_and_they_are_not_strong():
    assert llm._gemma_models(GEMINI_LIST) == ["gemma-3-27b-it", "gemma-3-12b-it"]
    assert not llm.strong("gemma-3-27b-it", "gemini") and llm.strong("gemini-flash-latest", "gemini")


def test_background_page_reads_go_to_gemma_first_live_ones_to_the_fast_models(monkeypatch):
    used = _setup(monkeypatch, {"gemini": GEMINI_LIST, "groq": ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]},
                  env=("GROQ_API_KEY", "GEMINI_API_KEY"))
    _background(llm.ask_json("page", prefer="bulk", step="read_price"))
    assert used[-1].startswith("gemma-3-")
    asyncio.run(llm.ask_json("page", prefer="bulk", step="read_price"))
    assert used[-1] == "llama-3.1-8b-instant"
    asyncio.run(llm.ask_json("plan", step="plan"))  # planning never starts with Gemma
    assert not used[-1].startswith("gemma")


def test_gemma_is_asked_for_json_in_the_prompt_not_json_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    sent = {}

    class Client:
        async def post(self, url, headers, json):
            sent.update(json)
            return httpx.Response(200, json={})
    asyncio.run(llm._gemini(Client(), "gemma-3-27b-it", "say hi"))
    assert "responseMimeType" not in sent["generationConfig"] and "thinkingConfig" not in sent["generationConfig"]
    assert sent["contents"][0]["parts"][0]["text"].endswith("Reply with JSON only.")


def test_background_work_stops_at_its_share_of_the_daily_quota(monkeypatch):
    used = _setup(monkeypatch, {"groq": ["openai/gpt-oss-120b"]})
    monkeypatch.setattr(llm, "DAILY_LIMITS", [("groq", "", 10)])
    for _ in range(6):  # 60% of 10
        asyncio.run(llm.ask_json("x", step="plan"))
    with pytest.raises(llm.LLMError):
        _background(llm.ask_json("titles", prefer="bulk", step="understand"))
    assert len(used) == 6
    for _ in range(4):  # shoppers still get the rest of the day
        asyncio.run(llm.ask_json("x", step="plan"))
    with pytest.raises(llm.LLMError):  # and then nothing more is sent: a 429 would only cost time
        asyncio.run(llm.ask_json("x", step="plan"))
    assert len(used) == 10
    assert llm.calls_today("groq", "openai/gpt-oss-120b") == (10, 10)


def test_openrouter_counts_all_its_free_models_together():
    llm._PROVIDER.update({"a:free": "openrouter", "b:free": "openrouter"})
    day = llm.quota_day()
    llm._loaded_days.add(day)
    llm._DAY[(day, "a:free", "plan")] = [30, 0]
    llm._DAY[(day, "b:free", "judge")] = [20, 0]
    assert llm.calls_today("openrouter", "c:free") == (50, 50)
    assert llm.over_budget("openrouter", "c:free", background=False)


def test_the_day_counts_are_saved_and_read_back_after_a_restart(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    db.close()
    llm.PERSIST = True
    monkeypatch.setattr(llm, "FLUSH_EVERY", 10 ** 9)  # saved by flush_usage below, not by a thread
    llm._last_flush[0] = time.time()
    llm._loaded_days.add(llm.quota_day())  # nothing saved yet for today
    llm._tally("groq", "openai/gpt-oss-120b", "judge", 100)
    llm._tally("groq", "openai/gpt-oss-120b", "judge", 50)
    llm.flush_usage()
    llm.flush_usage()  # nothing new: nothing added twice
    for book in (llm._DAY, llm._UNSAVED, llm._PROVIDER):  # the server restarts
        book.clear()
    llm._loaded_days.clear()
    assert llm.calls_today("groq", "openai/gpt-oss-120b") == (2, 1000)
    report = llm.today_report()
    assert report[0]["model"] == "openai/gpt-oss-120b" and report[0]["tokens"] == 150
    assert report[0]["steps"] == {"judge": 2} and report[0]["limit"] == 1000
    db.close()


def test_special_purpose_models_in_long_lists_are_not_used():
    """Live test 2026-11-24: NVIDIA's list gave 'nvidia/ising-calibration-1.5-31b', which then read prices."""
    listed = ["nvidia/ising-calibration-1.5-31b", "nvidia/llama-3.3-nemotron-super-49b-v1", "foo/bar-70b",
              "nvidia/nv-rerankqa-mistral-4b-v3", "meta/llama-3.3-70b-instruct", "qwen/qwen3-next-80b-a3b-instruct"]
    picked = llm._pick(listed, (), None, "nvidia")
    assert "nvidia/ising-calibration-1.5-31b" not in picked and "foo/bar-70b" not in picked
    assert {"meta/llama-3.3-70b-instruct", "nvidia/llama-3.3-nemotron-super-49b-v1"} <= set(picked)


def test_a_search_keeps_which_models_failed_each_step(monkeypatch):
    """Live 2026-11-25: page reads timed out on every big store, and nothing said which models were slow."""
    _setup(monkeypatch, {"groq": ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]})

    async def busy(client, model, prompt, on_text=None):
        if model == "llama-3.1-8b-instant":
            return httpx.Response(429, json={"error": "rate"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})
    monkeypatch.setattr(llm, "_caller", lambda p: busy)

    async def run():
        llm.reset_usage()
        await llm.ask_json("page", prefer="bulk", step="read_page")
        return llm.usage()
    used = asyncio.run(run())
    step = used["by_step"]["read_page"]
    assert step["fails"] == {"llama-3.1-8b-instant: 429": 1} and step["calls"] == 1


def test_models_resting_past_the_call_are_not_asked_again(monkeypatch):
    """Live 2026-11-25: with every model resting, a retired model (410) was asked 8 times in one search."""
    used = _setup(monkeypatch, {"groq": ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]})
    for m in ("llama-3.1-8b-instant", "openai/gpt-oss-120b"):
        llm._COOLDOWN[m] = time.time() + 3600
    t = time.monotonic()
    with pytest.raises(llm.LLMError):
        asyncio.run(llm.ask_json("page", prefer="bulk", step="read_page"))
    assert used == [] and time.monotonic() - t < 1
    llm._COOLDOWN["llama-3.1-8b-instant"] = time.time() + 1  # free again within the call: it waits and asks it
    asyncio.run(llm.ask_json("page", prefer="bulk", step="read_page"))
    assert used == ["llama-3.1-8b-instant"]


def test_a_key_without_an_active_plan_is_skipped_until_tomorrow(monkeypatch):
    """Live 2026-11-25: a new Mistral key answered 402 (no plan yet) and was asked 8 times in one search."""
    used = _setup(monkeypatch, {"groq": ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]})

    async def pay(client, model, prompt, on_text=None):
        used.append(model)
        if model == "llama-3.1-8b-instant":
            return httpx.Response(402, json={"message": "No active plan"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})
    monkeypatch.setattr(llm, "_caller", lambda p: pay)
    asyncio.run(llm.ask_json("page", prefer="bulk", step="read_page"))
    asyncio.run(llm.ask_json("page", prefer="bulk", step="read_page"))
    assert used.count("llama-3.1-8b-instant") == 1
    assert llm._COOLDOWN["llama-3.1-8b-instant"] - time.time() > 3600
