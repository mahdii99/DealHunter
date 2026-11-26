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


def test_a_key_without_an_active_plan_rests_half_an_hour(monkeypatch):
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
    assert 1700 < llm._COOLDOWN["llama-3.1-8b-instant"] - time.time() <= 1800


def test_admin_limits_show_each_daily_limit_used_and_left(monkeypatch):
    """mahdy, 26 Nov: /admin shows each model's free daily limit (used, left, when it renews), not an error log."""
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "NVIDIA_API_KEY",
              "CLOUDFLARE_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm._COOLDOWN.clear()
    for _ in range(20):
        llm._tally("gemini", "gemini-flash-latest", "plan", 10)
    llm._tally("gemini", "gemini-flash-lite-latest", "judge", 10)
    llm._tally("nvidia", "meta/llama-3.3-70b-instruct", "judge", 10)
    rows = {r["name"]: r for r in llm.quota_report()}
    assert rows["gemini-flash-latest"]["used"] == 20 and rows["gemini-flash-latest"]["left"] == 0
    assert rows["gemini-flash-lite-latest"]["limit"] == 500 and rows["gemini-flash-lite-latest"]["left"] == 499
    assert rows["gemini gemma"]["used"] == 0 and rows["gemini gemma"]["limit"] == 14400   # unused, still shown
    assert rows["groq llama-3.1-8b"]["limit"] == 14400 and rows["groq other models"]["limit"] == 1000
    assert "openrouter:*" not in rows                                                      # no key
    assert rows["meta/llama-3.3-70b-instruct"]["limit"] is None and rows["meta/llama-3.3-70b-instruct"]["used"] == 1
    assert 0 < rows["gemini-flash-latest"]["resets_at"] - time.time() <= 86400
    assert list(rows)[0] == "gemini-flash-latest"                                          # used up first


def test_web_searches_are_counted_per_month_and_saved(monkeypatch, tmp_path):
    from app import websearch
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    db.close()
    for k in websearch.KEYS.values():
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setenv("SEARCH_MONTHLY_LIMITS", "tavily=300")
    websearch.PERSIST = True
    monkeypatch.setattr(websearch, "FLUSH_EVERY", 10 ** 9)
    websearch._last_flush[0] = time.time()
    for _ in range(3):
        websearch._count("tavily")
    websearch.flush_usage()
    websearch._MONTH.clear(); websearch._loaded_months.clear()  # the server restarts
    [row] = websearch.quota_report()
    assert row["name"] == "tavily" and row["used"] == 3 and row["limit"] == 300 and row["left"] == 297
    assert row["resets_at"] > time.time() and row["quota_out_until"] is None
    db.close()


def _clear_keys(monkeypatch, *keep):
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "NVIDIA_API_KEY",
              "CLOUDFLARE_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k in keep:
        monkeypatch.setenv(k, "k")


def test_a_daily_429_shows_used_up_even_when_our_count_is_zero(monkeypatch):
    """mahdy, 26 Nov: /admin said gemini flash 0 of 20 while Gemini answered 429: the provider's word wins."""
    _clear_keys(monkeypatch, "GEMINI_API_KEY")
    body = ('{"error": {"code": 429, "message": "You exceeded your current quota", "details": [{"violations": '
            '[{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests", '
            '"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}]}}')
    llm._note_reported("gemini", "gemini-flash-latest", httpx.Response(429, text=body))
    rows = {r["name"]: r for r in llm.quota_report()}
    row = rows["gemini-flash-latest"]
    assert row["used"] == 20 and row["left"] == 0 and row["limit"] == 20 and row["source"] == "provider"
    assert row["out_until"] > time.time() and "gemini flash" not in rows       # no 0-of-20 placeholder beside it
    assert list(rows)[0] == "gemini-flash-latest"
    assert llm.over_budget("gemini", "gemini-flash-latest", False)


def test_groq_daily_headers_beat_our_count(monkeypatch):
    _clear_keys(monkeypatch, "GROQ_API_KEY")
    llm._tally("groq", "openai/gpt-oss-120b", "judge", 10)  # we counted 1; the key was also used elsewhere
    r = httpx.Response(200, headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "12",
                                     "x-ratelimit-reset-requests": "2m59.5s"}, json={})
    llm._note_reported("groq", "openai/gpt-oss-120b", r)
    row = {r["name"]: r for r in llm.quota_report()}["openai/gpt-oss-120b"]
    assert row["used"] == 988 and row["left"] == 12 and row["source"] == "provider"
    assert row["resets_at"] - time.time() > 3600 * 0 and not row.get("out_until")
    assert llm.real_use("groq", "openai/gpt-oss-120b") == (988, 1000)
    assert llm.over_budget("groq", "openai/gpt-oss-120b", True)        # past the background share
    assert not llm.over_budget("groq", "openai/gpt-oss-120b", False)


def test_a_used_up_model_stays_skipped_after_a_restart(monkeypatch, tmp_path):
    _clear_keys(monkeypatch, "OPENROUTER_API_KEY")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    db.close()
    llm.PERSIST = True
    reset_ms = int((time.time() + 5 * 3600) * 1000)
    r = httpx.Response(429, text='{"error":{"message":"Rate limit exceeded: free-models-per-day. "}}',
                       headers={"X-RateLimit-Limit": "50", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset_ms)})
    llm._note_reported("openrouter", "openai/gpt-oss-20b:free", r)
    for _ in range(50):  # the save runs on a thread
        if db.quota_states():
            break
        time.sleep(0.02)
    llm._REPORTED.clear(); llm._reported_loaded[0] = False; llm._COOLDOWN.clear()  # the server restarts
    assert llm.over_budget("openrouter", "meta-llama/llama-3.3-70b-instruct:free", False)  # one limit for all its models
    row = {r["name"]: r for r in llm.quota_report()}["openrouter:*"]
    assert row["left"] == 0 and row["limit"] == 50 and abs(row["resets_at"] - reset_ms / 1000) < 1
    db.close()


def test_search_services_own_usage_wins_and_used_up_survives_a_restart(monkeypatch, tmp_path):
    from app import websearch
    for k in websearch.KEYS.values():
        monkeypatch.delenv(k, raising=False)
    for k in ("TAVILY_API_KEY", "FIRECRAWL_API_KEY", "EXA_API_KEY"):
        monkeypatch.setenv(k, "k")

    def handler(request: httpx.Request):
        if request.url.host == "api.tavily.com":
            return httpx.Response(200, json={"key": {"usage": 40, "limit": None},
                                             "account": {"current_plan": "Researcher", "plan_usage": 997, "plan_limit": 1000}})
        return httpx.Response(200, json={"success": True, "data": {"remainingCredits": 300, "planCredits": 1000,
                                                                   "billingPeriodEnd": "2026-12-20T00:00:00Z"}})
    real = httpx.AsyncClient
    monkeypatch.setattr(websearch.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    live = asyncio.run(websearch.provider_usage())
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "prices.db")
    db.close()
    websearch.PERSIST = True
    websearch._mark_out("exa", time.time() + 3600)
    for _ in range(50):
        if db.quota_states():
            break
        time.sleep(0.02)
    websearch._QUOTA_OUT.clear(); websearch._REST.clear()  # the server restarts
    rows = {r["name"]: r for r in websearch.quota_report(live)}
    assert rows["tavily"]["used"] == 997 and rows["tavily"]["left"] == 3 and rows["tavily"]["source"] == "provider"
    assert rows["firecrawl"]["left"] == 150 and rows["firecrawl"]["limit"] == 500 and rows["firecrawl"]["used"] == 350
    assert rows["exa"]["source"] == "ours" and rows["exa"]["left"] == 0 and rows["exa"]["quota_out_until"]
    assert websearch._REST["exa"] > time.time()   # skipped by searches too
    db.close()
