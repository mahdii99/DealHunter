import warnings

# Starlette's test client now prefers the separate `httpx2` package and warns once on import when it falls back to
# plain httpx (which we use anyway). That notice is about the test tool, not our code, so `pytest -W error` would
# otherwise fail before any test runs. Import it here once with only that warning ignored.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=r"Using `httpx` with `starlette.testclient` is deprecated")
    import starlette.testclient  # noqa: F401

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """`from app import main` loads the real .env: without this, tests would call the real search/LLM services (and
    spend the free quotas) and depend on which keys this machine has. Tests that need a key set a fake one."""
    from app import llm, websearch
    names = {*websearch.KEYS.values(), "GEMINI_API_KEY", "GEMINI_MODEL", "JINA_API_KEY", "CLOUDFLARE_ACCOUNT_ID",
             *(key for _, key, _ in llm.COMPAT.values()), *(f"{p.upper()}_MODEL" for p in llm.COMPAT)}
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(websearch, "KEYLESS", False)  # no keyless search service either (tests that want it set it)
    websearch._CACHE.clear()


@pytest.fixture(autouse=True)
def _debug_files_in_tmp(tmp_path, monkeypatch):
    """Tests must not overwrite the real debug/last_judge.json or debug/web/ pages (a live run may be using them)."""
    from app import agent
    monkeypatch.setattr(agent, "DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(agent, "WEB_DEBUG", tmp_path / "debug" / "web")


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch):
    """Store addresses in tests are made up: treat every host as a public internet address (a test that checks
    private addresses sets its own)."""
    from app import main

    async def public(host):
        return [(0, 0, 0, "", ("93.184.216.34", 443))]
    monkeypatch.setattr(main, "_resolve", public)


@pytest.fixture(autouse=True)
def _no_result_page_check(monkeypatch):
    """Opening every result's page again before showing it would make network calls in tests: off unless a test
    checks that step itself."""
    from app import agent
    monkeypatch.setattr(agent, "VERIFY_PICKS", False)
    from app import recipes
    monkeypatch.setattr(recipes, "ENABLED", False)  # tests that check it turn it on



@pytest.fixture(autouse=True)
def _stores_allow_us(monkeypatch):
    """Following a product asks the store's robots.txt first (a real request): allowed, unless a test says not."""
    from app import main

    async def allowed(url):
        return True
    monkeypatch.setattr(main, "_may_open", allowed)


@pytest.fixture(autouse=True)
def _problems_in_memory(monkeypatch):
    """Problems noted during tests go to a list (tests can read it), not to the real database."""
    from app import problems
    seen: list[tuple[str, str, str]] = []
    monkeypatch.setattr(problems, "note", lambda kind, detail="", request="": seen.append((kind, detail, request)))
    return seen


@pytest.fixture(autouse=True)
def _fresh_agent_memory():
    """Judge verdicts and shopper-words searches the agent keeps in memory: each test starts without them."""
    from app import agent
    agent._VERDICTS.clear()
    agent._HITS.clear()
    yield
    agent._VERDICTS.clear()
    agent._HITS.clear()


@pytest.fixture(autouse=True)
def _llm_counts_in_memory():
    """Each test counts its LLM calls and web searches from zero, and doesn't save them in the database (tests that
    check it do)."""
    from app import llm, websearch
    llm.PERSIST = websearch.PERSIST = False
    for book in (llm._DAY, llm._UNSAVED, llm._PROVIDER, websearch._MONTH, websearch._UNSAVED, websearch._QUOTA_OUT,
                 websearch._loaded_months):
        book.clear()
    llm._loaded_days.clear()
    llm._REPORTED.clear()
    llm._reported_loaded[0] = websearch._out_loaded[0] = False
    websearch._LIVE.clear()
    yield
    llm.PERSIST = websearch.PERSIST = False


@pytest.fixture(autouse=True)
def _no_real_store_searches(monkeypatch):
    """The big stores' own search pages (agent._store_search) are real sites: tests that need them set their own."""
    from app import agent
    monkeypatch.setattr(agent, "STORE_SEARCHES", {})


@pytest.fixture(autouse=True)
def _fresh_site_memory():
    """What sites did lately (app/agent.py _HOSTS) is kept in memory: each test starts with none."""
    from app import agent
    agent._HOSTS.clear()
    agent._HOST_SEMS.clear()
    yield
    agent._HOSTS.clear()
