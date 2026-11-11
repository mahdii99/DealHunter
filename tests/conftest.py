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
