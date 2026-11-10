"""Local price database (SQLite, part of Python: free, no server).

Every price the agent shows is saved with the date, so we can:
  - answer the same request again from the cache (survives server restarts), without spending the free quotas,
  - show a product's price history and the lowest price we've seen,
  - (next) send price-drop alerts for followed products / categories.
"""
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(os.getenv("PRICE_DB", Path(__file__).resolve().parent.parent / "data" / "prices.db"))
_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    key TEXT PRIMARY KEY,           -- normalized request + market
    market TEXT NOT NULL,
    saved_at REAL NOT NULL,
    payload TEXT NOT NULL           -- the JSON answer sent to the page
);
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY,
    market TEXT NOT NULL,
    url TEXT NOT NULL,              -- the product page: one product at one store
    store TEXT NOT NULL,
    title TEXT NOT NULL,
    price REAL NOT NULL,
    old_price REAL,
    in_stock INTEGER,               -- 1 / 0 / NULL (unknown)
    seen_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS prices_url ON prices (url, seen_at);
"""
_conn: sqlite3.Connection | None = None
_conn_path: Path | None = None


def _db() -> sqlite3.Connection:
    global _conn, _conn_path
    if _conn is None or _conn_path != DB_PATH:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.executescript(_SCHEMA)
        _conn_path = DB_PATH
    return _conn


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def cache_key(request: str, market: str) -> str:
    return f"{market}:{' '.join(request.lower().split())}"


def cached_search(key: str, max_age: float) -> dict | None:
    with _LOCK:
        row = _db().execute("SELECT saved_at, payload FROM searches WHERE key = ?", (key,)).fetchone()
    if row and time.time() - row[0] < max_age:
        return json.loads(row[1])
    return None


def save_search(key: str, market: str, payload: dict) -> None:
    with _LOCK, _db() as c:
        c.execute("INSERT OR REPLACE INTO searches (key, market, saved_at, payload) VALUES (?, ?, ?, ?)",
                  (key, market, time.time(), json.dumps(payload, ensure_ascii=False)))


def record_prices(results: list[dict], market: str) -> None:
    """Save the prices shown to the shopper. The same price seen again within a day isn't saved twice."""
    now = time.time()
    with _LOCK, _db() as c:
        for r in results:
            if not r.get("url") or not r.get("price"):
                continue
            last = c.execute("SELECT price, seen_at FROM prices WHERE url = ? ORDER BY seen_at DESC LIMIT 1",
                             (r["url"],)).fetchone()
            if last and last[0] == r["price"] and now - last[1] < 24 * 3600:
                continue
            stock = r.get("in_stock")
            c.execute("INSERT INTO prices (market, url, store, title, price, old_price, in_stock, seen_at) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (market, r["url"], r.get("store") or "", r.get("title") or "", r["price"], r.get("old_price"),
                       None if stock is None else int(bool(stock)), now))


def history(url: str, limit: int = 200) -> list[dict]:
    """Oldest first: [{"price", "in_stock", "seen_at"}]."""
    with _LOCK:
        rows = _db().execute("SELECT price, in_stock, seen_at FROM prices WHERE url = ? ORDER BY seen_at DESC LIMIT ?",
                             (url, limit)).fetchall()
    return [{"price": p, "in_stock": None if s is None else bool(s), "seen_at": t} for p, s, t in reversed(rows)]


def add_price_context(results: list[dict]) -> None:
    """Adds to each result what we saw before: lowest_seen and previous_price (to say 'dropped from X')."""
    with _LOCK:
        db = _db()
        for r in results:
            if not r.get("url"):
                continue
            low, n = db.execute("SELECT MIN(price), COUNT(*) FROM prices WHERE url = ?", (r["url"],)).fetchone()
            prev = db.execute("SELECT price FROM prices WHERE url = ? ORDER BY seen_at DESC LIMIT 1",
                              (r["url"],)).fetchone()
            r["lowest_seen"] = low if n else None
            r["previous_price"] = prev[0] if prev and prev[0] != r.get("price") else None
