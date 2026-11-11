"""Price database: every price the agent shows is saved with the date, so we can:
  - answer the same request again from the cache without spending the free quotas,
  - show a product's price history and the lowest price we've seen,
  - skip sites whose pages never open, count each visitor's searches, keep "this result is wrong" reports,
  - keep shopper accounts: Google sign-in, region, preferences, followed products, notifications (app/accounts.py).

Where it's stored:
  - DATABASE_URL set (a free Supabase Postgres): survives restarts and deploys. Needed on Render, whose free disk
    is wiped every time the service sleeps or updates.
  - not set: a local SQLite file, data/prices.db (your own PC, the tests).
"""
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("uvicorn.error")
DB_PATH = Path(os.getenv("PRICE_DB", Path(__file__).resolve().parent.parent / "data" / "prices.db"))
# {ID} and {FLOAT} differ between SQLite and Postgres; everything else is the same SQL on both
_SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    key TEXT PRIMARY KEY,           -- normalized request + market
    market TEXT NOT NULL,
    saved_at {FLOAT} NOT NULL,
    payload TEXT NOT NULL           -- the JSON answer sent to the page
);
CREATE TABLE IF NOT EXISTS prices (
    id {ID},
    market TEXT NOT NULL,
    url TEXT NOT NULL,              -- the product page: one product at one store
    store TEXT NOT NULL,
    title TEXT NOT NULL,
    price {FLOAT} NOT NULL,
    old_price {FLOAT},
    in_stock INTEGER,               -- 1 / 0 / NULL (unknown)
    seen_at {FLOAT} NOT NULL
);
CREATE INDEX IF NOT EXISTS prices_url ON prices (url, seen_at);
CREATE TABLE IF NOT EXISTS blocked_sites (
    domain TEXT PRIMARY KEY,        -- a site whose pages we can't open (bot protection, JS-only, robots.txt...)
    fails INTEGER NOT NULL,         -- searches in a row where none of its pages opened
    reason TEXT NOT NULL,           -- the last error
    last_failed {FLOAT} NOT NULL,
    blocked INTEGER NOT NULL        -- 1 = skipped in new searches until unblocked (scripts/blocked_sites.py)
);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,              -- YYYY-MM-DD (UTC)
    client TEXT NOT NULL,           -- a hash of the visitor's IP, never the IP itself
    searches INTEGER NOT NULL,
    PRIMARY KEY (day, client)
);
CREATE TABLE IF NOT EXISTS reports (
    id {ID},
    at {FLOAT} NOT NULL,
    reason TEXT NOT NULL,           -- wrong_product / wrong_price / bad_store / sold_out / other
    request TEXT NOT NULL,          -- what the shopper asked for
    url TEXT NOT NULL,
    store TEXT NOT NULL,
    title TEXT NOT NULL,
    price {FLOAT},
    client TEXT NOT NULL
);
-- shopper accounts (app/accounts.py)
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,                -- 'g:' + the Google account id (never changes, unlike the email)
    email TEXT NOT NULL,
    name TEXT NOT NULL,
    picture TEXT NOT NULL,
    market TEXT,                        -- EG / SA / AE, chosen at sign-up (NULL until chosen)
    email_alerts INTEGER NOT NULL DEFAULT 1,
    prefs TEXT NOT NULL DEFAULT '{{}}', -- JSON, see PREFS
    created_at {FLOAT} NOT NULL,
    last_login {FLOAT} NOT NULL
);
CREATE TABLE IF NOT EXISTS watches (
    id {ID},
    user_id TEXT NOT NULL,
    url TEXT NOT NULL,                  -- the product page: one product at one store
    title TEXT NOT NULL,
    store TEXT NOT NULL,
    image TEXT NOT NULL,
    market TEXT NOT NULL,
    first_price {FLOAT} NOT NULL,       -- when the shopper started following it
    price {FLOAT} NOT NULL,             -- the last price we saw
    target_price {FLOAT},               -- tell me only at or below this (NULL = any drop)
    created_at {FLOAT} NOT NULL,
    checked_at {FLOAT} NOT NULL,
    UNIQUE (user_id, url)
);
CREATE INDEX IF NOT EXISTS watches_url ON watches (url);
CREATE TABLE IF NOT EXISTS notifications (
    id {ID},
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,                 -- price_drop / info
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    url TEXT NOT NULL,
    created_at {FLOAT} NOT NULL,
    seen INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS notifications_user ON notifications (user_id, created_at)
"""
PG_RETRY = 300  # seconds on the local file after Postgres couldn't be reached, before trying it again
BLOCK_AFTER = 2  # failed searches in a row before a site is skipped (one failure can be a slow moment)


class _Store:
    """One connection, to Postgres (DATABASE_URL) or SQLite. Queries are written once, with ? placeholders."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conn = None
        self._key = None
        self.pg = False
        self._pg_down_until = 0.0

    def _connect(self):
        url = os.getenv("DATABASE_URL", "")
        if url and time.time() < self._pg_down_until:
            url = ""  # Postgres failed a moment ago: keep the site working on the local file meanwhile
        key = url or str(DB_PATH)
        if self._conn is not None and self._key == key:
            return self._conn
        self.close()  # the setting changed (tests): don't leave the old connection open
        conn = None
        if url:
            try:
                conn = self._connect_pg(url)
            except Exception as e:  # wrong DATABASE_URL, or the database is down/paused: searches must still work
                log.error("price db: can't reach DATABASE_URL (%s), using the local file for %d min",
                          type(e).__name__, PG_RETRY // 60)
                self._pg_down_until = time.time() + PG_RETRY
                url, key = "", str(DB_PATH)
        if conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
            conn.executescript(_SCHEMA.format(ID="INTEGER PRIMARY KEY", FLOAT="REAL"))
        self._conn, self._key, self.pg = conn, key, bool(url)
        return conn

    @staticmethod
    def _connect_pg(url: str):
        import psycopg  # only needed with Postgres
        # prepare_threshold=None: Supabase's pooler can't keep prepared statements between connections
        conn = psycopg.connect(url, autocommit=True, prepare_threshold=None, connect_timeout=10)
        for stmt in _SCHEMA.format(ID="BIGSERIAL PRIMARY KEY", FLOAT="DOUBLE PRECISION").split(";"):
            conn.execute(stmt)
        return conn

    def _do(self, work):
        with self._lock:
            for attempt in (1, 2):
                conn = self._connect()
                try:
                    return work(conn)
                except sqlite3.Error:
                    raise
                except Exception as e:  # a Postgres connection the server dropped: connect again once
                    if attempt == 2 or not self.pg or "Operational" not in type(e).__name__:
                        raise
                    log.warning("price db: reconnecting after %s", type(e).__name__)
                    self.close()

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.pg else sql

    def run(self, sql: str, args: tuple = ()) -> list[tuple]:
        def work(conn):
            cur = conn.execute(self._sql(sql), args)
            return cur.fetchall() if cur.description else []
        return self._do(work)

    def count(self, sql: str, args: tuple = ()) -> int:
        """Runs a change and says how many rows it touched."""
        return self._do(lambda conn: conn.execute(self._sql(sql), args).rowcount)

    def run_many(self, sql: str, rows: list[tuple]) -> None:
        if rows:
            self._do(lambda conn: conn.cursor().executemany(self._sql(sql), rows))

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


_db = _Store()


def close() -> None:
    _db.close()


def cache_key(request: str, market: str) -> str:
    return f"{market}:{' '.join(request.lower().split())}"


def cached_search(key: str, max_age: float) -> dict | None:
    row = next(iter(_db.run("SELECT saved_at, payload FROM searches WHERE key = ?", (key,))), None)
    if row and time.time() - row[0] < max_age:
        return json.loads(row[1])
    return None


def save_search(key: str, market: str, payload: dict) -> None:
    _db.run("INSERT INTO searches (key, market, saved_at, payload) VALUES (?, ?, ?, ?) ON CONFLICT (key) "
            "DO UPDATE SET market = excluded.market, saved_at = excluded.saved_at, payload = excluded.payload",
            (key, market, time.time(), json.dumps(payload, ensure_ascii=False)))


def _last_prices(urls: list[str]) -> dict[str, tuple]:
    """{url: (price, seen_at)} of the latest saved price of each page, in one query (the database may be far away)."""
    if not urls:
        return {}
    rows = _db.run(f"SELECT url, price, seen_at FROM prices WHERE url IN ({', '.join('?' * len(urls))}) "
                   "ORDER BY seen_at", tuple(urls))
    return {url: (price, at) for url, price, at in rows}  # later rows win: the latest


def record_prices(results: list[dict], market: str) -> None:
    """Save the prices shown to the shopper. The same price seen again within a day isn't saved twice."""
    now = time.time()
    results = [r for r in results if r.get("url") and r.get("price")]
    last = _last_prices(list({r["url"] for r in results}))
    rows, seen = [], set()
    for r in results:
        before = last.get(r["url"])
        if r["url"] in seen or (before and before[0] == r["price"] and now - before[1] < 24 * 3600):
            continue
        seen.add(r["url"])
        stock = r.get("in_stock")
        rows.append((market, r["url"], r.get("store") or "", r.get("title") or "", r["price"], r.get("old_price"),
                     None if stock is None else int(bool(stock)), now))
    _db.run_many("INSERT INTO prices (market, url, store, title, price, old_price, in_stock, seen_at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def history(url: str, limit: int = 200) -> list[dict]:
    """Oldest first: [{"price", "in_stock", "seen_at"}]."""
    rows = _db.run("SELECT price, in_stock, seen_at FROM prices WHERE url = ? ORDER BY seen_at DESC LIMIT ?",
                   (url, limit))
    return [{"price": p, "in_stock": None if s is None else bool(s), "seen_at": t} for p, s, t in reversed(rows)]


def add_price_context(results: list[dict]) -> None:
    """Adds to each result what we saw before: lowest_seen and previous_price (to say 'dropped from X')."""
    urls = list({r["url"] for r in results if r.get("url")})
    if not urls:
        return
    lowest = dict(_db.run(f"SELECT url, MIN(price) FROM prices WHERE url IN ({', '.join('?' * len(urls))}) "
                          "GROUP BY url", tuple(urls)))
    last = _last_prices(urls)
    for r in results:
        if not r.get("url"):
            continue
        prev = last.get(r["url"])
        r["lowest_seen"] = lowest.get(r["url"])
        r["previous_price"] = prev[0] if prev and prev[0] != r.get("price") else None


BLOCK_DAYS = 7  # a skipped site is tried again after this (sites change, and a block may have been a bad week)


def blocked_sites() -> set[str]:
    return {r[0] for r in _db.run("SELECT domain FROM blocked_sites WHERE blocked = 1 AND last_failed > ?",
                                  (time.time() - BLOCK_DAYS * 86400,))}


def site_failed(domain: str, reason: str, now_blocked: bool = False) -> None:
    """None of this site's pages opened in a search. After BLOCK_AFTER searches in a row (or right away when
    `now_blocked`, e.g. a store in another currency), new searches skip it."""
    row = _db.run("SELECT fails FROM blocked_sites WHERE domain = ?", (domain,))
    fails = (row[0][0] if row else 0) + 1
    _db.run("INSERT INTO blocked_sites (domain, fails, reason, last_failed, blocked) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (domain) DO UPDATE SET fails = excluded.fails, reason = excluded.reason, "
            "last_failed = excluded.last_failed, blocked = excluded.blocked",
            (domain, fails, reason[:300], time.time(), int(now_blocked or fails >= BLOCK_AFTER)))


def site_opened(domain: str) -> None:
    """A page of this site worked: its failures so far were a bad moment."""
    _db.run("DELETE FROM blocked_sites WHERE domain = ?", (domain,))


def list_blocked() -> list[dict]:
    rows = _db.run("SELECT domain, fails, reason, last_failed, blocked FROM blocked_sites "
                   "ORDER BY blocked DESC, last_failed DESC")
    return [{"domain": d, "fails": f, "reason": r, "last_failed": t, "blocked": bool(b)} for d, f, r, t, b in rows]


def unblock(domain: str) -> bool:
    return _db.count("DELETE FROM blocked_sites WHERE domain = ?", (domain,)) > 0


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def searches_today(client: str) -> int:
    row = _db.run("SELECT searches FROM usage WHERE day = ? AND client = ?", (_today(), client))
    return row[0][0] if row else 0


def count_search(client: str) -> None:
    _db.run("INSERT INTO usage (day, client, searches) VALUES (?, ?, 1) "
            "ON CONFLICT (day, client) DO UPDATE SET searches = usage.searches + 1", (_today(), client))


def add_report(reason: str, request: str, url: str, store: str, title: str, price: float | None, client: str) -> None:
    """A shopper said a result is wrong: kept so we can see what to fix (scripts/reports.py lists them)."""
    _db.run("INSERT INTO reports (at, reason, request, url, store, title, price, client) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), reason, request[:600], url[:1000], store[:100], title[:300], price, client))


def list_reports(limit: int = 200) -> list[dict]:
    rows = _db.run("SELECT at, reason, request, store, price, title, url FROM reports ORDER BY at DESC LIMIT ?",
                   (limit,))
    return [dict(zip(("at", "reason", "request", "store", "price", "title", "url"), r)) for r in rows]
