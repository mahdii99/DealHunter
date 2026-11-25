"""Price database: every price the agent shows is saved with the date, so we can:
  - answer the same request again from the cache without spending the free quotas,
  - show a product's price history and the lowest price we've seen,
  - skip sites whose pages never open, count each visitor's searches, keep "this result is wrong" reports
    and the complaints & suggestions box,
  - keep shopper accounts: Google sign-in, region, preferences, followed products, notifications (app/accounts.py),
  - count how the site is used for the owner's /admin page: visits, chat turns, clicks on store links (events).

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

from . import pricechart

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
CREATE TABLE IF NOT EXISTS feedback (
    id {ID},
    at {FLOAT} NOT NULL,
    kind TEXT NOT NULL,             -- complaint / suggestion / result / question
    message TEXT NOT NULL,
    contact TEXT NOT NULL,          -- email or phone the visitor chose to leave, "" if none
    lang TEXT NOT NULL,
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
    wanted TEXT NOT NULL DEFAULT '',    -- the option the shopper wants (e.g. "مقاس 46"): alerts only when it's in stock
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
CREATE INDEX IF NOT EXISTS notifications_user ON notifications (user_id, created_at);
CREATE TABLE IF NOT EXISTS chats (
    id TEXT NOT NULL,                   -- made by the page (random), so it can save before the server answers
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,                -- the shopper's first message
    data TEXT NOT NULL,                 -- JSON: the conversation and the answers shown, as the page keeps them
    updated_at {FLOAT} NOT NULL,
    PRIMARY KEY (user_id, id)
);
CREATE INDEX IF NOT EXISTS chats_user ON chats (user_id, updated_at);
-- site usage for the owner's /admin page (kept EVENTS_DAYS days)
CREATE TABLE IF NOT EXISTS events (
    id {ID},
    at {FLOAT} NOT NULL,
    kind TEXT NOT NULL,                 -- visit / turn (a chat message or search) / click (a store link opened) / follow / problem (app/problems.py)
    client TEXT NOT NULL,               -- a hash of the visitor's IP, never the IP itself
    member INTEGER NOT NULL,            -- 1 = signed in
    detail TEXT NOT NULL,               -- visit: the site they came from, turn: their message, click: the page
    request TEXT NOT NULL,              -- turn / click: the search request it was about
    store TEXT NOT NULL,                -- click: the store
    title TEXT NOT NULL,                -- click: the product
    price {FLOAT},                      -- click: the price shown
    action TEXT NOT NULL,               -- turn: search / ask / recommend / answer / limit, visit: mobile / desktop
    results INTEGER,                    -- turn: results shown (search), click: its place in the list (0 = first)
    cached INTEGER                      -- turn: 1 = answered from the price database
);
CREATE INDEX IF NOT EXISTS events_at ON events (kind, at);
-- noon's public product sitemaps (app/noon_index.py): name and address of every product noon lists, no price. A
-- search shows the few that fit as a "see it on noon" link. Rebuilt one sitemap file at a time (`file`, with no index on
-- it: ~1.2M rows on the free database, a weekly delete can read the table).
CREATE TABLE IF NOT EXISTS noon_index (
    market TEXT NOT NULL,
    file INTEGER NOT NULL,              -- the sitemap file it came from (sitemap-eg-pdp-<file>.xml)
    head TEXT NOT NULL,                 -- the first word of its name (usually the brand): what a search looks it up by
    sku TEXT NOT NULL,
    slug TEXT NOT NULL,                 -- its name as noon writes it in the address, e.g. samsung-galaxy-a56-5g-...
    img TEXT NOT NULL                   -- the version part of its picture's address, or ''
);
CREATE INDEX IF NOT EXISTS noon_index_head ON noon_index (market, head);
-- our own price list: the big stores' category pages read on a schedule (app/catalog.py). Latest row per product page,
-- every price change also goes to `prices`, which keeps the history.
CREATE TABLE IF NOT EXISTS catalog (
    url TEXT PRIMARY KEY,               -- the product page: one product at one store
    market TEXT NOT NULL,
    category TEXT NOT NULL,             -- phones / tvs / ... (catalog.CATEGORIES)
    store TEXT NOT NULL,
    title TEXT NOT NULL,
    price {FLOAT} NOT NULL,             -- as the store's own listing showed it at seen_at
    old_price {FLOAT},
    image TEXT NOT NULL,
    first_seen {FLOAT} NOT NULL,
    seen_at {FLOAT} NOT NULL
);
CREATE INDEX IF NOT EXISTS catalog_seen ON catalog (market, seen_at);
-- how to read one store's product page without the LLM (app/recipes.py): CSS selectors the LLM picked once on a page
-- it had read, confirmed on more pages before use, learned again when they stop working
CREATE TABLE IF NOT EXISTS recipes (
    domain TEXT PRIMARY KEY,
    recipe TEXT NOT NULL,               -- JSON: CSS selectors of the price, the buy button, a sold-out mark
    ok INTEGER NOT NULL DEFAULT 0,      -- pages where it gave the price the LLM read (in a row)
    fails INTEGER NOT NULL DEFAULT 0,   -- pages where it gave nothing or another price (in a row)
    uses INTEGER NOT NULL DEFAULT 0,    -- pages read with it instead of the LLM
    learned_at {FLOAT} NOT NULL,
    checked_at {FLOAT} NOT NULL
);
-- store promo codes (app/coupons.py): only from an affiliate network's feed or added by the owner from the store
-- itself / a network's offer page. `source` says which, and the page shows it.
CREATE TABLE IF NOT EXISTS coupons (
    id {ID},
    market TEXT NOT NULL,
    store TEXT NOT NULL,                -- the store's domain: noon.com, amazon.eg, ...
    code TEXT NOT NULL,                 -- '' = a discount that needs no code (the link applies it)
    title TEXT NOT NULL,                -- what it gives: "10% off electronics, up to 300 EGP"
    link TEXT NOT NULL,                 -- where the shopper uses it (a network's tracking link or the store page)
    source TEXT NOT NULL,               -- admitad / owner:<where it came from>
    ext_id TEXT NOT NULL DEFAULT '',    -- the feed's own id, so a re-read replaces instead of adding twice
    expires_at {FLOAT},                 -- NULL = the source gave no end date
    added_at {FLOAT} NOT NULL
);
CREATE INDEX IF NOT EXISTS coupons_market ON coupons (market, store);
-- one page per product for shoppers coming from Google (app/pages.py): the offers checked on their store page
CREATE TABLE IF NOT EXISTS product_pages (
    slug TEXT NOT NULL,                 -- the page address: /p/<slug>
    market TEXT NOT NULL,
    name TEXT NOT NULL,                 -- "Samsung Galaxy A56 256GB"
    category TEXT NOT NULL,
    brand TEXT NOT NULL,
    urls TEXT NOT NULL,                 -- JSON: the store pages of this product (its price history)
    offers TEXT NOT NULL,               -- JSON: the offers whose page showed their price at their checked_at
    checked_at {FLOAT} NOT NULL,        -- when the offers were last checked
    first_made {FLOAT} NOT NULL,
    PRIMARY KEY (slug, market)
);
CREATE TABLE IF NOT EXISTS llm_usage (
    day TEXT NOT NULL,                  -- the free-quota day (starts at midnight US Pacific, app/llm.quota_day)
    model TEXT NOT NULL,
    step TEXT NOT NULL,                 -- plan / judge / read_price / understand / ... (what the call was for)
    provider TEXT NOT NULL,
    calls INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    PRIMARY KEY (day, model, step)
)
"""
# columns added after a table first went live (CREATE TABLE IF NOT EXISTS doesn't add them to an existing table)
_ADDED_COLUMNS = ["ALTER TABLE watches ADD COLUMN wanted TEXT NOT NULL DEFAULT ''",
                  # catalog: a shop's trust score (NULL = a big store read from its category pages), what the LLM
                  # made of the title (JSON) and the words a search finds it by (English and Arabic), when it did
                  "ALTER TABLE catalog ADD COLUMN trust INTEGER",
                  "ALTER TABLE catalog ADD COLUMN attrs TEXT NOT NULL DEFAULT ''",
                  "ALTER TABLE catalog ADD COLUMN keywords TEXT NOT NULL DEFAULT ''",
                  "ALTER TABLE catalog ADD COLUMN norm_at DOUBLE PRECISION"]
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
            for stmt in _ADDED_COLUMNS:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:  # duplicate column: already there
                    pass
        self._conn, self._key, self.pg = conn, key, bool(url)
        return conn

    @staticmethod
    def _connect_pg(url: str):
        import psycopg  # only needed with Postgres
        # prepare_threshold=None: Supabase's pooler can't keep prepared statements between connections
        conn = psycopg.connect(url, autocommit=True, prepare_threshold=None, connect_timeout=10)
        for stmt in _SCHEMA.format(ID="BIGSERIAL PRIMARY KEY", FLOAT="DOUBLE PRECISION").split(";"):
            conn.execute(stmt)
        for stmt in _ADDED_COLUMNS:
            conn.execute(stmt.replace("ADD COLUMN", "ADD COLUMN IF NOT EXISTS"))
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


TRENDING_KEY = "__trending__"  # the request ideas under the search box are kept in `searches` under this request


def recent_requests(market: str, days: int = 14, limit: int = 80) -> list[str]:
    """What shoppers searched lately (one row per distinct request), newest first."""
    rows = _db.run("SELECT key FROM searches WHERE market = ? AND saved_at > ? ORDER BY saved_at DESC LIMIT ?",
                   (market, time.time() - days * 86400, limit + 1))
    prefix = f"{market}:"
    # only shoppers' requests: other keys keep our own notes (checked prices, re-read pages)
    return [k.removeprefix(prefix) for (k,) in rows
            if k.startswith(prefix) and k != cache_key(TRENDING_KEY, market)][:limit]


def save_search(key: str, market: str, payload: dict) -> None:
    _db.run("INSERT INTO searches (key, market, saved_at, payload) VALUES (?, ?, ?, ?) ON CONFLICT (key) "
            "DO UPDATE SET market = excluded.market, saved_at = excluded.saved_at, payload = excluded.payload",
            (key, market, time.time(), json.dumps(payload, ensure_ascii=False)))


def recent_prices_like(word: str, market: str, since: float, limit: int = 6) -> list[dict]:
    """Prices we saw lately (in anyone's search) on pages whose title has this word, e.g. a model number: what we
    already know for free before spending a web search on it. Newest first."""
    rows = _db.run("SELECT title, store, price, url FROM prices WHERE market = ? AND seen_at > ? AND LOWER(title) LIKE ? "
                   "ORDER BY seen_at DESC LIMIT ?", (market, since, f"%{word.lower()}%", limit))
    return [{"title": t, "store": s, "price": p, "url": u} for t, s, p, u in rows]


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


def histories(urls: list[str], since: float) -> dict[str, list[dict]]:
    """{url: history points since `since`, oldest first, plus the last one before it (the price that held then)},
    asked in chunks: one page of deals or a product page with several stores costs a few queries, not one each."""
    out: dict[str, list[dict]] = {}
    urls = list(dict.fromkeys(u for u in urls if u))
    for i in range(0, len(urls), 400):
        part = urls[i:i + 400]
        marks = ", ".join("?" * len(part))
        held = _db.run(f"SELECT p.url, p.price, p.in_stock, p.seen_at FROM prices p JOIN (SELECT url, MAX(seen_at) AS at "
                       f"FROM prices WHERE url IN ({marks}) AND seen_at < ? GROUP BY url) m "
                       "ON p.url = m.url AND p.seen_at = m.at", (*part, since))
        rows = _db.run(f"SELECT url, price, in_stock, seen_at FROM prices WHERE url IN ({marks}) AND seen_at >= ? "
                       "ORDER BY seen_at", (*part, since))
        for url, price, stock, at in list(held) + list(rows):
            out.setdefault(url, []).append({"price": price, "in_stock": None if stock is None else bool(stock),
                                            "seen_at": at})
    return out


def known_product(url: str) -> dict | None:
    """The store, title and latest price of a product page we showed or listed (our prices or price list), or None:
    our /go link only sends shoppers to pages like these (never anywhere a link someone made up points to)."""
    rows = _db.run("SELECT store, title, price FROM prices WHERE url = ? ORDER BY seen_at DESC LIMIT 1", (url,)) or \
        _db.run("SELECT store, title, price FROM catalog WHERE url = ? LIMIT 1", (url,))
    return {"store": rows[0][0], "title": rows[0][1], "price": rows[0][2]} if rows else None


NOON_COLS = ("market", "file", "head", "sku", "slug", "img")


def noon_replace_file(market: str, file: int, rows: list[tuple]) -> None:
    """The products of one noon sitemap file, in place of what that file had before (rows: head, sku, slug, img)."""
    full = [(market, file, *r) for r in rows]

    def work(conn):
        if _db.pg:
            with conn.transaction():
                conn.execute("DELETE FROM noon_index WHERE market = %s AND file = %s", (market, file))
                with conn.cursor().copy(f"COPY noon_index ({', '.join(NOON_COLS)}) FROM STDIN") as cp:
                    for r in full:
                        cp.write_row(r)
        else:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM noon_index WHERE market = ? AND file = ?", (market, file))
            conn.executemany(f"INSERT INTO noon_index ({', '.join(NOON_COLS)}) VALUES (?, ?, ?, ?, ?, ?)", full)
            conn.execute("COMMIT")
    _db._do(work)


def noon_find(market: str, head: str, words: list[str], limit: int = 400) -> list[dict]:
    """noon products whose name starts with `head` and has every one of `words` as a whole word of the name."""
    likes = " ".join("AND ('-' || slug || '-') LIKE ?" for _ in words)
    return _rows(f"SELECT sku, slug, img FROM noon_index WHERE market = ? AND head = ? {likes} LIMIT ?",
                 (market, head, *(f"%-{w}-%" for w in words), limit), "sku slug img")


def noon_known(market: str, head: str, sku: str, slug: str) -> bool:
    return bool(_db.run("SELECT 1 FROM noon_index WHERE market = ? AND head = ? AND sku = ? AND slug = ? LIMIT 1",
                        (market, head, sku, slug)))


def noon_count(market: str) -> int:
    return _db.run("SELECT COUNT(*) FROM noon_index WHERE market = ?", (market,))[0][0]


def size_mb() -> float | None:
    """How big the database is (Postgres only: the free plan has a size limit)."""
    _db.run("SELECT 1")  # connected, so we know which database it is
    if not _db.pg:
        return None
    return round(_db.run("SELECT pg_database_size(current_database())")[0][0] / 1e6, 1)


def add_price_context(results: list[dict]) -> None:
    """Adds to each result what we saw before: lowest_seen, previous_price (to say 'dropped from X') and
    price_verdict (app/pricechart.verdict: is its discount real, judged on the last 30 days of this page)."""
    urls = list({r["url"] for r in results if r.get("url")})
    if not urls:
        return
    lowest = dict(_db.run(f"SELECT url, MIN(price) FROM prices WHERE url IN ({', '.join('?' * len(urls))}) "
                          "GROUP BY url", tuple(urls)))
    last = _last_prices(urls)
    past = histories(urls, time.time() - (pricechart.DAYS + 1) * 86400)
    for r in results:
        if not r.get("url"):
            continue
        prev = last.get(r["url"])
        r["lowest_seen"] = lowest.get(r["url"])
        r["previous_price"] = prev[0] if prev and prev[0] != r.get("price") else None
        r["price_verdict"] = pricechart.verdict(past.get(r["url"], []), r.get("price"), r.get("old_price"))


FOUND = "found"  # catalog category of offers a search found on the web and checked on their own page


def save_catalog(rows: list[dict], market: str, category: str, history: bool = True) -> int:
    """Products just read from a store's category page (or checked on their own page by a search: category FOUND,
    with the shop's trust score): the latest price per page, plus the price history. A page's category from the
    stores' own lists is kept when a search finds it again. A new title is understood again (norm_at back to NULL)."""
    now = time.time()
    rows = [r for r in rows if r.get("url") and r.get("price") and r.get("title")]
    rows = list({r["url"]: r for r in rows}.values())
    keep_cat = "CASE WHEN catalog.category <> 'found' THEN catalog.category ELSE excluded.category END" \
        if category == FOUND else "excluded.category"
    _db.run_many("INSERT INTO catalog (url, market, category, store, title, price, old_price, image, first_seen, seen_at,"
                 " trust) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (url) DO UPDATE SET "
                 f"market = excluded.market, category = {keep_cat}, store = excluded.store, "
                 "norm_at = CASE WHEN catalog.title = excluded.title THEN catalog.norm_at ELSE NULL END, "
                 "title = excluded.title, price = excluded.price, old_price = excluded.old_price, "
                 "image = CASE WHEN excluded.image <> '' THEN excluded.image ELSE catalog.image END, "
                 "seen_at = excluded.seen_at, "
                 "trust = CASE WHEN catalog.trust IS NULL AND catalog.category <> 'found' THEN NULL ELSE excluded.trust END",
                 [(r["url"], market, category, r.get("store") or "", r["title"][:300], r["price"], r.get("old_price"),
                   (r.get("image") or "")[:1000], now, now, r.get("trust")) for r in rows])
    if history:
        record_prices(rows, market)
    return len(rows)


_CATALOG_COLS = ("url", "store", "title", "price", "old_price", "image", "category", "seen_at", "trust", "attrs",
                 "keywords")


def catalog_like(words: list[str], market: str, since: float, limit: int = 300, found_since: float | None = None) -> list[dict]:
    """Catalog products seen since `since` (offers a search found: since `found_since`) whose title or the words the
    LLM gave it (keywords, English and Arabic) have any of these words (the caller ranks them)."""
    words = [w.lower() for w in words if w][:8]
    if not words:
        return []
    text = "LOWER(title || ' ' || keywords)"
    like = " OR ".join(f"{text} LIKE ?" for _ in words)
    # the products with the most of these words first (the words come rarest first, so the first ones weigh more):
    # with thousands of products, "samsung" alone matches TVs and fridges too, and a newest-first cut lost the phone
    score = " + ".join(f"(CASE WHEN {text} LIKE ? THEN {len(words) - i} ELSE 0 END)" for i in range(len(words)))
    fresh = "(seen_at > ? OR (category = 'found' AND seen_at > ?))"
    pats = [f"%{w}%" for w in words]
    rows = _db.run(f"SELECT {', '.join(_CATALOG_COLS)} FROM catalog "
                   f"WHERE market = ? AND {fresh} AND ({like}) ORDER BY ({score}) DESC, seen_at DESC LIMIT ?",
                   (market, since, since if found_since is None else found_since, *pats, *pats, limit))
    return [dict(zip(_CATALOG_COLS, r)) for r in rows]


def catalog_to_understand(market: str, limit: int) -> list[dict]:
    """Catalog products the LLM hasn't understood yet (new, or their title changed), newest first."""
    return _rows("SELECT url, title, category FROM catalog WHERE market = ? AND norm_at IS NULL "
                 "ORDER BY seen_at DESC LIMIT ?", (market, limit), "url title category")


def save_understood(items: list[tuple[str, str, str]]) -> None:
    """(url, attrs JSON, keywords) for products the LLM read the title of."""
    now = time.time()
    _db.run_many("UPDATE catalog SET attrs = ?, keywords = ?, norm_at = ? WHERE url = ?",
                 [(attrs[:2000], words[:1000], now, url) for url, attrs, words in items])


def get_recipe(domain: str) -> dict | None:
    rows = _db.run("SELECT recipe, ok, fails, uses FROM recipes WHERE domain = ?", (domain,))
    if not rows:
        return None
    recipe, ok, fails, uses = rows[0]
    return {"recipe": recipe, "ok": ok, "fails": fails, "uses": uses}


def save_recipe(domain: str, recipe: str) -> None:
    """A recipe just learned (and already right on the page it was learned on: ok = 1)."""
    now = time.time()
    _db.run("INSERT INTO recipes (domain, recipe, ok, fails, uses, learned_at, checked_at) VALUES (?, ?, 1, 0, 0, ?, ?) "
            "ON CONFLICT (domain) DO UPDATE SET recipe = excluded.recipe, ok = 1, fails = 0, uses = 0, "
            "learned_at = excluded.learned_at, checked_at = excluded.checked_at", (domain, recipe, now, now))


def recipe_result(domain: str, right: bool | None) -> None:
    """right: the recipe agreed with the LLM's read of a page (True), didn't (False), or was used instead of it (None)."""
    if right is None:
        sql = "UPDATE recipes SET uses = uses + 1 WHERE domain = ?"
    elif right:
        sql = "UPDATE recipes SET ok = ok + 1, fails = 0, checked_at = ? WHERE domain = ?"
    else:
        sql = "UPDATE recipes SET fails = fails + 1, ok = 0, checked_at = ? WHERE domain = ?"
    _db.run(sql, (domain,) if right is None else (time.time(), domain))


def recipes_list() -> list[dict]:
    return _rows("SELECT domain, ok, fails, uses, learned_at FROM recipes ORDER BY uses DESC", (),
                 "domain ok fails uses learned_at")


def save_product_page(slug: str, market: str, name: str, category: str, brand: str, urls: list[str],
                      offers: list[dict], now: float) -> None:
    """A product page after a check. A check that confirmed no offer keeps the offers of the last good one (they stop
    being shown once that check is too old, app/pages.SHOW_HOURS)."""
    keep = "product_pages.offers" if not offers else "excluded.offers"
    when = "product_pages.checked_at" if not offers else "excluded.checked_at"
    _db.run("INSERT INTO product_pages (slug, market, name, category, brand, urls, offers, checked_at, first_made) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (slug, market) DO UPDATE SET name = excluded.name, "
            f"category = excluded.category, brand = excluded.brand, urls = excluded.urls, offers = {keep}, "
            f"checked_at = {when}",
            (slug, market, name, category, brand, json.dumps(urls), json.dumps(offers, ensure_ascii=False),
             now if offers else 0, now))


_PAGE_COLS = ("slug", "name", "category", "brand", "urls", "offers", "checked_at", "first_made")


def _page_row(r: tuple) -> dict:
    d = dict(zip(_PAGE_COLS, r))
    d["urls"], d["offers"] = json.loads(d["urls"] or "[]"), json.loads(d["offers"] or "[]")
    return d


def product_page(slug: str, market: str) -> dict | None:
    rows = _db.run(f"SELECT {', '.join(_PAGE_COLS)} FROM product_pages WHERE slug = ? AND market = ?", (slug, market))
    return _page_row(rows[0]) if rows else None


def product_pages(market: str, since: float) -> list[dict]:
    """Product pages whose offers were checked since then, newest check first."""
    rows = _db.run(f"SELECT {', '.join(_PAGE_COLS)} FROM product_pages WHERE market = ? AND checked_at >= ? "
                   "ORDER BY checked_at DESC LIMIT 5000", (market, since))
    return [_page_row(r) for r in rows]


def catalog_since(market: str, since: float) -> list[dict]:
    """The big stores' products read from their category pages since then (not offers searches found)."""
    rows = _db.run(f"SELECT {', '.join(_CATALOG_COLS)} FROM catalog "
                   "WHERE market = ? AND seen_at > ? AND category <> 'found'", (market, since))
    return [dict(zip(_CATALOG_COLS, r)) for r in rows]


def catalog_stats(market: str, since: float) -> list[dict]:
    return _rows("SELECT store, category, COUNT(*), MAX(seen_at) FROM catalog WHERE market = ? AND seen_at > ? "
                 "GROUP BY store, category ORDER BY 1, 2", (market, since), "store category n last")


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


def add_llm_usage(rows: list[tuple]) -> None:
    """rows: (day, model, step, provider, calls, tokens) to add to that day's counts."""
    _db.run_many("INSERT INTO llm_usage (day, model, step, provider, calls, tokens) VALUES (?, ?, ?, ?, ?, ?) "
                 "ON CONFLICT (day, model, step) DO UPDATE SET calls = llm_usage.calls + excluded.calls, "
                 "tokens = llm_usage.tokens + excluded.tokens, provider = excluded.provider", rows)


def llm_usage_day(day: str) -> list[dict]:
    rows = _db.run("SELECT model, step, provider, calls, tokens FROM llm_usage WHERE day = ?", (day,))
    return [dict(zip(("model", "step", "provider", "calls", "tokens"), r)) for r in rows]


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


def add_feedback(kind: str, message: str, contact: str, lang: str, client: str) -> None:
    """The complaints & suggestions box on the page (scripts/feedback.py lists them)."""
    _db.run("INSERT INTO feedback (at, kind, message, contact, lang, client) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), kind, message[:2000], contact[:200], lang[:5], client))


def list_feedback(limit: int = 200) -> list[dict]:
    rows = _db.run("SELECT at, kind, message, contact, lang FROM feedback ORDER BY at DESC LIMIT ?", (limit,))
    return [dict(zip(("at", "kind", "message", "contact", "lang"), r)) for r in rows]


def list_reports(limit: int = 200) -> list[dict]:
    rows = _db.run("SELECT at, reason, request, store, price, title, url FROM reports ORDER BY at DESC LIMIT ?",
                   (limit,))
    return [dict(zip(("at", "reason", "request", "store", "price", "title", "url"), r)) for r in rows]


EVENTS_DAYS = 180  # usage events older than this are deleted
_EVENT_FIELDS = ("detail", "request", "store", "title", "price", "action", "results", "cached")
_LIMITS = {"detail": 1000, "request": 600, "store": 100, "title": 300, "action": 20}
_pruned = [0.0]


def add_event(kind: str, client: str, member: bool = False, **fields) -> None:
    """One thing a visitor did (see the events table). Text is cut to its column's size; old events are deleted
    about once an hour."""
    now = time.time()
    vals = []
    for name in _EVENT_FIELDS:
        v = fields.get(name)
        if name in _LIMITS:
            v = str(v or "")[:_LIMITS[name]]
        elif name == "cached" and v is not None:
            v = int(bool(v))
        vals.append(v)
    _db.run(f"INSERT INTO events (at, kind, client, member, {', '.join(_EVENT_FIELDS)}) "
            f"VALUES (?, ?, ?, ?, {', '.join('?' * len(_EVENT_FIELDS))})", (now, kind, client, int(member), *vals))
    if now - _pruned[0] > 3600:
        _pruned[0] = now
        _db.run("DELETE FROM events WHERE at < ?", (now - EVENTS_DAYS * 86400,))


def _rows(sql: str, args: tuple, names: str) -> list[dict]:
    return [dict(zip(names.split(), r)) for r in _db.run(sql, args)]


def usage_stats(days: int, tz_minutes: int = 0, top: int = 25) -> dict:
    """Everything the /admin page shows for the last `days` days. tz_minutes = the viewer's offset from UTC, so the
    days in the chart start at their midnight (Cairo is +120 or +180)."""
    now = time.time()
    since, off = now - days * 86400, tz_minutes * 60
    # whole days in the viewer's time: CAST(...)/86400 is integer division in both SQLite and Postgres
    daily = _rows("SELECT CAST(at + ? AS BIGINT) / 86400 AS d, kind, COUNT(*), COUNT(DISTINCT client) FROM events "
                  "WHERE at > ? GROUP BY d, kind ORDER BY d", (off, since), "day kind n people")
    totals = _rows("SELECT kind, COUNT(*), COUNT(DISTINCT client), SUM(member) FROM events WHERE at > ? GROUP BY kind",
                   (since,), "kind n people members")
    turn = "FROM events WHERE kind = 'turn' AND at > ?"
    actions = _rows(f"SELECT action, COUNT(*) {turn} GROUP BY action ORDER BY 2 DESC", (since,), "action n")
    searches = _rows(f"SELECT LOWER(request), COUNT(*), COUNT(DISTINCT client), MAX(at), AVG(results), SUM(cached) "
                     f"{turn} AND action = 'search' AND request <> '' GROUP BY LOWER(request) ORDER BY 2 DESC, 4 DESC "
                     "LIMIT ?", (since, top), "request n people last avg_results cached")
    empty = _rows(f"SELECT LOWER(request), COUNT(*), MAX(at) {turn} AND action = 'search' AND results = 0 "
                  "GROUP BY LOWER(request) ORDER BY 2 DESC, 3 DESC LIMIT ?", (since, top), "request n last")
    click = "FROM events WHERE kind = 'click' AND at > ?"
    stores = _rows(f"SELECT store, COUNT(*), COUNT(DISTINCT client) {click} GROUP BY store ORDER BY 2 DESC LIMIT ?",
                   (since, top), "store n people")
    products = _rows(f"SELECT detail, MAX(title), MAX(store), MIN(price), COUNT(*), MAX(at) {click} "
                     "GROUP BY detail ORDER BY 5 DESC, 6 DESC LIMIT ?", (since, top), "url title store price n last")
    ranks = _rows(f"SELECT results, COUNT(*) {click} AND results IS NOT NULL GROUP BY results ORDER BY 1",
                  (since,), "rank n")
    sources = _rows("SELECT detail, COUNT(*), COUNT(DISTINCT client) FROM events WHERE kind = 'visit' AND at > ? "
                    "GROUP BY detail ORDER BY 2 DESC LIMIT ?", (since, top), "source n people")
    devices = _rows("SELECT action, COUNT(DISTINCT client) FROM events WHERE kind = 'visit' AND at > ? "
                    "GROUP BY action", (since,), "device people")
    recent = _rows("SELECT at, kind, client, member, detail, request, store, title, price, action, results, cached "
                   "FROM events WHERE kind NOT IN ('visit', 'problem', 'llm') AND at > ? ORDER BY at DESC LIMIT 80", (since,),
                   "at kind client member detail request store title price action results cached")
    for r in recent:
        r["client"] = r["client"][:6]  # enough to tell visitors apart on the page
    accounts = {
        "users": _db.run("SELECT COUNT(*) FROM users")[0][0],
        "new_users": _db.run("SELECT COUNT(*) FROM users WHERE created_at > ?", (since,))[0][0],
        "active_users": _db.run("SELECT COUNT(*) FROM users WHERE last_login > ?", (since,))[0][0],
        "watches": _db.run("SELECT COUNT(*) FROM watches")[0][0],
        "alerts": _db.run("SELECT COUNT(*) FROM notifications WHERE kind = 'price_drop' AND created_at > ?",
                          (since,))[0][0],
        "saved_chats": _db.run("SELECT COUNT(*) FROM chats")[0][0],
    }
    feedback = _rows("SELECT at, kind, message, contact FROM feedback WHERE at > ? ORDER BY at DESC LIMIT 20",
                     (since,), "at kind message contact")
    reports = _rows("SELECT at, reason, request, store, title, price, url FROM reports WHERE at > ? "
                    "ORDER BY at DESC LIMIT 20", (since,), "at reason request store title price url")
    # the server's own problems (free quotas used up, every model busy, errors): see app/problems.py
    problems = _rows("SELECT action, COUNT(*), MAX(at) FROM events WHERE kind = 'problem' AND at > ? GROUP BY action "
                     "ORDER BY 3 DESC", (since,), "kind n last")
    problem_log = _rows("SELECT at, action, detail, request FROM events WHERE kind = 'problem' AND at > ? "
                        "ORDER BY at DESC LIMIT 40", (since,), "at kind detail request")
    # LLM calls per step (main._track_llm): one 'turn' row per chat turn with its total, one row per step
    llm_steps = _rows("SELECT action, COUNT(*), SUM(results), SUM(price) FROM events WHERE kind = 'llm' AND at > ? "
                      "GROUP BY action ORDER BY 3 DESC", (since,), "step turns calls tokens")
    llm_by_action = _rows("SELECT detail, COUNT(*), SUM(results), SUM(price) FROM events WHERE kind = 'llm' "
                          "AND action = 'turn' AND at > ? GROUP BY detail ORDER BY 2 DESC", (since,),
                          "action turns calls tokens")
    llm_daily = _rows("SELECT CAST(at + ? AS BIGINT) / 86400 AS d, COUNT(*), SUM(results), SUM(price) FROM events "
                      "WHERE kind = 'llm' AND action = 'turn' AND at > ? GROUP BY d ORDER BY d", (off, since),
                      "day turns calls tokens")
    llm_models = _rows("SELECT detail, SUM(results) FROM events WHERE kind = 'llm' AND action <> 'turn' AND detail <> '' "
                       "AND at > ? GROUP BY detail ORDER BY 2 DESC LIMIT 15", (since,), "model calls")
    for r in llm_steps + llm_by_action + llm_daily + llm_models:  # Postgres SUM can be a Decimal
        for k in ("calls", "tokens"):
            if k in r:
                r[k] = int(r[k] or 0)
    for r in searches:  # Postgres gives AVG as a Decimal, which JSON can't carry
        r["avg_results"] = None if r["avg_results"] is None else round(float(r["avg_results"]), 1)
    return {"days": days, "now": now, "tz_minutes": tz_minutes, "daily": daily, "totals": totals,
            "actions": actions, "searches": searches, "empty_searches": empty, "stores": stores,
            "products": products, "ranks": ranks, "sources": sources, "devices": devices, "recent": recent,
            "accounts": accounts, "feedback": feedback, "reports": reports,
            "problems": problems, "problem_log": problem_log,
            "llm": {"steps": llm_steps, "by_action": llm_by_action, "daily": llm_daily, "models": llm_models},
            "blocked_sites": _db.run("SELECT COUNT(*) FROM blocked_sites WHERE blocked = 1")[0][0]}


# ---------- store promo codes (app/coupons.py) ----------
_COUPON_NAMES = "id store code title link source expires_at added_at"
_COUPON_COLS = ", ".join(_COUPON_NAMES.split())


def add_coupon(market: str, store: str, code: str, title: str, link: str, source: str,
               expires_at: float | None, ext_id: str = "") -> None:
    _db.run("INSERT INTO coupons (market, store, code, title, link, source, ext_id, expires_at, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (market, store, code, title, link, source, ext_id, expires_at, time.time()))


def replace_feed_coupons(market: str, source: str, rows: list[dict]) -> None:
    """A network's whole list, read again: what it no longer lists is gone (ended or withdrawn)."""
    _db.run("DELETE FROM coupons WHERE market = ? AND source = ?", (market, source))
    now = time.time()
    _db.run_many("INSERT INTO coupons (market, store, code, title, link, source, ext_id, expires_at, added_at) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 [(market, r["store"], r["code"], r["title"], r["link"], source, r.get("ext_id", ""),
                   r.get("expires_at"), now) for r in rows])


def live_coupons(market: str, now: float | None = None) -> list[dict]:
    now = now or time.time()
    return _rows(f"SELECT {_COUPON_COLS} FROM coupons WHERE market = ? AND (expires_at IS NULL OR expires_at > ?) "
                 "ORDER BY store, added_at DESC", (market, now), _COUPON_NAMES)


def all_coupons(market: str) -> list[dict]:
    return _rows(f"SELECT {_COUPON_COLS} FROM coupons WHERE market = ? ORDER BY added_at DESC LIMIT 500",
                 (market,), _COUPON_NAMES)


def delete_coupon(coupon_id: int) -> bool:
    return _db.count("DELETE FROM coupons WHERE id = ?", (coupon_id,)) > 0
