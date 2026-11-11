"""Shopper accounts: Google sign-in, saved preferences and region, followed products, notifications.

Where it's stored:
  - DATABASE_URL set (a free Supabase Postgres): survives restarts and deploys. Needed on Render, whose free disk
    is wiped on every restart.
  - not set: a local SQLite file, data/accounts.db (your own PC, the tests).

Sign-in: the page shows Google's "Sign in with Google" button (GOOGLE_CLIENT_ID), Google gives it a signed ID token,
we check that token with Google and keep the shopper signed in with a signed cookie (SESSION_SECRET). No passwords
are ever stored here.

Price-drop alerts: a shopper follows a product (one product page at one store). Whenever we see that page's price
again lower than the last price we saw (in anyone's search, or in the scheduled re-check, see main.check_watches),
the shopper gets a notification on the site, and an email when they allowed it and BREVO_API_KEY is set.
"""
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger("uvicorn.error")

SQLITE_PATH = Path(os.getenv("ACCOUNTS_DB", Path(__file__).resolve().parent.parent / "data" / "accounts.db"))
COOKIE = "dh_session"
SESSION_DAYS = 30
MAX_WATCHES = 50          # followed products per shopper
REGIONS = ("EG", "SA", "AE")
# Preferences a shopper can save, with their defaults. Unknown keys sent by the page are ignored.
PREFS = {
    "hide_unsure_shops": False,  # show only famous / trusted shops (trust 60+)
}

_SCHEMA = """
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
CREATE INDEX IF NOT EXISTS notifications_user ON notifications (user_id, created_at);
"""


class _Store:
    """The same few queries on SQLite or Postgres. Queries are written with ? placeholders."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conn = None
        self._key = None

    def _connect(self):
        url = os.getenv("DATABASE_URL", "")
        key = url or str(SQLITE_PATH)
        if self._conn is not None and self._key == key:
            return self._conn
        self.close()  # the database setting changed (tests): don't leave the old connection open
        if url:
            import psycopg  # only needed with Postgres
            # prepare_threshold=None: Supabase's pooler can't keep prepared statements between connections
            conn = psycopg.connect(url, autocommit=True, prepare_threshold=None, connect_timeout=10)
            schema = _SCHEMA.format(ID="BIGSERIAL PRIMARY KEY", FLOAT="DOUBLE PRECISION")
            for stmt in filter(str.strip, schema.split(";")):
                conn.execute(stmt)
        else:
            SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False, isolation_level=None)
            conn.executescript(_SCHEMA.format(ID="INTEGER PRIMARY KEY", FLOAT="REAL"))
        self._conn, self._key, self.pg = conn, key, bool(url)
        return conn

    def run(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            for attempt in (1, 2):
                conn = self._connect()
                try:
                    cur = conn.execute(sql.replace("?", "%s") if self.pg else sql, args)
                    return cur.fetchall() if cur.description else []
                except sqlite3.Error:
                    raise
                except Exception as e:  # a Postgres connection dropped by the server: connect again once
                    if attempt == 2 or not self.pg or "Operational" not in type(e).__name__:
                        raise
                    log.warning("accounts db: reconnecting after %s", type(e).__name__)
                    self.close()
        return []

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


_db = _Store()


def close() -> None:
    _db.close()


# ---------- sign-in ----------

class AuthError(Exception):
    pass


def google_client_id() -> str:
    return os.getenv("GOOGLE_CLIENT_ID", "").strip()


def enabled() -> bool:
    return bool(google_client_id())


async def verify_google(credential: str, client: httpx.AsyncClient | None = None) -> dict:
    """Google's ID token from the sign-in button -> {"id", "email", "name", "picture"}. Google checks the signature;
    we check it was made for OUR site (aud), by Google (iss), is still valid and the email is verified."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        r = await client.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": credential})
    except httpx.HTTPError as e:
        raise AuthError(f"google unreachable: {type(e).__name__}")
    finally:
        if own:
            await client.aclose()
    if r.status_code != 200:
        raise AuthError("invalid token")
    d = r.json()
    if d.get("aud") != google_client_id():
        raise AuthError("token for another site")
    if d.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise AuthError("not issued by google")
    if int(d.get("exp", 0)) < time.time():
        raise AuthError("token expired")
    if str(d.get("email_verified")).lower() != "true" or not d.get("email") or not d.get("sub"):
        raise AuthError("email not verified")
    return {"id": "g:" + d["sub"], "email": d["email"], "name": d.get("name") or d["email"].split("@")[0],
            "picture": d.get("picture") or ""}


def _secret() -> bytes:
    s = os.getenv("SESSION_SECRET", "")
    if not s:  # fine on your own PC; on a server set SESSION_SECRET or everyone is signed out on each restart
        s = _secret.fallback  # type: ignore[attr-defined]
    return s.encode()


_secret.fallback = hashlib.sha256(os.urandom(32)).hexdigest()  # type: ignore[attr-defined]


def make_session(user_id: str, now: float | None = None) -> str:
    exp = int((now or time.time()) + SESSION_DAYS * 86400)
    body = f"{user_id}|{exp}"
    return body + "|" + hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()


def read_session(token: str | None) -> str | None:
    """The signed cookie -> the user id, or None when missing, edited or expired."""
    if not token or token.count("|") != 2:
        return None
    user_id, exp, sig = token.split("|")
    good = hmac.new(_secret(), f"{user_id}|{exp}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, good) or not exp.isdigit() or int(exp) < time.time():
        return None
    return user_id


# ---------- users and preferences ----------

_USER_COLS = "id, email, name, picture, market, email_alerts, prefs, created_at"


def _user(row) -> dict:
    uid, email, name, picture, mk, alerts, prefs, created = row
    saved = json.loads(prefs or "{}")
    return {"id": uid, "email": email, "name": name, "picture": picture, "market": mk,
            "email_alerts": bool(alerts), "prefs": {k: saved.get(k, v) for k, v in PREFS.items()},
            "created_at": created}


def upsert_user(info: dict) -> dict:
    """First sign-in creates the account (region not chosen yet); later ones refresh the name and photo."""
    now = time.time()
    _db.run("INSERT INTO users (id, email, name, picture, created_at, last_login) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET email = excluded.email, name = excluded.name, "
            "picture = excluded.picture, last_login = excluded.last_login",
            (info["id"], info["email"][:200], info["name"][:100], info["picture"][:500], now, now))
    return get_user(info["id"])


def get_user(user_id: str | None) -> dict | None:
    if not user_id:
        return None
    rows = _db.run(f"SELECT {_USER_COLS} FROM users WHERE id = ?", (user_id,))
    return _user(rows[0]) if rows else None


def update_user(user_id: str, market: str | None = None, email_alerts: bool | None = None,
                prefs: dict | None = None) -> dict | None:
    user = get_user(user_id)
    if not user:
        return None
    if market is not None:
        if market not in REGIONS:
            raise ValueError("unknown region")
        _db.run("UPDATE users SET market = ? WHERE id = ?", (market, user_id))
    if email_alerts is not None:
        _db.run("UPDATE users SET email_alerts = ? WHERE id = ?", (int(bool(email_alerts)), user_id))
    if prefs:
        merged = {**user["prefs"], **{k: type(PREFS[k])(v) for k, v in prefs.items() if k in PREFS}}
        _db.run("UPDATE users SET prefs = ? WHERE id = ?", (json.dumps(merged), user_id))
    return get_user(user_id)


def delete_user(user_id: str) -> None:
    """Everything we keep about a shopper goes (the "delete my account" button)."""
    for table, col in (("notifications", "user_id"), ("watches", "user_id"), ("users", "id")):
        _db.run(f"DELETE FROM {table} WHERE {col} = ?", (user_id,))


# ---------- followed products ----------

_WATCH_COLS = "id, url, title, store, image, market, first_price, price, target_price, created_at"


def _watch(row) -> dict:
    return dict(zip(("id", "url", "title", "store", "image", "market", "first_price", "price", "target_price",
                     "created_at"), row))


def add_watch(user_id: str, url: str, title: str, store: str, price: float, market: str, image: str = "",
              target_price: float | None = None) -> dict:
    if not get_watch(user_id, url) and len(list_watches(user_id)) >= MAX_WATCHES:
        raise ValueError("too many")
    now = time.time()
    _db.run("INSERT INTO watches (user_id, url, title, store, image, market, first_price, price, target_price, "
            "created_at, checked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (user_id, url) DO UPDATE SET target_price = excluded.target_price",
            (user_id, url[:1000], title[:300], store[:100], (image or "")[:1000], market, price, price,
             target_price, now, now))
    return get_watch(user_id, url)


def get_watch(user_id: str, url: str) -> dict | None:
    rows = _db.run(f"SELECT {_WATCH_COLS} FROM watches WHERE user_id = ? AND url = ?", (user_id, url))
    return _watch(rows[0]) if rows else None


def list_watches(user_id: str) -> list[dict]:
    return [_watch(r) for r in _db.run(f"SELECT {_WATCH_COLS} FROM watches WHERE user_id = ? ORDER BY created_at DESC",
                                       (user_id,))]


def remove_watch(user_id: str, watch_id: int) -> bool:
    found = _db.run("SELECT id FROM watches WHERE id = ? AND user_id = ?", (watch_id, user_id))
    _db.run("DELETE FROM watches WHERE id = ? AND user_id = ?", (watch_id, user_id))
    return bool(found)


def urls_to_check(older_than: float, limit: int) -> list[str]:
    """Followed product pages not seen for a while (oldest first), for the scheduled re-check."""
    rows = _db.run("SELECT url, MIN(checked_at) AS c FROM watches GROUP BY url HAVING MIN(checked_at) < ? "
                   "ORDER BY c LIMIT ?", (time.time() - older_than, limit))
    return [r[0] for r in rows]


def _money(n: float, mk: str) -> str:
    cur = {"EG": "ج.م", "SA": "ر.س", "AE": "د.إ"}.get(mk, "")
    return f"{n:,.0f} {cur}".strip()


def prices_seen(results: list[dict]) -> list[dict]:
    """New prices of product pages (from a search or the re-check). Each follower whose last seen price is higher
    gets a notification (when at/below their target, if they set one). Returns the notifications made."""
    made = []
    now = time.time()
    urls = list({r["url"] for r in results if r.get("url") and r.get("price")})
    if not urls:
        return made
    # one query to find which of these pages anyone follows (most searches: none), not one per result
    followed = {row[0] for row in _db.run(f"SELECT DISTINCT url FROM watches WHERE url IN ({','.join('?' * len(urls))})",
                                          tuple(urls))}
    for r in results:
        url, price = r.get("url"), r.get("price")
        if url not in followed or not price or r.get("in_stock") is False:
            continue
        rows = _db.run("SELECT w.id, w.user_id, w.title, w.store, w.market, w.price, w.target_price, w.first_price, "
                       "u.email, u.name, u.email_alerts FROM watches w JOIN users u ON u.id = w.user_id "
                       "WHERE w.url = ?", (url,))
        for wid, uid, title, store, mk, last, target, first, email, name, alerts in rows:
            if price < last - 0.5 and (target is None or price <= target):
                body = f"نزل من {_money(last, mk)} لـ {_money(price, mk)} في {store}"
                if first > price:
                    body += f" (أقل بـ {round((first - price) / first * 100)}% من وقت ما تابعته)"
                n = notify(uid, "price_drop", f"السعر نزل: {title[:80]}", body, url)
                made.append(n | {"email": email, "name": name, "email_alerts": bool(alerts)})
            _db.run("UPDATE watches SET price = ?, checked_at = ? WHERE id = ?", (price, now, wid))
    return made


def checked(url: str) -> None:
    """The re-check looked at this page (even if no price was found), so it isn't retried right away."""
    _db.run("UPDATE watches SET checked_at = ? WHERE url = ?", (time.time(), url))


# ---------- notifications ----------

def notify(user_id: str, kind: str, title: str, body: str, url: str = "") -> dict:
    now = time.time()
    _db.run("INSERT INTO notifications (user_id, kind, title, body, url, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, kind, title[:200], body[:500], url[:1000], now))
    return {"user_id": user_id, "kind": kind, "title": title, "body": body, "url": url, "created_at": now}


def list_notifications(user_id: str, limit: int = 30) -> list[dict]:
    rows = _db.run("SELECT id, kind, title, body, url, created_at, seen FROM notifications WHERE user_id = ? "
                   "ORDER BY created_at DESC LIMIT ?", (user_id, limit))
    return [dict(zip(("id", "kind", "title", "body", "url", "created_at", "seen"), r[:6] + (bool(r[6]),)))
            for r in rows]


def unseen_count(user_id: str) -> int:
    return _db.run("SELECT COUNT(*) FROM notifications WHERE user_id = ? AND seen = 0", (user_id,))[0][0]


def mark_seen(user_id: str) -> None:
    _db.run("UPDATE notifications SET seen = 1 WHERE user_id = ? AND seen = 0", (user_id,))


def users_in(market: str) -> list[str]:
    return [r[0] for r in _db.run("SELECT id FROM users WHERE market = ?", (market,))]


# ---------- email (optional) ----------

def email_enabled() -> bool:
    return bool(os.getenv("BREVO_API_KEY") and os.getenv("MAIL_FROM"))


async def send_email(client: httpx.AsyncClient, to: str, name: str, subject: str, text: str, url: str = "") -> bool:
    """Through Brevo's free plan (300 emails/day, no card). Render's free plan blocks SMTP ports, so Gmail SMTP
    won't work there; an HTTP email API does."""
    if not email_enabled():
        return False
    link = f'<p><a href="{url}">افتح صفحة المنتج</a></p>' if url else ""
    html = (f'<div dir="rtl" style="font-family:Tahoma,sans-serif"><p>أهلاً {_esc(name)}،</p>'
            f'<p>{_esc(text)}</p>{link}<p style="color:#888">DealHunter - تقدر تقفل إيميلات التنبيه من الإعدادات في الموقع.</p></div>')
    try:
        r = await client.post("https://api.brevo.com/v3/smtp/email", timeout=15,
                              headers={"api-key": os.environ["BREVO_API_KEY"], "accept": "application/json"},
                              json={"sender": {"name": "DealHunter", "email": os.environ["MAIL_FROM"]},
                                    "to": [{"email": to, "name": name}], "subject": subject, "htmlContent": html})
    except httpx.HTTPError as e:
        log.warning("email to user failed: %s", type(e).__name__)
        return False
    if r.status_code >= 300:
        log.warning("email failed: %s %s", r.status_code, r.text[:200])
    return r.status_code < 300


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
