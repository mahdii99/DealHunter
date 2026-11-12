"""The owner's private /admin page: how many people visit, what they ask for, which store links they open.

Off unless ADMIN_PASSWORD is set (in Render's environment, never in the code: the repository is public). Signing in
with it gives a signed cookie for ADMIN_DAYS days; changing the password signs every browser out. Too many wrong
passwords from one visitor lock the form for a while.
"""
import hashlib
import hmac
import os
import time

COOKIE = "dh_admin"
ADMIN_DAYS = 7
MAX_TRIES = 8          # wrong passwords per visitor per TRY_WINDOW
TRY_WINDOW = 3600
_tries: dict[str, list[float]] = {}


def enabled() -> bool:
    return bool(password())


def password() -> str:
    return os.getenv("ADMIN_PASSWORD", "")


def _key() -> bytes:
    # the password is part of the key: a new password makes every old cookie invalid
    return hashlib.sha256(f"admin|{password()}|{os.getenv('SESSION_SECRET', '')}".encode()).digest()


def make_cookie(now: float | None = None) -> str:
    exp = str(int((now or time.time()) + ADMIN_DAYS * 86400))
    return exp + "|" + hmac.new(_key(), exp.encode(), hashlib.sha256).hexdigest()


def is_admin(token: str | None) -> bool:
    if not enabled() or not token or token.count("|") != 1:
        return False
    exp, sig = token.split("|")
    good = hmac.new(_key(), exp.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time()


def locked(client: str) -> bool:
    now = time.time()
    recent = [t for t in _tries.get(client, []) if now - t < TRY_WINDOW]
    _tries[client] = recent
    return len(recent) >= MAX_TRIES


def check(given: str, client: str) -> bool:
    """The password typed on the form. A wrong one counts towards the lock."""
    if enabled() and hmac.compare_digest(given.encode(), password().encode()):
        _tries.pop(client, None)
        return True
    if len(_tries) > 10_000:
        _tries.clear()
    _tries.setdefault(client, []).append(time.time())
    return False
