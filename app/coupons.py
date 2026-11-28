"""Store promo codes: only real ones, from a source that has the right to give them.

Where coupon apps get their codes: the stores hand them to affiliate networks (Admitad, ArabClicks, Boostiny, DCM...)
for their partners to publish, and the network pays the partner when a shopper uses the code. So a code is shown
here only when it comes from:
  - Admitad's coupon feed for our website (ADMITAD_CLIENT_ID / ADMITAD_CLIENT_SECRET / ADMITAD_WEBSITE_ID, all from
    the Admitad account; read after each price run by .github/workflows/scrape-prices.yml), or
  - the owner, from /admin: a code a network gave us (ArabClicks / Boostiny send theirs by dashboard or email) or one
    the store itself publishes on its own site. The owner writes where it came from, and the page shows it.
Codes copied from other coupon sites are never added: we can't know they work, and they aren't ours to give.
An expired code is never shown.
"""
import asyncio
import base64
import logging
import os
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx

from . import affiliate, db, market

log = logging.getLogger("uvicorn.error")

ADMITAD = "https://api.admitad.com"
FEED_SOURCE = "admitad"
NO_CODE = {"", "not required", "not needed", "no code", "none"}
STORE_NAMES = {"amazon.eg": "أمازون", "noon.com": "نون", "jumia.com.eg": "جوميا", "btech.com": "بي تك",
               "2b.com.eg": "2B", "shein.com": "شي إن", "namshi.com": "نمشي", "carrefouregypt.com": "كارفور"}


def store_name(domain: str) -> str:
    return STORE_NAMES.get(domain, domain)


def domain_of(url_or_domain: str) -> str:
    text = (url_or_domain or "").strip().lower()
    host = urlsplit(text if "://" in text else "https://" + text).hostname or ""
    return host.removeprefix("www.")


def _when(text) -> float | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def from_admitad(item: dict, now: float | None = None) -> dict | None:
    """One coupon of Admitad's feed as our row, or None when it isn't usable (ended, no link, no store)."""
    now = now or time.time()
    camp = item.get("campaign") or {}
    store = domain_of(camp.get("site_url") or "")
    link = item.get("goto_link") or ""
    ends = _when(item.get("date_end"))
    if not store or not link.startswith("https://") or (ends and ends <= now):
        return None
    code = str(item.get("promocode") or "").strip()
    if code.lower() in NO_CODE:
        code = ""
    title = " ".join(str(item.get("name") or item.get("description") or "").split())[:200]
    if item.get("discount") and str(item["discount"]) not in title:
        title = f"{title} ({item['discount']})" if title else str(item["discount"])
    if not title:
        return None
    return {"store": store, "code": code[:60], "title": title, "link": link, "expires_at": ends,
            "ext_id": str(item.get("id") or "")}


async def read_admitad(client: httpx.AsyncClient | None = None) -> list[dict] | None:
    """Our website's coupons from Admitad, for this market's country. None = not set up (or the network failed)."""
    cid, secret, site = (os.getenv(k, "").strip() for k in
                         ("ADMITAD_CLIENT_ID", "ADMITAD_CLIENT_SECRET", "ADMITAD_WEBSITE_ID"))
    if not (cid and secret and site.isdigit()):
        return None
    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()
        r = await client.post(f"{ADMITAD}/token/", headers={"Authorization": f"Basic {basic}"},
                              data={"grant_type": "client_credentials", "client_id": cid,
                                    "scope": "coupons_for_website"})
        r.raise_for_status()
        token = r.json()["access_token"]
        out, offset = [], 0
        while offset < 2000:
            r = await client.get(f"{ADMITAD}/coupons/website/{site}/",
                                 params={"region": market.current().code, "limit": 500, "offset": offset},
                                 headers={"Authorization": f"Bearer {token}"})
            r.raise_for_status()
            page = r.json().get("results") or []
            out += [c for c in (from_admitad(i) for i in page) if c]
            if len(page) < 500:
                break
            offset += 500
        return out
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("coupons: Admitad feed failed: %s", type(e).__name__)
        return None
    finally:
        if own:
            await client.aclose()


async def refresh() -> dict:
    """Reads the networks' feeds again (the cron job). The owner's own codes stay as they are."""
    got = await read_admitad()
    if got is None:
        return {"admitad": "off" if not os.getenv("ADMITAD_WEBSITE_ID") else "failed"}
    await asyncio.to_thread(db.replace_feed_coupons, market.current().code, FEED_SOURCE, got)
    return {"admitad": len(got)}


def source_label(source: str) -> str:
    if source == FEED_SOURCE:
        return "من شبكة Admitad الرسمية للمتجر"
    where = source.partition(":")[2].strip()
    return f"من {where}" if where else "من المتجر"


def current() -> list[dict]:
    """Codes that still work, with the link the shopper opens (ours on it when the store has an affiliate code)."""
    out = []
    for c in db.live_coupons(market.current().code):
        link = c["link"] if c["source"] == FEED_SOURCE else affiliate.link(c["link"])
        out.append({**c, "link": link, "store_name": store_name(c["store"]), "from": source_label(c["source"])})
    return out
