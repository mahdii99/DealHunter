"""Warms the live site's answer cache before a busy time (a launch post): runs the requests shoppers are most likely
to type, one after another, the way the chat page sends them. The next shopper writing the same words gets the saved
plan (24 h) and answer (CACHE_HOURS, 6 h by default; prices are still checked again on the store pages), with no
LLM call for the plan and no web search.

    SITE_URL=https://dealhunter-opk9.onrender.com CRON_SECRET=... python -m scripts.prewarm
    python -m scripts.prewarm "سخان غاز 10 لتر" "آيفون 13"      (your own list)

Run it 1-3 hours before the posts go out (the answers last 6 hours). Each new request uses part of the free daily
LLM and web-search quotas (about 5-15 LLM calls): keep the list to the requests that matter. CRON_SECRET (the same
one the price jobs use) keeps these searches out of the per-visitor daily limit and the visitor counts.
Only the standard library: it also runs in the GitHub workflow "prewarm".
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

REQUESTS = [
    "موبايل سامسونج A15", "موبايل سامسونج A55", "آيفون 13", "آيفون 15", "شاومي ريدمي نوت 13", "ريلمي 12",
    "سماعات بلوتوث", "سماعة بلوتوث عزل ضوضاء", "ساعة ذكية", "باور بانك 20000",
    "شاشة سامسونج 55 بوصة", "شاشة 43 بوصة", "تلاجة 18 قدم", "غسالة اطباق", "غسالة توشيبا 10 كيلو",
    "سخان غاز 10 لتر", "سخان كهربا 50 لتر", "قلاية هوائية", "مكنسة روبوت", "تكييف شارب 1.5 حصان",
    "بلايستيشن 5", "لابتوب i7", "لابتوب للمذاكرة", "ايرفراير", "خلاط",
]
TIMEOUT = 300
PAUSE = 20  # seconds between searches: the free models' per-minute limits recover, shoppers on the site come first


def warm(site: str, text: str, secret: str) -> tuple[float, int, bool]:
    body = json.dumps({"messages": [{"role": "user", "text": text}]}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "dealhunter-prewarm"}
    if secret:
        headers["X-Cron-Secret"] = secret
    req = urllib.request.Request(site.rstrip("/") + "/api/chat", data=body, method="POST", headers=headers)
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        d = json.loads(r.read())
    return round(time.monotonic() - t, 1), len(d.get("results") or []), bool(d.get("cached"))


def main(argv: list[str]) -> int:
    site = os.environ.get("SITE_URL") or "https://dealhunter-opk9.onrender.com"
    secret = os.environ.get("CRON_SECRET", "")
    if not secret:
        print("note: no CRON_SECRET: these searches count toward this computer's daily limit")
    texts = [t for t in argv if t.strip()] or REQUESTS
    empty = 0
    for i, text in enumerate(texts):
        try:
            secs, n, cached = warm(site, text, secret)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"{text}: failed ({e})")
            empty += 1
            continue
        empty += n == 0
        print(f"{text}: {n} results in {secs}s" + (" (was already saved)" if cached else ""), flush=True)
        if i < len(texts) - 1 and not cached:
            time.sleep(PAUSE)
    print(f"done: {len(texts) - empty} of {len(texts)} requests have results saved")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
