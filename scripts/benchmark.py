"""Runs a fixed set of real requests and saves a report, so we can tell if a change made results better or worse.

    python -m scripts.benchmark            # all requests
    python -m scripts.benchmark 5          # only the first 5 (saves free quota)

Report: debug/benchmark/<date-time>.csv (open in Excel) + a summary line in the terminal.
Each request uses about 4 web searches and 3-8 LLM calls, so 20 requests use a real part of the free daily quota.
"""
import asyncio
import csv
import sys
import time
from datetime import datetime

from app import main  # noqa: F401  (loads .env)
from app import agent

REQUESTS = [
    "سخان اوليمبك هيرو 50 لتر بكرة",
    "سماعة انكر ساوند كور R50i NC",
    "تلاجه 18 قدم نوع كويس لون سيلفر",
    "غسالة فول اوتوماتيك 8 كيلو",
    "باور بانك 20000 مللي امبير شحن سريع",
    "iphone 15 128gb",
    "samsung galaxy a55",
    "شاشة سامسونج 55 بوصة سمارت 4K",
    "لابتوب للبرمجة تحت 30 الف",
    "مكنسة كهربائية 2000 وات",
    "خلاط مولينكس",
    "airpods pro 2",
    "ماوس لوجيتك لاسلكي",
    "تكييف شارب 1.5 حصان بارد ساخن",
    "كوتشي نايك رجالي مقاس 43",
    "ساعة شاومي سمارت",
    "playstation 5 slim",
    "ميكروويف 25 لتر",
    "شاحن انكر 65 وات",
    "مروحة ستاند توشيبا",
]


async def bench(requests: list[str]) -> None:
    out_dir = main.ROOT / "debug" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y-%m-%d_%H-%M}.csv"
    rows, total_time = [], 0.0
    for i, q in enumerate(requests, 1):
        t = time.monotonic()
        try:
            r = await agent.run(q)
            err = ""
        except Exception as e:  # keep going: one failure shouldn't stop the report
            r, err = {"results": [], "store_errors": {}}, f"{type(e).__name__}: {e}"
        secs = time.monotonic() - t
        total_time += secs
        res = r.get("results", [])
        cheapest = min(res, key=lambda x: x["price"]) if res else None
        rows.append({
            "request": q, "seconds": round(secs, 1), "results": len(res),
            "stores": len({x["store"] for x in res}),
            "cheapest_price": cheapest["price"] if cheapest else "", "cheapest_store": cheapest["store"] if cheapest else "",
            "cheapest_title": cheapest["title"][:80] if cheapest else "",
            "unsure": sum(1 for x in res if x.get("unconfirmed")), "failed_sites": len(r.get("store_errors", {})),
            "error": err,
            **{f"t_{k}": v for k, v in (r.get("timings") or {}).items()},
        })
        print(f"[{i}/{len(requests)}] {q}: {len(res)} results in {secs:.0f}s {r.get('timings') or ''}"
              + (f" ({err})" if err else ""))
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel shows Arabic correctly
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    empty = sum(1 for r in rows if not r["results"])
    print(f"\nDONE: {len(rows)} requests, avg {total_time / len(rows):.0f}s, "
          f"avg {sum(r['results'] for r in rows) / len(rows):.1f} results, {empty} with no results")
    print(f"report: {path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(REQUESTS)
    asyncio.run(bench(REQUESTS[:n]))
