"""Live test: real chat conversations (with follow-ups) across many kinds of products, saved as a spreadsheet.

    python -m scripts.live_test              # all conversations
    python -m scripts.live_test 3 7          # only conversations 3 and 7 (e.g. re-check after a fix)

Each turn goes through the same code as the chat page (/api/chat), without the cache, so every turn is a real search.
Report: debug/live_test/<date-time>.xlsx (sheet "turns": one row per message, sheet "results": one row per offer).
Without openpyxl it writes two .csv files instead (they open in Excel too).
About 7 web searches and 4-12 LLM calls per search turn: the full run uses a real part of the free quotas.
"""
import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime

from app import main  # noqa: F401  (loads .env)

CONVERSATIONS = [
    ["سخان اوليمبيك هيرو 50 لتر بكرة"],
    ["سماعة انكر ساوند كور R50i NC", "اول نتيجة دي اصلية ولا ممكن تكون تقليد؟"],
    ["عايز تيشرتات بيزك سادة ماركة نضيفة", "عايز ماركة زي زارا او جاك اند جونس"],
    ["موبايل سامسونج بطاريته كبيرة تحت 10000", "طب شاومي بس"],
    ["لابتوب للبرمجة تحت 30 الف"],
    ["iphone 15 128gb", "الاسود بس"],
    ["كوتشي نايك رجالي مقاس 43"],
    ["برفان ديور سوفاج 100 مل"],
    ["كريم مرطب سيرافي للبشرة الجافة"],
    ["غسالة فول اوتوماتيك 8 كيلو", "توشيبا او شارب بس"],
    ["تكييف شارب 1.5 حصان بارد ساخن"],
    ["شاشة سامسونج 55 بوصة سمارت 4K"],
    ["playstation 5 slim", "النسخة الديجيتال"],
    ["كرسي مكتب طبي مريح"],
    ["حفاضات بامبرز مقاس 4"],
    ["زيت عربية موبيل 1 5w30 4 لتر"],
    ["مكنسة كهربائية 2000 وات"],
    ["ساعة شاومي سمارت"],
    ["رواية الفيل الأزرق احمد مراد"],
    ["شنطة لابتوب 15.6 بوصة ضد المية"],
]


def remember(d: dict) -> str:
    """What the assistant 'said', exactly like the chat page keeps it (static/index.html remember())."""
    if d.get("action") == "ask":
        return d.get("question") or ""
    if d.get("action") == "answer":
        return d.get("answer") or ""
    top = "; ".join(f"{i + 1}) {p['title']} - {p['price']} EGP ({p['store']})" for i, p in enumerate(d["results"][:5]))
    return " ".join(x for x in [d.get("understood"), d.get("summary"), top and "Shown: " + top] if x)[:1500]


async def run_conversation(n: int, messages: list[str], turns: list[dict], offers: list[dict]) -> None:
    history, request = [], ""
    for k, text in enumerate(messages, 1):
        history.append({"role": "user", "text": text})
        t = time.monotonic()
        (main.ROOT / "debug" / "last_judge.json").unlink(missing_ok=True)  # so a turn without a judge call has none
        logging.getLogger("uvicorn.error").info("##### TURN %s.%s: %s", n, k, text)
        try:
            d = await main._chat_turn(main.ChatTurn(messages=history, request=request))
            err = ""
        except Exception as e:  # keep going: one failure shouldn't stop the report
            d, err = {"action": "error", "results": []}, f"{type(e).__name__}: {e}"
        secs = round(time.monotonic() - t, 1)
        res = d.get("results") or []
        if d.get("action") == "search":
            request = d.get("request") or request
        history.append({"role": "assistant", "text": remember(d) if not err else err})
        cheapest = min(res, key=lambda r: r["price"]) if res else None
        turns.append({
            "conv": n, "turn": k, "message": text, "action": d.get("action"), "seconds": secs,
            "understood": d.get("understood", ""), "answer_or_question": d.get("answer") or d.get("question") or "",
            "queries": " || ".join(d.get("queries") or []), "requirements": " | ".join(d.get("requirements") or []),
            "results": len(res), "stores": len({r["store"] for r in res}), "checked": d.get("candidates_checked", ""),
            "unsure": sum(1 for r in res if r.get("unconfirmed")),
            "cheapest_price": cheapest["price"] if cheapest else "", "cheapest_store": cheapest["store"] if cheapest else "",
            "cheapest_title": cheapest["title"][:100] if cheapest else "",
            "summary": d.get("summary", ""), "failed_sites": len(d.get("store_errors") or {}), "error": err,
            **{f"t_{a}": b for a, b in (d.get("timings") or {}).items()},
            "judge_model": ((d.get("llm_usage") or {}).get("by_step") or {}).get("judge", {}).get("model", ""),
            "llm_calls": (d.get("llm_usage") or {}).get("calls", ""), "llm_tokens": (d.get("llm_usage") or {}).get("tokens", ""),
            **{f"tok_{a}": b["tokens"] for a, b in ((d.get("llm_usage") or {}).get("by_step") or {}).items()},
        })
        for r in res:
            offers.append({"conv": n, "turn": k, "message": text, "store": r["store"], "price": r["price"],
                           "old_price": r.get("old_price") or "", "title": r["title"], "note": r.get("note") or "",
                           "unsure": "yes" if r.get("unconfirmed") else "", "best_deal": "yes" if r.get("best_deal") else "",
                           "alternative": "yes" if r.get("alternative") else "", "trust": r.get("trust_score", ""),
                           "url": r["url"]})
        _keep_judge(n, k, d)
        degraded = bool(err) or d.get("mode") == "rules" or "الموديلات المجانية مشغولة" in (d.get("summary") or d.get("answer") or "")
        _QUOTA["bad_in_a_row"] = _QUOTA["bad_in_a_row"] + 1 if degraded else 0
        turns[-1]["llm_degraded"] = "yes" if degraded else ""
        print(f"[{n}.{k}] {text}: {d.get('action')} {len(res)} results in {secs:.0f}s"
              + (f" cheapest {cheapest['price']} @ {cheapest['store']}" if cheapest else "") + (f" ({err})" if err else ""))


_QUOTA = {"bad_in_a_row": 0}
STOP_AFTER_DEGRADED = 2  # turns in a row without the LLM (free quotas gone): stop instead of burning web searches
RUN_DIR = None  # debug/live_test/<time>/: the server log and each turn's judge exchange, to see WHY a result came out


def _keep_judge(n: int, k: int, d: dict) -> None:
    """last_judge.json is overwritten every search: keep a copy per turn (only if this turn wrote it)."""
    src = main.ROOT / "debug" / "last_judge.json"
    if RUN_DIR is None or d.get("action") != "search" or not src.exists():
        return
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
        (RUN_DIR / f"judge_{n}.{k}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except (OSError, ValueError):
        pass


def save(turns: list[dict], offers: list[dict]) -> str:
    out = main.ROOT / "debug" / "live_test"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{datetime.now():%Y-%m-%d_%H-%M}"
    sheets = {"turns": turns, "results": offers}
    try:
        from openpyxl import Workbook
    except ImportError:
        for name, rows in sheets.items():
            with open(f"{stem}_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Arabic in Excel
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)) or ["empty"])
                w.writeheader()
                w.writerows(rows)
        return f"{stem}_turns.csv + {stem}_results.csv"
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        ws.sheet_view.rightToLeft = True
        cols = list(dict.fromkeys(k for r in rows for k in r))
        ws.append(cols)
        for r in rows:
            ws.append([r.get(c, "") for c in cols])
        ws.freeze_panes = "A2"
        for i, c in enumerate(cols, 1):
            width = max([len(str(c))] + [len(str(r.get(c, ""))) for r in rows[:200]])
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(8, width + 2), 60)
    wb.save(f"{stem}.xlsx")
    return f"{stem}.xlsx"


async def run(picked: list[int]) -> None:
    global RUN_DIR
    main.CACHE_TTL = 0  # every turn is a real search
    RUN_DIR = main.ROOT / "debug" / "live_test" / f"{datetime.now():%Y-%m-%d_%H-%M}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(RUN_DIR / "server.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    server_log = logging.getLogger("uvicorn.error")
    server_log.addHandler(handler)
    server_log.setLevel(logging.INFO)
    print("server log + judge files:", RUN_DIR)
    turns, offers = [], []
    for n, conv in enumerate(CONVERSATIONS, 1):
        if _QUOTA["bad_in_a_row"] >= STOP_AFTER_DEGRADED:
            print(f"STOPPED before conversation {n}: {STOP_AFTER_DEGRADED} turns in a row without the LLM "
                  "(free quotas used up), so the rest would only burn web searches")
            break
        if not picked or n in picked:
            await run_conversation(n, conv, turns, offers)
    searches = [t for t in turns if t["action"] == "search"]
    if searches:
        print(f"\nDONE: {len(turns)} turns, {len(searches)} searches, avg {sum(t['seconds'] for t in searches) / len(searches):.0f}s, "
              f"avg {sum(t['results'] for t in searches) / len(searches):.1f} results, "
              f"{sum(1 for t in searches if not t['results'])} with no results")
    print("report:", save(turns, offers))


if __name__ == "__main__":
    asyncio.run(run([int(x) for x in sys.argv[1:]]))
