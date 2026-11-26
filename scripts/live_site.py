"""Times real searches on the live site, the way the chat page sees them (the streamed /api/chat/stream answer).

    SITE_URL=https://dealhunter-opk9.onrender.com python -m scripts.live_site "غسالة توشيبا 10 كيلو" "ريلمي 12 برو"

For each request it prints when the first checked offers came, when the 25-second answer came, when the full
answer came, the results, the LLM calls per step, the time per step and the stores that failed. Only the standard
library: it also runs in the GitHub workflow "live search check" (the site's free server can't be reached from
everywhere). A request that is in the site's answer cache comes back at once: use new words to time a real search.
"""
import json
import os
import sys
import time
import urllib.request

TIMEOUT = 300


def search(site: str, text: str) -> dict:
    body = json.dumps({"messages": [{"role": "user", "text": text}]}).encode()
    req = urllib.request.Request(site.rstrip("/") + "/api/chat/stream", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "dealhunter-live-check"})
    start, seen = time.monotonic(), {}
    final = None
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            if not raw.strip():
                continue
            e = json.loads(raw)
            kind = e.get("type")
            at = round(time.monotonic() - start, 1)
            if kind == "status":
                seen.setdefault("first_status", at)
            elif kind == "partial":
                seen.setdefault("first_partial", at)
                seen.setdefault("first_partial_results", len((e.get("data") or {}).get("results") or []))
            elif kind == "done" and (e.get("data") or {}).get("pending"):
                seen["interim"] = at
                seen["interim_results"] = len(e["data"].get("results") or [])
            elif kind in ("done", "update"):
                seen["full"] = at
                final = e.get("data") or {}
            elif kind == "error":
                seen["error"] = e.get("status")
    return {"seen": seen, "final": final or {}}


def report(text: str, got: dict) -> str:
    s, d = got["seen"], got["final"]
    used = d.get("llm_usage") or {}
    lines = [f"== {text}",
             "  first status {first_status}s | first offers {first_partial}s ({first_partial_results}) | "
             "25s answer {interim}s ({interim_results}) | full {full}s".format(**{
                 k: s.get(k, "-") for k in ("first_status", "first_partial", "first_partial_results", "interim",
                                            "interim_results", "full")}),
             (f"  action {d.get('action')} | cached {d.get('cached')} | results {len(d.get('results') or [])} | "
              f"checked {d.get('candidates_checked')} | error {s.get('error', '-')}"),
             f"  LLM calls {used.get('calls')}: " + ", ".join(
                 f"{k} {v.get('calls')}" for k, v in (used.get("by_step") or {}).items()),
             "  failed models: " + ("; ".join(f"{k}: " + ", ".join(f"{m} x{n}" for m, n in v["fails"].items())
                                          for k, v in (used.get("by_step") or {}).items() if v.get("fails")) or "-"),
             "  time per step: " + ", ".join(f"{k} {v}" for k, v in (d.get("timings") or {}).items()),
             f"  store errors: {d.get('store_errors') or '-'}",
             "  store pages read: %s without the LLM, %s by the LLM" % (
                 (d.get("page_reads") or {}).get("without_llm", "?"), (d.get("page_reads") or {}).get("llm", "?")),
             "  pages the LLM read: " + (", ".join(d.get("llm_pages") or []) or "-")]
    for p in (d.get("results") or [])[:8]:
        lines.append(f"    {p.get('price')} | {p.get('store')} | {(p.get('title') or '')[:70]}")
    noon = d.get("noon") or []
    lines.append(f"  noon links (no price): {len(noon)}")
    for n in noon:
        lines.append(f"    noon | {(n.get('title') or '')[:70]}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    site = os.environ.get("SITE_URL") or "https://dealhunter-opk9.onrender.com"
    texts = [t for t in argv if t.strip()]
    if not texts:
        print(__doc__)
        return 2
    try:  # the free server sleeps: wake it first, so the first search isn't timed with the wake-up
        urllib.request.urlopen(site.rstrip("/") + "/health", timeout=120).read()
    except OSError as e:  # (urllib's errors and timeouts)
        print("wake-up:", e)
    for text in texts:
        try:
            print(report(text, search(site, text)), flush=True)
        except (OSError, ValueError) as e:  # the site down, a timeout or a broken line
            print(f"== {text}\n  failed: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
