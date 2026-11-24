"""Problems the owner should know about (a free LLM or search quota used up, every model busy, a server error):
logged and saved for the /admin page, never shown to the shopper as they are. The shopper only sees a short,
friendly message.

note() can be called from anywhere (inside or outside an async request) and never raises: bookkeeping must not
break a search. The same problem is saved at most once every few minutes, so a busy hour doesn't flood the table."""
import asyncio
import logging
import time

log = logging.getLogger("uvicorn.error")
REPEAT_SECONDS = 300  # the same (kind, detail) is saved again only after this long
_last: dict[tuple[str, str], float] = {}

KINDS = {  # for the admin page
    "llm_busy": "كل موديلات الـ LLM مشغولة أو بطيئة",
    "llm_quota": "موديل خلص الحد اليومي المجاني",
    "llm_budget": "كل الموديلات وصلت حصتها اليومية (أو حصة شغل الخلفية)",
    "search_error": "مشكلة في خدمة البحث",
    "server_error": "خطأ في السيرفر",
}


def _save(kind: str, detail: str, request: str) -> None:
    from . import db
    try:
        db.add_event("problem", "server", False, action=kind, detail=detail, request=request)
    except Exception as e:
        log.warning("problems: could not save %s: %s", kind, type(e).__name__)


def note(kind: str, detail: str = "", request: str = "") -> None:
    now = time.time()
    key = (kind, detail[:200])
    if now - _last.get(key, 0) < REPEAT_SECONDS:
        return
    _last[key] = now
    log.warning("PROBLEM %s: %s %s", kind, detail[:300], f"(request: {request[:100]})" if request else "")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _save(kind, detail, request)
        return
    loop.run_in_executor(None, _save, kind, detail, request)  # the database may be far: off the event loop
