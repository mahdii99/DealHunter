"""The 25-second answer and what the LLM may say to the shopper (live test 2026-11-24)."""
import asyncio
import json

from fastapi.testclient import TestClient

from app import agent, llm, main
from app.models import Product


def _events(r) -> list[dict]:
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def _slow_turn(monkeypatch, wait: float):
    async def fake_turn(turn, client=None, user=None):
        agent._partial({"results": [{"title": "Fresh FTM-8S", "price": 15999, "store": "Jumia", "url": "u1"}],
                        "count": 1, "request": "غسالة فريش 8 كيلو", "summary": "s", "note": "n"})
        await asyncio.sleep(wait)
        return {"action": "search", "request": "غسالة فريش 8 كيلو", "count": 2, "results": [
            {"title": "Fresh FTM-8S", "price": 15499, "store": "2B", "url": "u2"},
            {"title": "Fresh FTM-8S", "price": 15999, "store": "Jumia", "url": "u1"}]}
    monkeypatch.setattr(main, "_chat_turn", fake_turn)


def _post():
    return TestClient(main.app).post("/api/chat/stream", json={"messages": [{"role": "user", "text": "غسالة فريش"}]})


def test_a_long_search_answers_with_the_checked_offers_then_updates(monkeypatch):
    """The shopper gets the offers already checked on their pages by the cap; the rest of the search goes on and
    its full answer comes as an update in the same stream."""
    monkeypatch.setattr(main, "SEARCH_CAP_SECONDS", 0.2)
    _slow_turn(monkeypatch, wait=0.6)
    kinds = [(e["type"], (e.get("data") or {}).get("pending")) for e in _events(_post()) if e["type"] != "ping"]
    assert kinds == [("partial", None), ("done", True), ("update", None)]
    events = _events(_post())
    interim = next(e["data"] for e in events if e["type"] == "done")
    assert interim["action"] == "search" and interim["results"][0]["price"] == 15999
    assert interim["query"] == "غسالة فريش 8 كيلو" and interim["candidates_checked"] == 1
    assert events[-1]["type"] == "update" and events[-1]["data"]["count"] == 2


def test_a_quick_search_answers_once(monkeypatch):
    monkeypatch.setattr(main, "SEARCH_CAP_SECONDS", 5)
    _slow_turn(monkeypatch, wait=0.05)
    kinds = [e["type"] for e in _events(_post()) if e["type"] != "ping"]
    assert kinds == ["partial", "done"]


def test_no_price_the_llm_guessed_reaches_the_shopper():
    """Live: "الغسالات بتبدأ من 9000 لحد 25000+ جنيه" came from web snippets, not from a store page."""
    assert agent.shopper_text("الغسالات دي بتبدأ من 9000 لحد 25000+ جنيه. محتاجها لكام فرد؟") == "محتاجها لكام فرد؟"
    assert agent.shopper_text("الأسعار من 9 لـ 25 ألف، تحب أوريك؟") == "تحب أوريك؟"
    # specs are not prices, and a price a shown result has is fine
    assert agent.shopper_text("غسالة 8 كيلو وموبايل 5000 مللي") == "غسالة 8 كيلو وموبايل 5000 مللي"
    assert agent.shopper_text("أرخصهم بـ 15,999 جنيه", [{"price": 15999}]) == "أرخصهم بـ 15,999 جنيه"
    # while it is written, a sentence with a number waits for its end (its currency word may be next)
    assert agent.shopper_text("تمام. بتبدأ من 9000", live=True) == "تمام."


def test_no_count_that_may_not_match_the_list():
    """Live: "رشحتلك 3 موديلات" above four of them."""
    assert agent.without_counts("رشحتلك 3 موديلات حلوة") == "رشحتلك موديلات حلوة"
    assert agent.without_counts("عندي تلات اختيارات ليك") == "عندي اختيارات ليك"
    assert agent.without_counts("غسالة 8 كيلو") == "غسالة 8 كيلو"


def test_the_requested_model_number_on_the_listing_is_the_exact_product(monkeypatch):
    """Live: Fresh FTM-8S, exactly what was asked for, was kept but flagged 'model doesn't match'."""
    async def fake_ask(prompt, **kw):
        return {"matches": [{"id": 0, "sure": False, "note": "الطراز غير متطابق"},
                            {"id": 1, "sure": False, "note": "اتأكد إنه أوتوماتيك"}], "summary": ""}
    monkeypatch.setattr(llm, "ask_json", fake_ask)
    exact = Product("Jumia", "Fresh Top Load Automatic Washing Machine 8 KG FTM-8S", 15999,
                    "https://www.jumia.com.eg/fresh-ftm-8s.html")
    longer = Product("2B", "Fresh washing machine FTM-8SX 8 KG", 15499, "https://2b.com.eg/p/fresh")
    _, picks, *_ = asyncio.run(agent._judge([exact, longer], {"requirements": []}, "غسالة فريش FTM-8S", None, None,
                                            ["FTM-8S"], []))
    by_store = {p.store: p for p in picks}
    assert by_store["Jumia"].unconfirmed is False and by_store["Jumia"].note is None
    assert by_store["2B"].unconfirmed is True  # another model number ("8SX") isn't the requested one


def test_a_too_cheap_note_never_names_the_llms_estimate():
    picks = [Product("shop", "Fresh FTM-8S", 5000, "u1")]
    agent._flag_price_outliers(picks, typical=16000)
    assert picks[0].unconfirmed and "16,000" not in picks[0].note


def test_uptime_monitors_can_ping_health():
    c = TestClient(main.app)
    assert c.head("/health").status_code == 200 and c.get("/health").json() == {"ok": True}
