"""The search agent.

1. UNDERSTAND: the LLM reads the shopper's request (Arabic/English) and plans web searches.
2. FIND:       a web search API finds store pages for the product anywhere on the web (no fixed store list).
3. VERIFY:     the agent opens each store page itself and reads products + prices from the page data
               (app.extract), so every price shown comes from the store page, not from the LLM.
4. JUDGE:      the LLM keeps only real matches from legitimate stores, with a short Arabic note each.
5. RANK:       cheapest first; well-known stores are marked as trusted.

Fallbacks: no search key -> the built-in direct stores (app.stores); no LLM key -> simple rules.
"""
import asyncio
import contextvars
import json
import logging
import re
import time
from html import unescape
from urllib.parse import unquote, urljoin, urlparse

import httpx

from . import db, fetchers, llm, market, trust, websearch
from . import extract as extract_mod
from .extract import extract_products
from .models import ParsedQuery, Product
from .query import parse_query
from .ranking import rank
from .stores import ALL_STORES
from .stores.base import DEBUG_DIR, HEADERS

log = logging.getLogger("uvicorn.error")
MAX_CANDIDATES = 50  # cheapest first; the judge reads every line, so more = more tokens per search
PER_STORE_FIRST = 8  # candidates from one store before other stores get their turn (more stores reach the judge)
JUDGE_DOWN_RESULTS = 12  # no model could judge: the cheapest candidates, all marked "check it"
MAX_PAGES = 20          # store pages opened per search
MAX_PER_PAGE = 15       # products kept from one page (listing pages can have many)
FAMOUS_RESULTS = 15     # search results for the well-known stores group (one search shared by 5-10 big stores)
WEB_DEBUG = DEBUG_DIR / "web"

UNDERSTAND_PROMPT = """You are the salesperson of a price-comparison agent for shoppers in Egypt: like a smart, friendly
salesperson standing in a shop, except you stand in every online store at once. Talk like one, in short, warm
Egyptian Arabic (English if the shopper writes English). Your job is to find out what the shopper really needs and get
them the right product at the best price, not just to run a search for the words they typed.

Shoppers often don't know the product's name or how stores list it: they describe what they need
("تيشرت راوند بيزك اسود", "موبايل مش غالي وبطاريته تكمل يومين"). Understand what they really want and turn it
into concrete specs and searches that find it in Egyptian online stores.

Choose ONE action for the newest message, the way a good salesperson would:
- "search": the shopper named a specific product/model, asked for prices/offers, picked one of your suggestions, or
  you already know enough to find good matches. Don't make a shopper who knows what they want answer questions.
- "ask": the request is too open to pick good products and ONE answer would change what you'd recommend a lot (who
  it's for, what they'll use it for, budget, size...). Ask one short question with 2-4 tap answers. Don't repeat
  a question already asked, and after two questions in a row stop asking: recommend or search.
- "recommend": the shopper described a need, or asked what to buy ("أنهي موبايل أحسن في حدود 8000؟", "عايز لابتوب
  للبرمجة"): suggest 2-4 specific products sold in Egypt now that fit, each with one line on why it fits them, from
  your own knowledge and WITHOUT prices (you don't know today's prices) and without searching. The shopper then picks
  one, or asks you to find prices for all of them (that is a "search" for those products).
- "answer": a QUESTION about results already shown or buying advice ("أول نتيجة دي أصلية؟", "الفرق بين الاتنين إيه؟",
  "أنهي أحسن؟", "عليها ضمان؟", "السعر ده كويس؟") that doesn't ask for different products: reply from the shown results
  (their stores, prices, discounts, trust, price history) and your knowledge, without a new search. To compare 2-4
  products or offers, also fill "table". Be honest: if a price looks too low for that product, say it may be a
  mistake, a different variant or a used/refurbished listing and how to check on the store page; if a store is
  unverified, say so.

The chat is a conversation. A new message usually CHANGES THE CURRENT REQUEST instead of starting over:
"سامسونج بس" / "عايزه أرخص" / "بلاش الأحمر" / "مقاس XL" / "تحت 5000" -> keep everything from the current request
that still applies and apply the change. Start fresh only if the shopper clearly asks for a different product.
Never change the product type the shopper named: if you don't recognize a model name (e.g. "anker r50i nc"),
search the shopper's own words as they are instead of guessing a category.
Every word the shopper wrote matters: never drop one from the request. A word that looks out of place often names
the variant in Egyptian shop slang (control type, size, material, edition...): keep it in the request and, when you
know what it means for this product, add that as a requirement in English.
Quality words are real requirements, never drop them: "ماركة نضيفة / معروفة / أصلي / كويس" means well-known
quality brands, so name those brands in suggested_products and add a requirement like "well-known brand (e.g. ...)",
so no-name products don't count as a match. The web queries must then NAME the brands (e.g. 'تيشيرت بيزك ديفاكتو',
'LC Waikiki basic t-shirt'), never generic words like "ماركة مشهورة" or "brand": those find no-name shops.
The search fields (need_based and everything after "request") are needed ONLY for "search": leave them empty
for the other actions.

{chat}
Reply JSON (put "reply" first):
{
  "reply": "what you say to the shopper now, as a salesperson: for ask the question itself; for recommend one short
            intro line; for answer 1-5 short sentences answering; for search one short line saying what you'll look
            for (mention the change if a follow-up)",
  "action": "search" or "ask" or "recommend" or "answer",
  "options": ["if ask: 2-4 short quick answers the shopper can tap, else []"],
  "recommendations": [{"name": "if recommend: a specific product, e.g. 'Samsung Galaxy A16'", "why": "one short
     Egyptian Arabic line on why it fits THIS shopper"}],
  "table": null or, for an answer that compares: {"columns": ["", "product 1", "product 2"],
           "rows": [["السعر", "...", "..."], ["الضمان", "...", "..."]]} (2-6 short rows),
  "suggestions": ["2-4 short Egyptian Arabic next messages the shopper may want to tap, written as the shopper
     would say them (e.g. 'أرخص من كده', 'سامسونج بس', 'إيه الفرق بينهم؟', 'دوّرلي على أسعارهم'); [] for ask"],
  "request": "the full standalone request in Egyptian Arabic built from the whole conversation, e.g.
              'موبايل سامسونج سعره مش غالي وبطاريته كبيرة تكمل يومين'",
  "need_based": true if the shopper described a need/use rather than naming a specific product,
  "suggested_products": ["if need_based: 3-6 specific products sold in Egypt now that fit the need, with the
     spec that makes them fit, e.g. 'Samsung Galaxy A16 (5000 mAh)', 'Xiaomi Redmi 14C (5160 mAh)'. Names and
     specs only, NEVER prices (you don't know today's prices); else []"],
  "stores": ["4-8 domains of the biggest well-known online stores in Egypt for THIS kind of product: big
     marketplaces plus the category's own leaders and the brands' official Egypt stores (electronics, fashion,
     beauty, furniture, groceries... each has different leaders). Only real domains you are sure of"],
  "shopper_stores": ["domains of the stores the shopper names anywhere in the chat (e.g. 'جوميا' -> 'jumia.com.eg',
     'من نون' -> 'noon.com'), so those stores are searched on their own too; else []"],
  "web_queries": ["2 web search queries, or 3 when the shopper names or needs several brands/products (one query
     per brand family); each runs once on the well-known stores and once on smaller shops:
     (1) how Egyptian shoppers search in Arabic, e.g. 'سعر سخان اوليمبيك هيرو 50 لتر ميكانيكي' or for a need
         'تيشرت قطن سادة اسود رقبة دائرية رجالي';
     (2) precise English: the official model/part number if you know it, e.g. 'Olympic Electric Hero 50L
         945105411 price Egypt'; for a need, the store wording or the best suggested products, e.g.
         'Samsung Galaxy A16 5000mAh price Egypt' or 'men black crew neck basic cotton t-shirt'"],
  "model_codes": ["official model / part numbers of the requested product if you know them, else []"],
  "queries": ["1-2 short English store search queries, most specific first"],
  "requirements": ["hard requirements a matching product must meet, in English, from the need too, e.g.
     'battery 5000 mAh or more', 'black', 'crew (round) neck', 'plain, no print'"],
  "avoid": ["things that must NOT be returned, e.g. 'cases/covers', 'bundles with other products', 'used items'"],
  "max_price": number or null,   // ONLY when the shopper clearly states a price/budget ("تحت 8000", "بحدود 5000 جنيه",
                                 // "budget 3000"). A bare number is a spec or model ("مكواه 650", "2200 وات"): put it
                                 // in requirements/queries, NOT here. Words like "رخيص" / "مش غالي" -> null.
  "min_price": number or null
}"""
MAX_HISTORY = 8  # chat messages the planner sees
_PRICE_TAIL = re.compile(r"\s*[-–—:,(]*\s*(?:EGP|LE|ج\.?م|جنيه)?\s*\d[\d,.]*\s*(?:EGP|LE|ج\.?م|جنيه)\)?\s*$"
                         r"|\s*[-–—:,(]+\s*(?:EGP|LE)\s*\d[\d,.]*\)?\s*$", re.I)


MAX_SHOWN = 10  # results of the last search the planner sees (to answer and compare)


def _money(n) -> str:
    return f"{n:g}" if isinstance(n, (int, float)) else "?"


def _shown_block(shown: list[dict]) -> str:
    """The results on screen, with what the shopper can't see at a glance: trust, stock check, price history."""
    lines = []
    for i, r in enumerate(shown[:MAX_SHOWN], 1):
        bits = [f"{i}) {str(r.get('title', ''))[:140]}", f"{_money(r.get('price'))} {market.current().currency}",
                f"store {r.get('store', '?')}"]
        if isinstance(r.get("old_price"), (int, float)) and r["old_price"] > (r.get("price") or 0):
            bits.append(f"was {_money(r['old_price'])}" + (f" (-{r['discount_pct']}%)" if r.get("discount_pct") else ""))
        if r.get("trusted"):
            bits.append("well-known store")
        elif r.get("trust_score") is not None:
            bits.append(f"store trust {r['trust_score']}/100")
        if r.get("verified") is True:
            bits.append("in stock (page checked now)")
        elif r.get("verified") is False:
            bits.append("stock not checked")
        if r.get("condition"):
            bits.append(f"CONDITION {r['condition']}")
        if r.get("unconfirmed"):
            bits.append("exact model not confirmed")
        if isinstance(r.get("lowest_seen"), (int, float)):
            bits.append(f"lowest price we've seen {_money(r['lowest_seen'])}")
        if isinstance(r.get("previous_price"), (int, float)):
            bits.append(f"last time {_money(r['previous_price'])}")
        if r.get("note"):
            bits.append(f"note: {str(r['note'])[:120]}")
        lines.append(" | ".join(bits))
    return ("Results shown to the shopper now (cheapest first):\n" + "\n".join(lines) + "\n") if lines else ""


def _chat_block(messages: list[dict], current: str, shown: list[dict] | None = None) -> str:
    lines = [("Shopper: " if m.get("role") == "user" else "Assistant: ") + str(m.get("text", ""))[:800]
             for m in messages[-MAX_HISTORY:]]
    return ("Current request (from the previous search): " + (current or "none") + "\n" + _shown_block(shown or [])
            + "Conversation (oldest first, the last line is the newest message):\n" + "\n".join(lines) + "\n")


_REPLY_SO_FAR = re.compile(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)')


def reply_so_far(text: str) -> str:
    """The "reply" field of a JSON answer that is still arriving, so the shopper can read it while it's written."""
    m = _REPLY_SO_FAR.search(text)
    if not m:
        return ""
    raw = re.sub(r"\\(u[0-9a-fA-F]{0,3})?$", "", m.group(1))  # a cut-off escape at the end waits for the next part
    try:
        return json.loads('"' + raw + '"')
    except ValueError:
        return raw.replace("\\n", "\n").replace('\\"', '"')


REPLY: contextvars.ContextVar = contextvars.ContextVar("reply", default=None)  # (text so far) -> shown live


def _strs(xs, n: int, size: int = 120) -> list[str]:
    return [x.strip()[:size] for x in xs or [] if isinstance(x, str) and x.strip()][:n]


def _table(t) -> dict | None:
    """A comparison table from the LLM, only when it has the right shape (a header and 1-8 rows of short cells)."""
    if not isinstance(t, dict) or not isinstance(t.get("columns"), list) or not isinstance(t.get("rows"), list):
        return None
    cols = [str(c)[:60] for c in t["columns"][:5]]
    rows = [[str(c)[:120] for c in r[:len(cols)]] + [""] * (len(cols) - len(r[:len(cols)]))
            for r in t["rows"][:8] if isinstance(r, list) and r]
    return {"columns": cols, "rows": rows} if len(cols) >= 2 and rows else None


async def plan_turn(messages: list[dict], current: str = "", shown: list[dict] | None = None) -> dict:
    """One LLM call: understand the newest message in the context of the chat (and the results on screen), decide
    what a good salesperson does now (search / ask / recommend / answer), and plan the searches.
    Raises llm.LLMError when no model answers."""
    live = REPLY.get()
    on_text = (lambda text: live(t) if (t := reply_so_far(text)) else None) if live else None
    plan = await llm.ask_json(UNDERSTAND_PROMPT.replace("{chat}", _chat_block(messages, current, shown)), timeout=12,
                              prefer="strong", step="plan", on_text=on_text)  # gpt-oss-120b on Groq first: ~1 s
    last = next((m.get("text", "") for m in reversed(messages) if m.get("role") == "user"), "")
    # names only: a price the LLM guessed would mislead the shopper and the judge
    plan["suggested_products"] = [_PRICE_TAIL.sub("", x).strip() for x in plan.get("suggested_products") or []
                                  if isinstance(x, str) and x.strip()]
    reply = plan.get("reply") if isinstance(plan.get("reply"), str) else ""
    reply = reply.strip() or str(plan.get("question") or plan.get("answer") or "").strip()  # older field names
    plan["reply"] = reply
    plan["suggestions"] = _strs(plan.get("suggestions"), 4, 60)
    if not isinstance(plan.get("request"), str) or not plan["request"].strip():
        plan["request"] = last
    recs = [{"name": str(r["name"]).strip()[:120], "why": str(r.get("why") or "").strip()[:200]}
            for r in plan.get("recommendations") or [] if isinstance(r, dict) and str(r.get("name") or "").strip()]
    action = plan.get("action")
    if action == "answer" and reply:
        plan["answer"] = reply
        plan["table"] = _table(plan.get("table"))
        plan["request"] = current or plan["request"]  # a question about the results keeps the current request
    elif action == "recommend" and recs:
        plan["recommendations"] = [{**r, "name": _PRICE_TAIL.sub("", r["name"]).strip()} for r in recs[:4]]
        plan["request"] = current or plan["request"]
    elif action == "ask" and reply:
        plan["question"] = reply
        plan["options"] = _strs(plan.get("options"), 4, 60)
    else:
        plan["action"] = "search"
        plan["understood"] = str(plan.get("understood") or reply).strip()
    log.info("agent: plan %s -> %r", plan["action"], reply if plan["action"] != "search" else plan["request"])
    return plan


CHECK_PROMPT = """You plan searches for a price-comparison agent for shoppers in Egypt.
Shopper's request: {request}
Our first reading: {understood}
Planned web searches: {queries}
Planned well-known stores: {stores}
Model numbers: {codes}
Requirements: {req}

Web search results for the shopper's own words (title | snippet):
{hits}

Use these results to check what the shopper really wants before we spend searches on it: the exact product, its
product type, the variant they mean, and its official model numbers. Shoppers write slang, misspellings and mixed
Arabic/English, and the same word can mean different things for different products: read every word in the context
of this product, the way stores selling it use it. If the results clearly show the product, trust them over the first
reading; if they are unrelated or unclear, keep the first reading. Never change the product type the shopper named.
Keep every requirement of the first reading. Model numbers only pin down ONE exact product the shopper named; when
they want a kind of product (any model that meets the specs), leave model_codes empty and keep the searches broad
enough to find every fitting model (one query per brand they want, not one example model).
If the shopper wants well-known brands, the queries must name brands
(never generic words like "ماركة مشهورة" / "brand", which find no-name shops).
Reply JSON:
{{"understood": "one short Egyptian Arabic sentence saying what you'll look for",
  "web_queries": ["2, or 3 when several brands/products are wanted: Arabic as Egyptian shoppers and stores write
                   it, and precise English with the official model number if known. Each item is ONE short search
                   in ONE language (never two searches joined with '|' or '/'); spell brand names correctly"],
  "stores": ["4-8 domains of the biggest well-known Egyptian online stores for this product (from the results
              and your knowledge)"],
  "model_codes": ["official model / part numbers of this exact variant, else []"],
  "requirements": ["hard requirements a matching product must meet, in English"],
  "changed": true if the first reading was wrong about the product or the variant}}"""
CHECK_HITS = 8
MAX_WEB_QUERIES = 3  # each one = 2 searches (well-known stores + other shops)
MARKETPLACES = ["amazon.eg", "noon.com", "jumia.com.eg"]  # sell almost everything: always in the well-known group
_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")


def _domains(names) -> list[str]:
    out = []
    for d in names or []:
        d = str(d).lower().strip().removeprefix("https://").removeprefix("http://").removeprefix("www.").split("/")[0]
        if _DOMAIN.match(d) and trust.is_shop_candidate("https://" + d):
            out.append(d)
    return list(dict.fromkeys(out))


def _famous_sites(plan: dict) -> list[str]:
    """Well-known stores for this product: the big marketplaces + the ones the LLM named for this category.
    Falls back to our own list when the LLM named none."""
    named = _domains(plan.get("stores"))
    return list(dict.fromkeys(MARKETPLACES + named[:8])) if named else trust.FAMOUS_SITES


async def _check_plan(plan: dict, request: str) -> tuple[dict, list[dict]]:
    """One web search of the shopper's own words, then one LLM call that fixes the plan with what the web says the
    product is, so the main searches look for the right thing. The search hits are reused as store pages."""
    t = time.monotonic()
    try:
        async with httpx.AsyncClient() as client:
            hits = await websearch.search(client, request, max_results=CHECK_HITS)
    except (websearch.SearchError, httpx.HTTPError) as e:
        log.info("agent: check search failed: %s", e)
        return plan, []
    if not hits:
        return plan, []
    _progress("بتأكد من المنتج قبل ما أدور...")
    prompt = CHECK_PROMPT.format(
        request=request, understood=plan.get("understood") or request,
        queries=json.dumps(plan.get("web_queries", []), ensure_ascii=False),
        stores=json.dumps(plan.get("stores", []), ensure_ascii=False),
        codes=json.dumps(plan.get("model_codes", []), ensure_ascii=False),
        req=json.dumps(plan.get("requirements", []), ensure_ascii=False),
        hits="\n".join(f"- {h.get('title', '')[:120]} | {h.get('snippet', '')[:200]}" for h in hits))
    try:
        fix = await llm.ask_json(prompt, timeout=10, budget=15, prefer="strong", step="check")
    except llm.LLMError as e:
        log.info("agent: plan check skipped: %s", e)
        return plan, hits
    queries = [q for q in fix.get("web_queries", []) if isinstance(q, str) and q.strip()][:MAX_WEB_QUERIES]
    if queries:
        plan = {**plan, "web_queries": queries}
        for k in ("model_codes", "requirements", "stores"):
            if isinstance(fix.get(k), list):
                plan[k] = [x for x in fix[k] if isinstance(x, str) and x.strip()]
        if isinstance(fix.get("understood"), str) and fix["understood"].strip():
            plan["understood"] = fix["understood"]
    TIMES["check"] = round(time.monotonic() - t, 1)
    log.info("agent: plan checked in %.1fs (changed=%s): %s", time.monotonic() - t, fix.get("changed"),
             plan.get("web_queries"))
    return plan, hits


JUDGE_PROMPT = """You are the judge of a price-comparison agent for Egypt. Shopper request: {raw}
Requirements: {req}
Model numbers of the requested product (from the planner, may be incomplete): {codes}
Products the planner thinks fit the shopper's need (examples, NOT a complete list): {suggested}
Avoid: {avoid}
Budget (EGP): min {minp}, max {maxp}
Candidates (id, store, store trust 0-100 with the signals found on its site, condition if not new,
price EGP, discount, title, product link path (often holds the model number):
{items}

Return EVERY candidate that fits (often 10-25), not just the best few: the shopper wants to compare offers from
as many different stores as have it (but never add a wrong product just to have another store). Check each
candidate against EVERY requirement one by one (brand, quality, size, type, spec): if its title shows it misses
one, drop it (e.g. a no-name item when the shopper wants well-known brands).
A candidate fits when it (1) is what the shopper asked for. When the shopper described a need (e.g. "battery lasts
two days"), any product that meets the requirements fits, not only the suggested ones; judge from the title/specs
and your knowledge of that model. Variants of one product line are DIFFERENT products
(e.g. knob/mechanical vs digital, Hero vs Hero Plus/Turbo/Lite, 50L vs 30L): drop any candidate whose title, model
number or link shows the wrong variant. Use model numbers in the title or link (e.g. 945105411 = Hero 50L mechanical)
to confirm. Store titles are often short: if nothing shows the variant either way, KEEP it with "sure": false and
say in the note what to check. Set "sure": true only when the title/model/link confirms the exact variant. And
(2) come from a legitimate retailer selling new items in Egypt. Smaller, lesser-known shops are WELCOME and often
cheaper: keep them when their trust is 35+ (real Egyptian phone, cash on delivery, return policy, warranty...).
Drop shops with trust below 30, copies/replicas, blogs, price-list / specs sites (titles like "سعر ومواصفات في مصر",
"price in Egypt <year>", links under /blog/ or /buying-guides/: they list prices but you can't buy there), classifieds and used listings (an ad by a
person selling their own item: links like /ad/ or /listing/, titles like "للبيع", "للبدل", "زي الجديد", "وارد ..."
with no shop behind it, even if the site looks organized), and prices
suspiciously far below the others for the same item. A branded item sold by a shop that isn't the brand or a big
retailer, far below what that brand charges in Egypt (common with clothes, shoes, perfume, bags), is likely a copy:
drop it, or keep it with "sure": false and a note saying it may not be original. Check every price against what that product really costs in
Egypt today (your market knowledge + the other candidates): a price far below it (e.g. an 18 ft fridge for 2,000 EGP)
is a wrong currency, a monthly installment, a deposit or an accessory, so drop it. Drop used / refurbished / open-box items unless the shopper
asked for them. Real discounts are a plus. Reply JSON:
{{"matches": [{{"id": <id>, "sure": true|false, "note": "very short Egyptian Arabic note on why it fits, or what to check"}}],
  "alternatives": [{{"id": <id>, "note": "..."}}],  // ONLY if matches is empty: closest good options from legit stores,
                                                  // e.g. used/refurbished from a real shop, or a close spec
  "typical_price": <what ONE genuine, new unit of the requested product usually costs in Egypt today in EGP, from
                    the big/official stores among the candidates and your own market knowledge; null when the shopper
                    described a need that many different products fit, or you really can't tell>,
  "best_deal_id": <id of the best value-for-money pick: low price + real discount + trustworthy shop, or null>,
  "summary": "1-2 Egyptian Arabic sentences: the best cheap option from a trusted store and any useful advice
              (name stores and prices, never the candidate ids; if the cheapest offer looks too cheap to be genuine,
              say so)",
  "suggestions": ["2-4 short Egyptian Arabic next messages the shopper may want to tap after seeing these results,
                   written as the shopper would say them, e.g. 'أرخص من كده', 'من متجر مشهور بس', 'إيه الفرق بين أول
                   اتنين؟', 'فيه لون تاني؟'"]}}"""


PAGE_READ_PROMPT = """This is the text of a web page found while searching Egyptian online stores (links shown as
[link: url] or markdown [text](url)). Extract the products for sale on it that could fit this shopper's request (any
brand or model that may fit, not only the one in the search words; a later step judges the exact match): {query}
Reply JSON: {{"page": "shop" (a store's product or listing page, where you can buy) or "article" (news, blog, review,
guide, price list or specs page: it only talks about products),
"currency": "currency of the prices on the page, e.g. EGP, USD, SAR",
"products": [{{"title": "...", "price": number (current price as shown, no old price),
"old_price": number or null (the crossed-out price before the discount, only if the page shows one),
"url": "the product link from the text or null", "condition": "new" | "used" | "refurbished",
"in_stock": true | false | null (false if the page shows out of stock / sold out / غير متوفر / نفذت الكمية)}}]}}
At most 12 products. Use only prices written on the page; never guess. Empty list if none, if the page is an
article, or if the prices are monthly installments.
PAGE:
{text}"""
MAX_READER_PAGES = 8  # failed pages retried through a reader service per search
MAX_LLM_PAGES = 6  # pages with no structured data that the LLM reads itself (free-tier friendly)
MIN_CRAWLED_TEXT = 800  # shorter "page text" from the search service is just a snippet
MIN_TRUST = 30  # unknown shops below this (few real-shop signals on their site) are never shown
MAX_LLM_READS = 8  # all LLM page reads per search, incl. pages the reader service opened
PER_STORE_READS = 2  # LLM reads per store, so more stores get read (a third listing page of one store mostly repeats)
LLM_READ_PARALLEL = 3  # pages read at once; each goes to a different model first, so per-minute limits hold
PAGE_TIMEOUT = 7         # one store page: slower than this = a slow site, the reader service tries it instead
PAGES_DEADLINE = 9       # all store pages: stop waiting for stragglers after this
READING_DEADLINE = 20    # LLM reading of pages without product data (incl. the reader service)
READER_DEADLINE = 12     # the reader service (Tavily Extract / Jina) for pages we couldn't open


async def _within(seconds: float, coros: list) -> list:
    """Run all, wait at most `seconds`; anything not done by then counts as a TimeoutError result."""
    return await _results(seconds, [asyncio.ensure_future(c) for c in coros])


async def _results(seconds: float, tasks: list) -> list:
    if not tasks:
        return []
    done, pending = await asyncio.wait(tasks, timeout=seconds)
    for task in pending:
        task.cancel()
    out = []
    for task in tasks:
        if task in pending:
            out.append(TimeoutError("slow"))
        else:
            out.append(task.exception() or task.result())
    return out


# Live progress for the page ("found 40 products, comparing..."): main.py sets a callback for the request.
PROGRESS: contextvars.ContextVar = contextvars.ContextVar("progress", default=None)


def _progress(text: str) -> None:
    if (say := PROGRESS.get()) is not None:
        say(text)


TIMES: dict[str, float] = {}  # seconds per stage of the last search (log + benchmark report)


PAGE_CHARS = 3000  # page text sent to the LLM: the part with the products (most of a page is menus and footers)
_PRICE_NUM = re.compile(r"\d[\d,.]{2,}")


def _page_text(html: str, focus: list[str], limit: int = PAGE_CHARS, base_url: str = "") -> str:
    """Readable page text with links kept, cut to the window that mentions the product most (search words and
    prices), so the LLM reads less text without losing the products."""
    html = re.sub(r"<(script|style|noscript|svg|head)\b.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<a\b[^>]*href=[\"']([^\"'#?]+)[^\"']*[\"'][^>]*>(.*?)</a>", r" \2 [link: \1] ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # markdown images (reader services): no names or prices
    text = re.sub(r"\]\((https?://[^)\s?#]+)[^)]*\)", r"](\1)", text)  # tracking parameters on markdown links
    if base_url and (site := urlparse(base_url).netloc):  # this site's own links as paths (urljoin restores them)
        text = re.sub(r"https?://" + re.escape(site) + r"(?=/)", "", text)
    text = re.sub(r"\[link: (?:javascript:|mailto:|tel:)[^\]]*\]", " ", text)
    seen: set[str] = set()

    def once(m: re.Match) -> str:  # a product card repeats its link 3-5 times (image, title, rating, price...)
        if m.group(1) in seen:
            return " "
        seen.add(m.group(1))
        return m.group(0)
    text = re.sub(r"\[link: ([^\]]+)\]", once, text)
    text = re.sub(r"\s+", " ", unescape(text))
    if len(text) <= limit:
        return text
    low = text.lower()
    marks = sorted(m.start() for w in {w.lower() for w in focus if len(w) > 2} for m in re.finditer(re.escape(w), low))
    if not marks:
        return text[:limit]
    prices = [m.start() for m in _PRICE_NUM.finditer(text)]

    def score(start: int) -> int:  # search words + prices inside the window
        end = start + limit
        return sum(start <= i < end for i in marks) + sum(start <= i < end for i in prices) // 2
    start = max((max(0, i - 300) for i in marks[::max(1, len(marks) // 40)]), key=score)
    return text[start:start + limit]


async def _llm_read_page(url: str, html: str, query: str, spread: int = 0, focus: str = "") -> list[Product]:
    """`query` says what the shopper wants; `focus` (default: the query) holds the words used to find the part
    of a long page that shows the product."""
    words = [w for w in re.split(r"\W+", focus or query) if len(w) > 2]
    data = await llm.ask_json(PAGE_READ_PROMPT.format(query=query, text=_page_text(html, words, base_url=url)),
                              prefer="bulk", spread=spread, step="read_page")
    if data.get("page") == "article":
        raise NotAStore("article page (not a store)")
    cur = str(data.get("currency") or "").strip()
    if cur and cur.upper() not in {c.upper() for c in market.current().currency_aliases + (market.current().currency,)} \
            and cur not in ("ج.م", "جم", "جنيه مصري", "L.E", "L.E."):
        raise NotAStore(f"page prices in {cur[:10]}")
    out = []
    items = data.get("products")
    for x in (items if isinstance(items, list) else [])[:15]:
        if not isinstance(x, dict):
            continue
        price = x.get("price")
        if not isinstance(price, (int, float)) or price < 5 or not x.get("title"):
            continue
        link = urljoin(url, x.get("url") or url)
        if not trust.is_shop_candidate(link):  # e.g. a link to another country's store
            continue
        cond = x.get("condition")
        old = x.get("old_price")
        out.append(Product(store=trust.store_name(link), title=str(x["title"])[:200], price=float(price), url=link,
                           old_price=float(old) if isinstance(old, (int, float)) and price < old <= price * 10 else None,
                           trusted=trust.is_known(link), condition=cond if cond in ("used", "refurbished") else None,
                           in_stock=x.get("in_stock") if isinstance(x.get("in_stock"), bool) else None))
    score, signals = trust.page_signals(html, url, False)
    for p in out:
        p.trust_score, p.trust_signals = (100, ["متجر مشهور"]) if p.trusted else (score, signals)
    return out


class NotAStore(Exception):
    """The LLM read the page: it isn't a place to buy (news, blog, review) or its prices are another currency."""


_ARTICLE_TYPE = re.compile(r'"@type"\s*:\s*"(?:NewsArticle|BlogPosting|Article|ReportageNewsArticle|AnalysisNewsArticle)"'
                           r"|<meta[^>]+property=[\"']og:type[\"'][^>]+content=[\"']article", re.I)
_BUY = re.compile(r"add.to.(?:cart|bag|basket)|buy now|[أا]ضف (?:إلى|الى|لل)|اشتري? الآن|"
                  r"\"@type\"\s*:\s*\"(?:Product|Offer)\"", re.I)


def _is_article(html: str) -> bool:
    """A news / blog page says so in its own page data (schema.org Article, og:type article) and has nothing to
    buy on it: no LLM read needed, and its site is probably no store."""
    return bool(_ARTICLE_TYPE.search(html)) and not _BUY.search(html)


class RobotsDisallowed(Exception):
    def __str__(self):
        return "robots.txt disallows"


async def _fetch_page(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> tuple[list[Product], str]:
    # the store's crawler rules apply to our own requests too, not only to the reader service (cached per site)
    if not await asyncio.wait_for(fetchers.robots_allows(client, url), timeout=PAGE_TIMEOUT):
        raise RobotsDisallowed()
    async with sem:
        r = await client.get(url, headers=HEADERS, follow_redirects=True, timeout=PAGE_TIMEOUT)
    r.raise_for_status()
    try:  # keep the latest copy of each store page so parsing problems can be checked later
        WEB_DEBUG.mkdir(parents=True, exist_ok=True)
        (WEB_DEBUG / f"{trust.domain(url)}.html").write_text(r.text, encoding="utf-8")
    except OSError:
        pass
    products = _cheapest_variant(extract_products(r.text, trust.store_name(url), str(r.url)))[:MAX_PER_PAGE]
    for p in products:
        p.store = trust.store_name(p.url or url)
        p.trusted = trust.is_known(p.url or url)
    score, signals = trust.page_signals(r.text, str(r.url), bool(products))
    for p in products:
        p.trust_score, p.trust_signals = (100, ["متجر مشهور"]) if p.trusted else (score, signals)
    return products, r.text


def _cheapest_variant(products: list[Product]) -> list[Product]:
    """One entry per product link (variants share it): fix x100 outliers, keep the cheapest, drop junk prices."""
    by_url: dict[str, list[Product]] = {}
    for p in products:
        by_url.setdefault(p.url, []).append(p)
    out = []
    for group in by_url.values():
        prices = sorted(p.price for p in group)
        base = prices[0]
        for p in group:
            if len(group) > 1 and p.price >= 50 * base and base * 1000 > p.price / 100 >= base / 2:
                p.price /= 100  # stored in piasters while siblings are in pounds
        group = [p for p in group if p.price >= 5]
        if group:
            # an in-stock variant before a cheaper one that's sold out
            out.append(min(group, key=lambda p: (p.in_stock is False, p.coupon_price or p.price)))
    return out


async def _web_find(web_queries: list[str], seed: list[dict] | None = None,
                    famous: list[str] | None = None, want: str = "",
                    named: list[str] | None = None) -> tuple[list[Product], dict[str, str]]:
    """`want`: what the shopper asked for, used when the LLM reads a page. Not just the first search query: with
    several brands each query names one, and a page found by another query would look like it has nothing.
    `named`: stores the shopper named ("جوميا أرخص"): each query is also searched on them alone and their pages
    are opened first, so a store the shopper points at isn't crowded out by the others."""
    want = want or web_queries[0]
    focus = " ".join(web_queries)
    t = time.monotonic()
    errors: dict[str, str] = {}
    async with httpx.AsyncClient() as client:
        # Two groups searched side by side: famous stores, and every other shop. The judge compares them all.
        blocked = await _blocked_sites()  # sites we couldn't open in earlier searches: don't spend searches on them
        famous = [d for d in famous or trust.FAMOUS_SITES if d not in blocked]
        named = [d for d in named or [] if d not in blocked][:3]
        groups = {"famous": dict(include_domains=famous),
                  "other": dict(exclude_domains=famous + trust.CLASSIFIEDS + sorted(blocked)[:100])}
        if named:
            groups["named"] = dict(include_domains=named)
        # the well-known group shares one search between many big stores: more results (same cost), so each of
        # them shows up instead of the biggest one taking every slot
        size = {"famous": FAMOUS_RESULTS, "other": 8, "named": 8}
        jobs = [(g, websearch.search(client, q, max_results=size[g], **kw)) for q in web_queries for g, kw in groups.items()]
        found = await asyncio.gather(*(j for _, j in jobs), return_exceptions=True)
        TIMES["search"] = round(time.monotonic() - t, 1)
        picked: dict[str, list[str]] = {"named": [], "famous": [], "other": []}
        crawled: dict[str, str] = {}  # page text the search service already has: used when we can't open the page
        jobs, found = [("check", None)] + jobs, [seed or []] + list(found)  # the plan-check search's hits count too
        for (group, _), res in zip(jobs, found):
            if isinstance(res, Exception):
                errors[f"web search ({group})"] = str(res)[:200]
                continue
            for hit in res:
                u = hit["url"]
                if len(hit.get("content") or "") >= MIN_CRAWLED_TEXT:
                    crawled.setdefault(u, hit["content"])
                if (trust.is_shop_candidate(u) and trust.domain(u) not in blocked
                        and all(u not in lst for lst in picked.values())):
                    big = trust.is_known(u) or any(trust._match(trust.domain(u), d) for d in famous)
                    mine = any(trust._match(trust.domain(u), d) for d in named)
                    picked["named" if mine else "famous" if big else "other"].append(u)
        # Same page budget for each group, alternating, so neither side crowds the other out; the stores the
        # shopper named come first (at most half the pages).
        known, small = _take_turns(picked["famous"]), _take_turns(picked["other"])
        urls = [u for pair in zip(known, small) for u in pair] + known[len(small):] + small[len(known):]
        urls = (_take_turns(picked["named"])[:MAX_PAGES // 2] + urls)[:MAX_PAGES]
        log.info("web: %d store pages to open: %s", len(urls), [trust.domain(u) for u in urls])
        _progress(f"لقيت {len(urls)} صفحة في {len({trust.domain(u) for u in urls})} متجر، بفتحها وبجمع الأسعار...")
        sem = asyncio.Semaphore(5)
        pages = await _within(PAGES_DEADLINE, [_fetch_page(client, sem, u) for u in urls])

        products, seen, unread, failed = [], set(), [], []
        for url, res in zip(urls, pages):
            if isinstance(res, Exception):
                code = getattr(getattr(res, "response", None), "status_code", None)
                errors[trust.domain(url)] = f"{code}" if code else type(res).__name__
                failed.append(url)
                continue
            page_products, html = res
            if not page_products and (cur := extract_mod.foreign_currency(html)):
                errors[trust.domain(url)] = f"prices in {cur}"  # another country's store: not worth an LLM read
                continue
            if not page_products and _is_article(html):
                errors[trust.domain(url)] = "article page (not a store)"  # news / blog: nothing to buy, no LLM read
                continue
            if not page_products:
                # Little visible text = a JavaScript-only page; a reader service renders it better.
                if len(re.sub(r"<[^>]+>|\s+", "", re.sub(r"<script.*?</script>", "", html, flags=re.S))) < 1500:
                    failed.append(url)
                else:
                    unread.append((url, html))
            for p in page_products:
                if (p.url, p.title) not in seen:
                    seen.add((p.url, p.title))
                    products.append(p)

        TIMES["pages"] = round(time.monotonic() - t - TIMES["search"], 1)
        if unread or failed:
            _progress(f"جمعت {len(products)} سعر لحد دلوقتي، وبقرا {len(unread) + len(failed)} صفحة كمان...")
        llm_sem = asyncio.Semaphore(LLM_READ_PARALLEL)
        n_read = [0]

        async def read_one(u, h):
            async with llm_sem:
                n_read[0] += 1  # each page goes to a different model first: each has its own per-minute limit
                return await _llm_read_page(u, h, want, spread=n_read[0], focus=focus)

        def collect(batch, results):
            for (url, _), res in zip(batch, results):
                if isinstance(res, NotAStore):
                    errors[trust.domain(url)] = str(res)
                    continue
                if isinstance(res, Exception):
                    errors[trust.domain(url)] = f"LLM read failed: {type(res).__name__}"
                    continue
                for p in res:
                    if (p.url, p.title) not in seen:
                        seen.add((p.url, p.title))
                        products.append(p)

        # Pages without product data in them (e.g. Amazon): the LLM reads the page text itself. This starts
        # right away, while the pages we couldn't open are fetched through a reader service at the same time.
        def worth_reading(pages: list, limit: int) -> list:
            """Well-known stores for this product first, and at most PER_STORE_READS pages per store, so the
            LLM's limited reads cover more stores."""
            pages = sorted(pages, key=lambda x: not any(trust._match(trust.domain(x[0]), d) for d in famous))
            out, per = [], {}
            for u, h in pages:
                d = trust.domain(u)
                if per.get(d, 0) < PER_STORE_READS and len(out) < limit:
                    per[d] = per.get(d, 0) + 1
                    out.append((u, h))
            return out

        first = worth_reading(unread, MAX_LLM_PAGES) if llm.available() else []
        if first:
            log.info("web: LLM reading %s", [trust.domain(u) for u, _ in first])
        first_tasks = [asyncio.ensure_future(read_one(u, h)) for u, h in first]
        later: list[tuple[str, str]] = []
        # pages we couldn't open but the search service already crawled: read its copy, no reader service needed
        have = [u for u in failed if u in crawled]
        for u in have:
            errors.pop(trust.domain(u), None)
            later.append((u, crawled[u]))
        failed = [u for u in failed if u not in crawled]
        if have:
            log.info("web: using the search service's copy of %s", [trust.domain(u) for u in have])
        if failed and llm.available():
            # the reader service gets part of the reading time and returns what it has by then; the LLM needs the rest
            failed = [u for u, _ in worth_reading([(u, "") for u in failed], MAX_READER_PAGES)]
            texts, why = await fetchers.read_pages(client, failed, deadline=READER_DEADLINE)
            for u, text in texts.items():
                errors.pop(trust.domain(u), None)
                later.append((u, text))
            for u, reason in why.items():
                errors[trust.domain(u)] = f"{errors.get(trust.domain(u), '')} / {reason}".strip(" /")
        later = worth_reading(later, max(0, MAX_LLM_READS - len(first)))
        if later:
            log.info("web: LLM reading (via reader) %s", [trust.domain(u) for u, _ in later])
        left = READING_DEADLINE - (time.monotonic() - t - TIMES["search"] - TIMES["pages"])
        later_res = await _within(max(left, 3), [read_one(u, h) for u, h in later])
        # the first batch has been running all along; it gets what's left of the reading time
        left = READING_DEADLINE - (time.monotonic() - t - TIMES["search"] - TIMES["pages"])
        collect(first, await _results(max(left, 1), first_tasks))
        collect(later, later_res)
        TIMES["reading"] = round(time.monotonic() - t - TIMES["search"] - TIMES["pages"], 1)
    await asyncio.to_thread(_remember_sites, urls, products, errors)  # the database may be far: off the event loop
    log.info("web: %d products from %d pages in %.1fs, failed=%s",
             len(products), len(urls) - len(errors), time.monotonic() - t, list(errors))
    return products, errors


_SLOW = re.compile(r"timeout|too slow|slow|LLM read failed", re.I)


def _take_turns(urls: list[str]) -> list[str]:
    """Order pages so every store gets a turn before any store gets a second page (each store's own pages keep
    the search's order), so the page budget covers more stores."""
    by_store: dict[str, list[str]] = {}
    for u in urls:
        by_store.setdefault(trust.domain(u), []).append(u)
    rounds = max((len(v) for v in by_store.values()), default=0)
    return [v[i] for i in range(rounds) for v in by_store.values() if i < len(v)]


async def _blocked_sites() -> set[str]:
    try:
        return await asyncio.to_thread(db.blocked_sites)
    except Exception as e:  # the database being down must not stop the search
        log.warning("could not read blocked sites: %s", e)
        return set()


def _remember_sites(urls: list[str], products: list[Product], errors: dict[str, str]) -> None:
    """Sites whose pages never open get skipped in later searches (db.blocked_sites); sites that worked are cleared.
    Being slow once (big pages, a busy moment, our own LLM or reader being busy) is not "can't open": it doesn't
    count. A well-known store is never skipped for that kind of failure (only for another country's prices): losing
    a big store costs far more than one wasted page."""
    worked = {trust.domain(p.url) for p in products}
    try:
        for d in {trust.domain(u) for u in urls}:
            why = errors.get(d)
            if d in worked:
                db.site_opened(d)
            elif why and why.startswith("prices in ") and not trust.is_known("https://" + d):
                # (a well-known brand or store has sections for many countries: samsung.com/in is not samsung.com/eg)
                db.site_failed(d, why, now_blocked=True)
            elif why and not _SLOW.search(why.split(" / ")[0]) and not trust.is_known("https://" + d):
                # (the first part is how the site answered US; what follows " / " is the reader service)
                db.site_failed(d, why)
    except Exception as e:  # never let bookkeeping break a search (SQLite or Postgres errors alike)
        log.warning("could not save blocked sites: %s", e)


async def _search_all(queries: list[str]) -> tuple[list[Product], dict[str, str]]:
    """Fallback without a web-search key: the built-in direct store searches."""
    t = time.monotonic()
    async with httpx.AsyncClient() as client:
        jobs = [(s, q) for q in queries for s in ALL_STORES]
        results = await asyncio.gather(*(s.search(client, q) for s, q in jobs), return_exceptions=True)
    products, errors, seen = [], {}, set()
    for (store, _), res in zip(jobs, results):
        if isinstance(res, Exception):
            errors[store.name] = f"{type(res).__name__}: {res}"[:200]
            continue
        for p in res:
            if (p.store, p.url, p.title) not in seen:
                seen.add((p.store, p.url, p.title))
                p.trusted, p.trust_score, p.trust_signals = True, 100, ["متجر مشهور"]
                products.append(p)
    log.info("stores: %d products for %s in %.1fs, errors=%s", len(products), queries, time.monotonic() - t, list(errors))
    return products, errors


BUDGET_WORDS = re.compile(
    r"تحت|اقل من|أقل من|مش اكتر من|مش أكتر من|ميزاني|حدود|بحد|لحد|في حدود|اكتر من|أكثر من|فوق|من \d+ ل|جنيه|جنية|ج\.?م|"
    r"\begp\b|\ble\b|budget|under|below|less than|max|over|above|between|\$|£",
    re.I)


def _states_budget(raw: str) -> bool:
    """A price limit counts only if the shopper used price words; '650' alone is usually a model or spec."""
    return bool(BUDGET_WORDS.search(raw))


def _slug(url: str) -> str:
    """Readable product link path, e.g. '/ar/p/olympic-electric-waterheater-50l-945105411'."""
    path = unquote(urlparse(url).path)
    return path if len(path) > 8 else ""


def _dedupe(products: list[Product]) -> list[Product]:
    """Same item found twice (two searches, or listing + product page): keep one, the one with a real link."""
    seen: dict[tuple, Product] = {}
    for p in products:
        key = (trust.domain(p.url), re.sub(r"\W+", " ", p.title.lower()).strip(), round(_paid(p)))
        if key not in seen or (seen[key].url.count("/") < p.url.count("/")):
            seen[key] = p
    # the same product under a short and a long title in one store (e.g. Amazon's search card vs. its page):
    # same shop, same price, and every word of the short title is in the long one -> keep the long one
    words = {id(p): set(re.findall(r"\w+", p.title.lower())) for p in seen.values()}
    items = list(seen.values())

    def covered(p: Product, q: Product) -> bool:  # p is a shorter title of q
        return (q is not p and trust.domain(q.url) == trust.domain(p.url) and round(_paid(q)) == round(_paid(p))
                and len(words[id(p)]) >= 3 and words[id(p)] <= words[id(q)]
                and (words[id(p)] != words[id(q)] or items.index(q) < items.index(p)))
    items = [p for p in items if not any(covered(p, q) for q in items)]
    # one product page reached in two languages (/ar/..., /en/...) or read twice with different titles (the
    # page's own <title> vs the product name): same shop, same page, same price -> one offer, the plainest title
    # (not two different items at one price on a listing page: those share the URL but both have plain names)
    out: list[Product] = []
    for p in items:
        page = _page_id(p.url)
        twin = next((q for q in out if page and _page_id(q.url) == page and trust.domain(q.url) == trust.domain(p.url)
                     and round(_paid(q)) == round(_paid(p)) and (q.url != p.url or "|" in q.title + p.title)), None)
        if twin is None:
            out.append(p)
        elif _title_noise(p.title) < _title_noise(twin.title):
            out[out.index(twin)] = p
    return out


_LANG_SEGMENT = re.compile(r"^(?:ar|en|eg|ar-eg|en-eg|eg-ar|eg-en|ar_eg|en_eg|egypt-ar|egypt-en)$", re.I)


def _page_id(url: str) -> str:
    """The product page's path without language parts, '' for a home/listing page (too short to tell)."""
    parts = urlparse(url)
    path = "/".join(s for s in unquote(parts.path).lower().split("/") if s and not _LANG_SEGMENT.match(s))
    return path + ("?" + parts.query if parts.query else "") if len(path) > 8 else ""


def _title_noise(title: str) -> tuple[int, int]:
    """Page titles carry ' | site name | slogan' parts; a product name doesn't."""
    return title.count("|"), -len(title)


def _save_debug(name: str, data: dict):
    """Keep the last plan/judge exchange so a surprising result can be explained later."""
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        (DEBUG_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _paid(p: Product) -> float:
    return p.coupon_price or p.price


def _candidates(pool: list[Product]) -> list[Product]:
    """What the judge sees, cheapest first: each store's PER_STORE_FIRST cheapest and every discounted offer get
    in before the remaining places go to the cheapest of the rest. One store's long listing page then can't fill
    every place, and a product that only one store sells still gets all the places it needs."""
    pool = sorted(pool, key=_paid)
    first, per = [], {}
    for p in pool:
        if p.discount_pct or per.get(p.store, 0) < PER_STORE_FIRST:
            per[p.store] = per.get(p.store, 0) + 1
            first.append(p)
    first = first[:MAX_CANDIDATES]
    taken = {id(p) for p in first}
    rest = [p for p in pool if id(p) not in taken][:MAX_CANDIDATES - len(first)]
    return sorted(first + rest, key=_paid)


DEAL_WEIGHT = 0.92  # a discounted offer is shown before others up to ~8% cheaper than it


def _show_order(picks: list[Product]) -> list[Product]:
    """The cheapest offer first; after it, discounted offers move ahead of slightly cheaper ones, and a run of
    one store's offers lets in another store's offer of about the same price (never a much dearer one)."""
    picks = sorted(picks, key=_paid)
    if len(picks) < 3:
        return picks
    rest = sorted(picks[1:], key=lambda p: _paid(p) * (DEAL_WEIGHT if p.discount_pct else 1))
    out = [picks[0]]
    while rest:
        p = rest[0]
        if len(out) >= 2 and out[-1].store == out[-2].store == p.store:
            other = next((q for q in rest[1:4] if q.store != p.store and _paid(q) <= _paid(p) * 1.1), None)
            p = other or p
        rest.remove(p)
        out.append(p)
    return out


def _discount(p: Product) -> int | None:
    """Real discount in %: against the store's own old price, or the checkout coupon."""
    ref = p.old_price if p.old_price and p.old_price > p.price else p.price
    pct = round((ref - _paid(p)) / ref * 100) if ref else 0
    return pct if 3 <= pct <= 90 else None


async def run(raw: str, plan: dict | None = None, budget_text: str | None = None) -> dict:
    """Search for one request. `plan` comes from plan_turn() when the chat already planned it;
    `budget_text` is what the shopper actually wrote (all their messages), used to check a budget is real."""
    TIMES.clear()
    if plan is None:
        llm.reset_usage()  # (a chat turn resets it before planning, in main._chat_turn)
    t0 = time.monotonic()
    if plan is None:
        if not llm.available():
            return await run_rules(raw, note="شغالين من غير LLM (مفيش مفتاح في .env)، فالفهم بسيط.")
        try:
            log.info("agent: understanding %r", raw)
            plan = await plan_turn([{"role": "user", "text": raw}])
        except llm.LLMError as e:
            return await run_rules(raw, note=f"الـ LLM مردش ({e})، فاستخدمنا الفهم البسيط.")
    seed: list[dict] = []
    if websearch.available() and llm.available():
        plan, seed = await _check_plan(plan, plan.get("request") or raw)
    queries = [q for q in plan.get("queries", []) if isinstance(q, str) and q.strip()][:2] or [raw]
    web_queries = [q for q in plan.get("web_queries", []) if isinstance(q, str) and q.strip()][:MAX_WEB_QUERIES] or queries
    codes = [c for c in plan.get("model_codes", []) if isinstance(c, str) and c.strip()]

    TIMES["understand"] = round(time.monotonic() - t0, 1)
    if websearch.available():
        source = "web"
        want = plan.get("request") or raw
        if reqs := [r for r in plan.get("requirements") or [] if isinstance(r, str)]:
            want += " (" + "; ".join(reqs[:6]) + ")"
        products, errors = await _web_find(web_queries, seed, _famous_sites(plan), want=want,
                                           named=_domains(plan.get("shopper_stores")))
        products = _dedupe(products)
    else:
        source = "stores"
        products, errors = await _search_all(queries)

    maxp, minp = plan.get("max_price"), plan.get("min_price")
    if (maxp is not None or minp is not None) and not _states_budget(budget_text or raw):
        log.info("agent: ignoring budget %s-%s the LLM guessed (no price words in the request)", minp, maxp)
        maxp = minp = None
    pool = [p for p in products if (maxp is None or _paid(p) <= maxp) and (minp is None or p.price >= minp)]
    sold_out = [p for p in pool if p.in_stock is False]
    if sold_out:
        log.info("agent: dropped %d sold-out items: %s", len(sold_out), [p.store for p in sold_out][:10])
    pool = [p for p in pool if p.in_stock is not False]
    shady = [p for p in pool if not p.trusted and p.trust_score is not None and p.trust_score < MIN_TRUST]
    if shady:
        log.info("agent: dropped %d items from low-trust shops: %s", len(shady), sorted({p.store for p in shady}))
        pool = [p for p in pool if not any(p is q for q in shady)]
    for p in pool:
        p.discount_pct = _discount(p)
    pool = _candidates(pool)

    picks, summary, typical, follow_ups = [], "", None, []
    if pool:
        items = "\n".join(
            f"{i}. [{p.store}] trust {p.trust_score if p.trust_score is not None else '?'}"
            + (f" ({', '.join(p.trust_signals[:5])})" if p.trust_signals and not p.trusted else "")
            + (f" | CONDITION: {p.condition}" if p.condition else "") + f" | {p.price:g}"
            + (f" ({p.coupon_price:g} with checkout coupon)" if p.coupon_price else "")
            + (f" | -{p.discount_pct}% off" if p.discount_pct else "") + f" | {p.title[:140]}"
            + (f" | link: {_slug(p.url)[:90]}" if _slug(p.url) else "")
            for i, p in enumerate(pool))
        log.info("agent: judging %d candidates", len(pool))
        _progress(f"بقارن {len(pool)} عرض من {len({p.store for p in pool})} متجر وبشيل اللي مش مطابق...")
        tj = time.monotonic()
        try:
            prompt = JUDGE_PROMPT.format(
                raw=raw, req=json.dumps(plan.get("requirements", []), ensure_ascii=False),
                avoid=json.dumps(plan.get("avoid", []), ensure_ascii=False), minp=minp, maxp=maxp, items=items,
                codes=json.dumps(codes, ensure_ascii=False),
                suggested=json.dumps(plan.get("suggested_products") or [], ensure_ascii=False))
            verdict = await llm.ask_json(prompt, timeout=20, budget=40, prefer="strong", step="judge")  # the big call: give it time, it decides quality
            TIMES["judge"] = round(time.monotonic() - tj, 1)
            _save_debug("last_judge.json", {"request": raw, "plan": plan, "prompt": prompt, "verdict": verdict})
            log.info("agent: judge kept %d of %d (model %s)", len(verdict.get("matches") or []), len(pool),
                     llm.last_model())
            summary = verdict.get("summary", "")
            follow_ups = _strs(verdict.get("suggestions"), 4, 60)
            if isinstance(verdict.get("typical_price"), (int, float)) and verdict["typical_price"] > 0:
                typical = float(verdict["typical_price"])
            best = verdict.get("best_deal_id")
            if isinstance(best, int) and 0 <= best < len(pool):
                pool[best].best_deal = True
            alternatives = not verdict.get("matches")
            for m in verdict.get("matches") or verdict.get("alternatives") or []:
                i = m.get("id")
                if isinstance(i, int) and 0 <= i < len(pool):
                    p = pool[i]
                    p.note = m.get("note")
                    p.unconfirmed = m.get("sure") is False
                    p.alternative = alternatives
                    picks.append(p)
        except llm.LLMError as e:
            # the simple word rules miss many titles (Arabic vs English, "2000 وات" vs "2000W"); the candidates
            # all came from searches for this request, so show the cheapest of them rather than nothing
            picks = _fallback_pick(pool, [raw, plan.get("request") or ""] + web_queries)[:JUDGE_DOWN_RESULTS]
            for p in picks:
                p.unconfirmed = True
            log.warning("agent: judge failed: %s", e)
            summary = ("الموديلات المجانية مشغولة دلوقتي فمقدرتش أراجع النتايج بدقة، فدي مرتبة بالسعر بس "
                       "واتأكد من الموديل في كل واحدة. جرب تاني كمان دقيقة لنتيجة أدق.")
    picks = list({id(p): p for p in picks}.values())
    gone = 0
    if picks and VERIFY_PICKS:
        _progress(f"بفتح صفحة كل نتيجة ({len(picks)}) وبتأكد إنها متوفرة وإن سعرها لسه زي ما هو...")
        tv = time.monotonic()
        picks, gone = await _verify_stock(picks)
        TIMES["verify"] = round(time.monotonic() - tv, 1)
        if gone:
            summary = (summary + " " if summary else "") + (
                f"(شلت {gone} عرض لقيته خلصان لما فتحت صفحته." + ("" if picks else " ومفيش عرض تاني متوفر دلوقتي.") + ")")
    _flag_price_outliers(picks, typical)
    picks = _show_order(picks)
    return {
        "mode": "llm", "source": source, "timings": dict(TIMES),
        "llm_usage": {"calls": llm.USAGE["calls"], "tokens": llm.USAGE["tokens"], "by_step": dict(llm.USAGE["by_step"])},
        "understood": plan.get("understood", ""),
        "request": plan.get("request") or raw,
        "suggested_products": plan.get("suggested_products") or [],
        "queries": web_queries if source == "web" else queries,
        "requirements": plan.get("requirements", []),
        "max_price": maxp, "min_price": minp,
        "summary": summary,
        "suggestions": follow_ups,
        "candidates_checked": len(pool),
        "sold_out_dropped": gone,
        "results": [p.to_dict() for p in picks],
        "store_errors": errors,
    }


VERIFY_PICKS = True    # open each result's own page before showing it (tests turn it off unless they check it)
VERIFY_MAX = 12        # results checked per search (the cheapest first)
VERIFY_DEADLINE = 7    # opening the result pages ourselves
VERIFY_READER_DEADLINE = 6  # pages we couldn't open, through the reader service


_PRODUCT_ID = re.compile(r"[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*")


def _product_ids(url: str) -> set[str]:
    """Path parts that look like a product number (8+ letters/digits with a digit, e.g. B09YXNHF8N, 945105411):
    the same product's link differs between a search and a store page (language part, tracking), its number not."""
    parts = re.split(r"[/.?&=]", urlparse(url).path)
    return {x.lower() for x in parts if len(x) >= 8 and _PRODUCT_ID.fullmatch(x) and not x.startswith("ref")}


def _same_product_link(a: str, b: str) -> bool:
    if trust.domain(a) != trust.domain(b):
        return False
    return _page_id(a) == _page_id(b) or bool(_product_ids(a) & _product_ids(b))


async def price_from_search(client: httpx.AsyncClient, url: str, title: str) -> Product | None:
    """A followed product whose page we may not open ourselves (robots.txt, bot wall, no price data in it): the
    web search service's own fresh copy of that page, read by the LLM, the same way searches read such pages.
    One search + one LLM read; None when the search doesn't return that page."""
    if not (websearch.available() and llm.available() and title):
        return None
    hits = await websearch.search(client, title, max_results=5, include_domains=[trust.domain(url)])
    hit = next((h for h in hits if _same_product_link(h["url"], url) and len(h.get("content") or "") >= 200), None)
    if hit is None:
        return None
    products = await _llm_read_page(hit["url"], hit["content"], title)
    same = [p for p in products if _same_product_link(p.url, url)] or (products if len(products) == 1 else [])
    if not same:
        return None
    p = same[0]
    p.url = url
    return p


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,.")


def _price_moved(html: str, before: float, now: float) -> bool:
    """Take the product page's price over the one we had only when the shopper can see it on the page and can't
    see the old one: a page's data can hold another product's or another option's price (related items, the
    cheapest size), and the price we had may be the right one."""
    if not before / 2 <= now <= before * 2:
        return False
    seen = extract_mod._visible_numbers(html.translate(_ARABIC_DIGITS))
    return now in seen and before not in seen


async def _verify_stock(picks: list[Product]) -> tuple[list[Product], int]:
    """Before results are shown, open each one's own product page: drop it when the store now says sold out,
    take the page's current price when it changed, and mark it `verified` when the page confirmed it.
    A listing page often shows stale stock/prices, and a sold-out item is a wasted click for the shopper.
    Returns (picks still for sale, how many were dropped as sold out). Results whose page can't be checked stay,
    unverified, so the page can ask the shopper to check them."""
    for p in picks:
        p.verified = False
    todo = sorted((p for p in picks if _page_id(p.url)), key=_paid)[:VERIFY_MAX]
    if not todo:
        return picks, 0
    sold: set[int] = set()
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(6)
        pages = await _within(VERIFY_DEADLINE, [_fetch_page(client, sem, p.url) for p in todo])
        retry = []
        for p, res in zip(todo, pages):
            if isinstance(res, Exception):
                retry.append(p)
                continue
            products, html = res
            pid = _page_id(p.url)
            by_link = [q for q in products if _page_id(q.url) == pid]
            same = by_link or (products if len(products) == 1 else [])
            if same:
                q = min(same, key=_paid)
                if q.in_stock is False:
                    sold.add(id(p))
                    continue
                if by_link and abs(q.price - p.price) >= 1 and _price_moved(html, p.price, q.price):
                    log.info("agent: %s price now %g (was %g)", p.store, q.price, p.price)
                    p.price, p.old_price, p.coupon_price = q.price, q.old_price, q.coupon_price
                    p.discount_pct = _discount(p)
                elif abs(q.price - p.price) >= 1:
                    log.info("agent: %s page data says %g, page shows %g: kept", p.store, q.price, p.price)
                p.in_stock = True if q.in_stock is None else q.in_stock
                p.verified = True
                continue
            said = extract_mod._page_says_sold_out(html, extract_mod._visible_text(html), p.title)
            if said:
                sold.add(id(p))
            elif said is False:  # the title is on the page and no sold-out words next to it
                p.verified = True
        if retry:
            texts, _ = await fetchers.read_pages(client, [p.url for p in retry], deadline=VERIFY_READER_DEADLINE)
            for p in retry:
                if (text := texts.get(p.url)) is not None:
                    said = extract_mod._page_says_sold_out(text, text, p.title)
                    if said:
                        sold.add(id(p))
                    elif said is False:
                        p.verified = True
    if sold:
        log.info("agent: %d results sold out on their own page: %s", len(sold),
                 [p.store for p in picks if id(p) in sold])
    return [p for p in picks if id(p) not in sold], len(sold)


_SEARCH_NOISE = {"price", "prices", "egypt", "سعر", "اسعار", "أسعار", "مصر", "في", "من", "the", "and", "for", "with",
                 "best", "buy", "online", "جنيه", "تحت", "عايز"}


def _fallback_pick(pool: list[Product], queries: list[str], keep: float = 0.6) -> list[Product]:
    """No model could judge: keep the candidates whose title has the distinctive words of one of the searches.
    A word every candidate shares ('55', 'inch', 'tv') says little; a rarer one ('samsung') decides, so other
    brands or grades drop out without any product-specific rule. Cheapest first."""
    import math
    titles = [set(re.findall(r"\w+", p.title.lower())) for p in pool]
    n = len(pool)

    def weight(w: str) -> float:
        return math.log((n + 1) / (sum(w in t for t in titles) + 0.5))
    best = [0.0] * n
    for q in queries:
        words = {w for w in re.findall(r"\w+", q.lower()) if w not in _SEARCH_NOISE and (len(w) > 1 or w.isdigit())}
        total = sum(max(weight(w), 0.05) for w in words)
        if not total:
            continue
        for i, t in enumerate(titles):
            best[i] = max(best[i], sum(max(weight(w), 0.05) for w in words if w in t) / total)
    return sorted((p for p, s in zip(pool, best) if s >= keep), key=_paid)


OUTLIER_RATIO = 0.6  # a match under 60% of the typical (median) price of the other matches looks wrong
UNIT_ERROR = 10     # over 10x the typical price of the other matches: a listing in the wrong unit
TOO_CHEAP = 0.5      # under half of what the judge says a genuine new one costs: likely a copy or a listing mistake


def _flag_price_outliers(picks: list[Product], typical: float | None = None) -> None:
    """Safety net after the judge: an offer far below what every other store asks for the same product (or, with
    few offers, far below what the judge says a genuine new one costs in Egypt) is usually a copy, a listing
    mistake, another variant or a used item. Keep it, but say so and never call it the best deal."""
    if not picks:
        return
    alternatives = any(p.alternative for p in picks)  # different products: only what a genuine one costs applies
    # 10x what the others (or a genuine one) cost is a unit mistake (piasters read as pounds, a bulk lot): never
    # a real offer for the same thing, so it goes
    for p in [] if alternatives else list(picks):
        others = sorted(_paid(q) for q in picks if q is not p)
        ref = others[len(others) // 2] if others else typical
        if ref and _paid(p) > UNIT_ERROR * ref:
            picks.remove(p)
            log.info("agent: dropped %s at %s (about %s elsewhere): wrong unit", p.store, _paid(p), ref)
    if not picks:
        return
    prices = sorted(_paid(p) for p in picks)
    for p in picks:
        others = [x for x in prices if x != _paid(p)] or prices
        median = others[len(others) // 2]
        if len(picks) >= 3 and not alternatives and _paid(p) < OUTLIER_RATIO * median:
            ref = median
        elif typical and _paid(p) < TOO_CHEAP * typical:
            ref = typical
        else:
            continue
        p.unconfirmed = True
        p.best_deal = False
        p.note = (f"السعر أقل بكتير من سعره المعتاد (حوالي {ref:,.0f} ج.م)، ممكن يكون تقليد أو غلط في الإعلان أو "
                  "موديل/حالة مختلفة. اتأكد من صفحة المنتج والمتجر قبل الشراء.")
        log.info("agent: price outlier %s %s (reference %s)", p.store, _paid(p), ref)


async def run_rules(raw: str, note: str) -> dict:
    q: ParsedQuery = parse_query(raw)
    products, errors = await _search_all([q.search_text])
    picks = rank(products, q)
    return {
        "mode": "rules", "source": "stores",
        "understood": note,
        "request": raw,
        "suggested_products": [],
        "queries": [q.search_text],
        "requirements": q.must_have,
        "max_price": q.max_price, "min_price": q.min_price,
        "summary": "",
        "candidates_checked": len(products),
        "results": [p.to_dict() for p in picks],
        "store_errors": errors,
    }
