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

from . import fetchers, llm, trust, websearch
from .extract import extract_products
from .models import ParsedQuery, Product
from .query import parse_query
from .ranking import rank
from .stores import ALL_STORES
from .stores.base import DEBUG_DIR, HEADERS

log = logging.getLogger("uvicorn.error")
MAX_CANDIDATES = 60
MAX_PAGES = 20          # store pages opened per search
MAX_PER_PAGE = 15       # products kept from one page (listing pages can have many)
WEB_DEBUG = DEBUG_DIR / "web"

UNDERSTAND_PROMPT = """You are the shopping assistant and search planner of a price-comparison agent for shoppers in Egypt.
Shoppers often don't know the product's name or how stores list it: they describe what they need
("تيشرت راوند بيزك اسود", "موبايل مش غالي وبطاريته تكمل يومين"). Understand what they really want and turn it
into concrete specs and searches that find it in Egyptian online stores.

The chat is a conversation. A new message usually CHANGES THE CURRENT REQUEST instead of starting over:
"سامسونج بس" / "عايزه أرخص" / "بلاش الأحمر" / "مقاس XL" / "تحت 5000" -> keep everything from the current request
that still applies and apply the change. Start fresh only if the shopper clearly asks for a different product.
Never change the product type the shopper named: if you don't recognize a model name (e.g. "anker r50i nc"),
search the shopper's own words as they are instead of guessing a category.
Quality words are real requirements, never drop them: "ماركة نضيفة / معروفة / أصلي / كويس" means well-known
quality brands, so name those brands in suggested_products and the queries, and add a requirement like
"well-known brand (e.g. ...)", so no-name products don't count as a match.
Ask a question ONLY if no sensible search is possible (e.g. "عايز هدية"); otherwise search and state assumptions.
If the newest message is a QUESTION about results already shown or about buying advice ("أول نتيجة دي أصلية؟",
"الفرق بين الاتنين إيه؟", "أنهي أحسن؟", "عليها ضمان؟") and does not ask for different products, use "answer":
reply from the conversation and your knowledge, without a new search. Be honest: if a price looks too low for that
product, say it may be a mistake, a different variant or a used/refurbished listing and how to check on the store page.

{chat}
Reply JSON:
{
  "action": "search" or "ask" or "answer",
  "answer": "if answer: 1-4 short Egyptian Arabic sentences answering the shopper's question, else null",
  "question": "if ask: one short Egyptian Arabic question, else null",
  "options": ["if ask: 2-4 short quick answers the shopper can tap, else []"],
  "request": "the full standalone request in Egyptian Arabic built from the whole conversation, e.g.
              'موبايل سامسونج سعره مش غالي وبطاريته كبيرة تكمل يومين'",
  "understood": "one short Egyptian Arabic sentence saying what you'll look for (mention the change if a follow-up)",
  "need_based": true if the shopper described a need/use rather than naming a specific product,
  "suggested_products": ["if need_based: 3-6 specific products sold in Egypt now that fit the need, with the
     spec that makes them fit, e.g. 'Samsung Galaxy A16 (5000 mAh)', 'Xiaomi Redmi 14C (5160 mAh)'. Names and
     specs only, NEVER prices (you don't know today's prices); else []"],
  "stores": ["4-8 domains of the biggest well-known online stores in Egypt for THIS kind of product: big
     marketplaces plus the category's own leaders and the brands' official Egypt stores (electronics, fashion,
     beauty, furniture, groceries... each has different leaders). Only real domains you are sure of"],
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


def _chat_block(messages: list[dict], current: str) -> str:
    lines = [("Shopper: " if m.get("role") == "user" else "Assistant: ") + str(m.get("text", ""))[:800]
             for m in messages[-MAX_HISTORY:]]
    return ("Current request (from the previous search): " + (current or "none") + "\n"
            "Conversation (oldest first, the last line is the newest message):\n" + "\n".join(lines) + "\n")


async def plan_turn(messages: list[dict], current: str = "") -> dict:
    """One LLM call: understand the newest message in the context of the chat, merge it into the request,
    and plan the searches. Raises llm.LLMError when no model answers."""
    plan = await llm.ask_json(UNDERSTAND_PROMPT.replace("{chat}", _chat_block(messages, current)), timeout=12,
                                prefer="groq")  # gpt-oss-120b on Groq: strong and answers in ~1 s
    last = next((m.get("text", "") for m in reversed(messages) if m.get("role") == "user"), "")
    # names only: a price the LLM guessed would mislead the shopper and the judge
    plan["suggested_products"] = [_PRICE_TAIL.sub("", x).strip() for x in plan.get("suggested_products") or []
                                  if isinstance(x, str) and x.strip()]
    if not isinstance(plan.get("request"), str) or not plan["request"].strip():
        plan["request"] = last
    if plan.get("action") == "answer" and isinstance(plan.get("answer"), str) and plan["answer"].strip():
        plan["request"] = current or plan["request"]  # a question about the results keeps the current request
    elif plan.get("action") != "ask" or not plan.get("question"):
        plan["action"] = "search"
    log.info("agent: plan %s -> %r", plan["action"], plan.get("question") or plan.get("answer") or plan["request"])
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
Reply JSON:
{{"understood": "one short Egyptian Arabic sentence saying what you'll look for",
  "web_queries": ["2, or 3 when several brands/products are wanted: Arabic as Egyptian shoppers and stores write
                   it, and precise English with the official model number if known"],
  "stores": ["4-8 domains of the biggest well-known Egyptian online stores for this product (from the results
              and your knowledge)"],
  "model_codes": ["official model / part numbers of this exact variant, else []"],
  "requirements": ["hard requirements a matching product must meet, in English"],
  "changed": true if the first reading was wrong about the product or the variant}}"""
CHECK_HITS = 8
MAX_WEB_QUERIES = 3  # each one = 2 searches (well-known stores + other shops)
MARKETPLACES = ["amazon.eg", "noon.com", "jumia.com.eg"]  # sell almost everything: always in the well-known group
_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")


def _famous_sites(plan: dict) -> list[str]:
    """Well-known stores for this product: the big marketplaces + the ones the LLM named for this category.
    Falls back to our own list when the LLM named none."""
    named = []
    for d in plan.get("stores") or []:
        d = str(d).lower().strip().removeprefix("https://").removeprefix("http://").removeprefix("www.").split("/")[0]
        if _DOMAIN.match(d) and trust.is_shop_candidate("https://" + d):
            named.append(d)
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
        fix = await llm.ask_json(prompt, timeout=10, budget=15, prefer="groq")
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

Return EVERY candidate that fits (often 5-20), not just the best few: the shopper wants to compare.
A candidate fits when it (1) is what the shopper asked for. When the shopper described a need (e.g. "battery lasts
two days"), any product that meets the requirements fits, not only the suggested ones; judge from the title/specs
and your knowledge of that model. Variants of one product line are DIFFERENT products
(e.g. knob/mechanical vs digital, Hero vs Hero Plus/Turbo/Lite, 50L vs 30L): drop any candidate whose title, model
number or link shows the wrong variant. Use model numbers in the title or link (e.g. 945105411 = Hero 50L mechanical)
to confirm. Store titles are often short: if nothing shows the variant either way, KEEP it with "sure": false and
say in the note what to check. Set "sure": true only when the title/model/link confirms the exact variant. And
(2) come from a legitimate retailer selling new items in Egypt. Smaller, lesser-known shops are WELCOME and often
cheaper: keep them when their trust is 35+ (real Egyptian phone, cash on delivery, return policy, warranty...).
Drop shops with trust below 30, copies/replicas, classifieds/used listings, blogs, price-list sites, and prices
suspiciously far below the others for the same item. A branded item sold by a shop that isn't the brand or a big
retailer, far below what that brand charges in Egypt (common with clothes, shoes, perfume, bags), is likely a copy:
drop it, or keep it with "sure": false and a note saying it may not be original. Check every price against what that product really costs in
Egypt today (your market knowledge + the other candidates): a price far below it (e.g. an 18 ft fridge for 2,000 EGP)
is a wrong currency, a monthly installment, a deposit or an accessory, so drop it. Drop used / refurbished / open-box items unless the shopper
asked for them. Real discounts are a plus. Reply JSON:
{{"matches": [{{"id": <id>, "sure": true|false, "note": "very short Egyptian Arabic note on why it fits, or what to check"}}],
  "alternatives": [{{"id": <id>, "note": "..."}}],  // ONLY if matches is empty: closest good options from legit stores,
                                                  // e.g. used/refurbished from a real shop, or a close spec
  "best_deal_id": <id of the best value-for-money pick: low price + real discount + trustworthy shop, or null>,
  "summary": "1-2 Egyptian Arabic sentences: the best cheap option from a trusted store and any useful advice
              (name stores and prices, never the candidate ids)"}}"""


PAGE_READ_PROMPT = """This is the text of an Egyptian online store page (links shown as [link: url] or markdown [text](url)).
Extract the products for sale on it that relate to: {query}
Reply JSON: {{"products": [{{"title": "...", "price": number (current price in EGP as shown, no old price),
"url": "the product link from the text or null", "condition": "new" | "used" | "refurbished",
"in_stock": true | false | null (false if the page shows out of stock / sold out / غير متوفر / نفذت الكمية)}}]}}
At most 15 products. Use only prices written on the page; never guess. Empty list if none, or if the prices are in
another currency (SAR, AED, USD...) or are monthly installments.
PAGE:
{text}"""
MAX_READER_PAGES = 8  # failed pages retried through a reader service per search
MAX_LLM_PAGES = 6  # pages with no structured data that the LLM reads itself (free-tier friendly)
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


def _page_text(html: str, focus: list[str], limit: int = 4500) -> str:
    """Readable page text with links kept, cut around the part that mentions the product."""
    html = re.sub(r"<(script|style|noscript|svg|head)\b.*?</\1>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<a\b[^>]*href=[\"']([^\"'#?]+)[^\"']*[\"'][^>]*>(.*?)</a>", r" \2 [link: \1] ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", unescape(text))
    low = text.lower()
    start = 0
    for w in sorted(focus, key=len, reverse=True):  # the most specific word, e.g. "zbook" not "hp"
        if (i := low.find(w.lower())) != -1:
            start = max(0, i - 300)
            break
    return text[start:start + limit]


async def _llm_read_page(url: str, html: str, query: str, spread: int = 0) -> list[Product]:
    words = [w for w in re.split(r"\W+", query) if len(w) > 2]
    data = await llm.ask_json(PAGE_READ_PROMPT.format(query=query, text=_page_text(html, words)), prefer="bulk", spread=spread)
    out = []
    items = data.get("products")
    for x in (items if isinstance(items, list) else [])[:15]:
        if not isinstance(x, dict):
            continue
        price = x.get("price")
        if not isinstance(price, (int, float)) or price < 5 or not x.get("title"):
            continue
        link = urljoin(url, x.get("url") or url)
        cond = x.get("condition")
        out.append(Product(store=trust.store_name(link), title=str(x["title"])[:200], price=float(price), url=link,
                           trusted=trust.is_known(link), condition=cond if cond in ("used", "refurbished") else None,
                           in_stock=x.get("in_stock") if isinstance(x.get("in_stock"), bool) else None))
    score, signals = trust.page_signals(html, url, False)
    for p in out:
        p.trust_score, p.trust_signals = (100, ["متجر مشهور"]) if p.trusted else (score, signals)
    return out


async def _fetch_page(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> tuple[list[Product], str]:
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
            out.append(min(group, key=lambda p: p.coupon_price or p.price))
    return out


async def _web_find(web_queries: list[str], seed: list[dict] | None = None,
                    famous: list[str] | None = None) -> tuple[list[Product], dict[str, str]]:
    t = time.monotonic()
    errors: dict[str, str] = {}
    async with httpx.AsyncClient() as client:
        # Two groups searched side by side: famous stores, and every other shop. The judge compares them all.
        famous = famous or trust.FAMOUS_SITES
        groups = {"famous": dict(include_domains=famous),
                  "other": dict(exclude_domains=famous + trust.CLASSIFIEDS)}
        jobs = [(g, websearch.search(client, q, max_results=8, **kw)) for q in web_queries for g, kw in groups.items()]
        found = await asyncio.gather(*(j for _, j in jobs), return_exceptions=True)
        TIMES["search"] = round(time.monotonic() - t, 1)
        picked: dict[str, list[str]] = {"famous": [], "other": []}
        jobs, found = [("check", None)] + jobs, [seed or []] + list(found)  # the plan-check search's hits count too
        for (group, _), res in zip(jobs, found):
            if isinstance(res, Exception):
                errors[f"web search ({group})"] = str(res)[:200]
                continue
            for hit in res:
                u = hit["url"]
                if trust.is_shop_candidate(u) and all(u not in lst for lst in picked.values()):
                    big = trust.is_known(u) or any(trust._match(trust.domain(u), d) for d in famous)
                    picked["famous" if big else "other"].append(u)
        # Same page budget for each group, alternating, so neither side crowds the other out.
        known, small = picked["famous"], picked["other"]
        urls = [u for pair in zip(known, small) for u in pair] + known[len(small):] + small[len(known):]
        urls = urls[:MAX_PAGES]
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
                return await _llm_read_page(u, h, web_queries[0], spread=n_read[0])

        def collect(batch, results):
            for (url, _), res in zip(batch, results):
                if isinstance(res, Exception):
                    errors[trust.domain(url)] = f"LLM read failed: {type(res).__name__}"
                    continue
                for p in res:
                    if (p.url, p.title) not in seen:
                        seen.add((p.url, p.title))
                        products.append(p)

        # Pages without product data in them (e.g. Amazon): the LLM reads the page text itself. This starts
        # right away, while the pages we couldn't open are fetched through a reader service at the same time.
        first = unread[:MAX_LLM_PAGES] if llm.available() else []
        if first:
            log.info("web: LLM reading %s", [trust.domain(u) for u, _ in first])
        first_tasks = [asyncio.ensure_future(read_one(u, h)) for u, h in first]
        later: list[tuple[str, str]] = []
        if failed and llm.available():
            # the reader service gets part of the reading time and returns what it has by then; the LLM needs the rest
            texts, why = await fetchers.read_pages(client, failed[:MAX_READER_PAGES], deadline=READER_DEADLINE)
            for u, text in texts.items():
                errors.pop(trust.domain(u), None)
                later.append((u, text))
            for u, reason in why.items():
                errors[trust.domain(u)] = f"{errors.get(trust.domain(u), '')} / {reason}".strip(" /")
        later = later[:max(0, MAX_LLM_PAGES + 2 - len(first))]
        if later:
            log.info("web: LLM reading (via reader) %s", [trust.domain(u) for u, _ in later])
        left = READING_DEADLINE - (time.monotonic() - t - TIMES["search"] - TIMES["pages"])
        later_res = await _within(max(left, 3), [read_one(u, h) for u, h in later])
        # the first batch has been running all along; it gets what's left of the reading time
        left = READING_DEADLINE - (time.monotonic() - t - TIMES["search"] - TIMES["pages"])
        collect(first, await _results(max(left, 1), first_tasks))
        collect(later, later_res)
        TIMES["reading"] = round(time.monotonic() - t - TIMES["search"] - TIMES["pages"], 1)
    log.info("web: %d products from %d pages in %.1fs, failed=%s",
             len(products), len(urls) - len(errors), time.monotonic() - t, list(errors))
    return products, errors


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
    return [p for p in items if not any(covered(p, q) for q in items)]


def _save_debug(name: str, data: dict):
    """Keep the last plan/judge exchange so a surprising result can be explained later."""
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        (DEBUG_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def _paid(p: Product) -> float:
    return p.coupon_price or p.price


def _discount(p: Product) -> int | None:
    """Real discount in %: against the store's own old price, or the checkout coupon."""
    ref = p.old_price if p.old_price and p.old_price > p.price else p.price
    pct = round((ref - _paid(p)) / ref * 100) if ref else 0
    return pct if 3 <= pct <= 90 else None


async def run(raw: str, plan: dict | None = None, budget_text: str | None = None) -> dict:
    """Search for one request. `plan` comes from plan_turn() when the chat already planned it;
    `budget_text` is what the shopper actually wrote (all their messages), used to check a budget is real."""
    TIMES.clear()
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
        products, errors = await _web_find(web_queries, seed, _famous_sites(plan))
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
    for p in pool:
        p.discount_pct = _discount(p)
    pool.sort(key=_paid)
    pool = pool[:MAX_CANDIDATES]

    picks, summary = [], ""
    if pool:
        items = "\n".join(
            f"{i}. [{p.store}] trust {p.trust_score if p.trust_score is not None else '?'}"
            + (f" ({', '.join(p.trust_signals[:5])})" if p.trust_signals and not p.trusted else "")
            + (f" | CONDITION: {p.condition}" if p.condition else "") + f" | {p.price:g}"
            + (f" ({p.coupon_price:g} with checkout coupon)" if p.coupon_price else "")
            + (f" | -{p.discount_pct}% off" if p.discount_pct else "") + f" | {p.title}"
            + (f" | link: {_slug(p.url)}" if _slug(p.url) else "")
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
            verdict = await llm.ask_json(prompt, timeout=20, budget=40)  # the big call: give it time, it decides quality
            TIMES["judge"] = round(time.monotonic() - tj, 1)
            _save_debug("last_judge.json", {"request": raw, "plan": plan, "prompt": prompt, "verdict": verdict})
            log.info("agent: judge kept %d of %d (model %s)", len(verdict.get("matches") or []), len(pool),
                     llm.last_model())
            summary = verdict.get("summary", "")
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
            picks = rank(pool, parse_query(raw))
            log.warning("agent: judge failed: %s", e)
            summary = ("الموديلات المجانية مشغولة دلوقتي فمقدرتش أراجع النتايج بدقة، فدي مرتبة بالسعر بس "
                       "واتأكد من الموديل في كل واحدة. جرب تاني كمان دقيقة لنتيجة أدق.")
    picks = list({id(p): p for p in picks}.values())
    _flag_price_outliers(picks)
    picks.sort(key=_paid)
    return {
        "mode": "llm", "source": source, "timings": dict(TIMES),
        "understood": plan.get("understood", ""),
        "request": plan.get("request") or raw,
        "suggested_products": plan.get("suggested_products") or [],
        "queries": web_queries if source == "web" else queries,
        "requirements": plan.get("requirements", []),
        "max_price": maxp, "min_price": minp,
        "summary": summary,
        "candidates_checked": len(pool),
        "results": [p.to_dict() for p in picks],
        "store_errors": errors,
    }


OUTLIER_RATIO = 0.6  # a match under 60% of the typical (median) price of the other matches looks wrong


def _flag_price_outliers(picks: list[Product]) -> None:
    """Safety net after the judge: an offer far below what every other store asks for the same product is usually
    a listing mistake, another variant or a used item. Keep it, but say so and never call it the best deal."""
    if len(picks) < 3 or any(p.alternative for p in picks):
        return
    prices = sorted(_paid(p) for p in picks)
    for p in picks:
        others = [x for x in prices if x != _paid(p)] or prices
        median = others[len(others) // 2]
        if _paid(p) < OUTLIER_RATIO * median:
            p.unconfirmed = True
            p.best_deal = False
            p.note = (f"السعر أقل بكتير من باقي المتاجر (حوالي {median:,.0f} ج.م)، ممكن يكون غلط في الإعلان أو موديل/حالة "
                      "مختلفة. اتأكد من صفحة المنتج قبل الشراء.")
            log.info("agent: price outlier %s %s (median %s)", p.store, _paid(p), median)


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
