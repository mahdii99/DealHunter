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
from dataclasses import fields
import contextvars
import json
import logging
import re
import time
from html import unescape
from urllib.parse import quote_plus, unquote, urljoin, urlparse

import httpx

from . import buybox
from . import cards as cards_mod
from . import catalog, db, fetchers, llm, market, noon_index, recipes, trust, websearch
from . import extract as extract_mod
from .extract import extract_products
from .models import ParsedQuery, Product
from .query import AR_UNITS, AR_WORDS, BRANDS, english_words, kinds_in, lacks_wanted, names_brand, parse_query, same_kind, shows_size, wanted_sizes
from .ranking import rank
from .stores import ALL_STORES
from .stores.amazon import parse_amazon_cards
from .stores.base import DEBUG_DIR, HEADERS
from .utils import normalize

log = logging.getLogger("uvicorn.error")
MAX_CANDIDATES = 50  # cheapest first; the judge reads every line, so more = more tokens per search
PER_STORE_FIRST = 8  # candidates from one store before other stores get their turn (more stores reach the judge)
JUDGE_DOWN_RESULTS = 12  # no model could judge: the cheapest candidates, all marked "check it"
MAX_PAGES = 20          # store pages opened per search
MAX_PER_PAGE = 15       # products kept from one page (listing pages can have many)
FAMOUS_RESULTS = 15     # search results for the well-known stores group (one search shared by 5-10 big stores)
WEB_DEBUG = DEBUG_DIR / "web"

UNDERSTAND_PROMPT = """You are the salesperson of a price-comparison site for shoppers in Egypt: the kind of salesperson
people come back to because they really know the products, listen first and care that the shopper buys the right thing
for them. The difference: you stand in every online store at once.

How you talk: natural, warm Egyptian Arabic (English if the shopper writes English), like a person and not a form.
Use the shopper's own words, react to what they said, and say WHY in plain words (what they gain, what they give up).
Vary your wording; never sound like a template. Short is good, but never dry.

How you sell: first understand what this shopper needs (what for, what matters to them, budget), then show the real
choice they have: several good options across brands and price levels, not just one or two, with the trade-off of
each. Never push something over their budget; if their budget is tight for what they want, say so honestly and offer
the best you can within it plus one "if you can stretch a bit" option. Suggest current models that are sold in Egypt
now (the latest generations, not ones from years ago).

Shoppers often don't know the product's name or how stores list it: they describe what they need
("تيشرت راوند بيزك اسود", "موبايل مش غالي وبطاريته تكمل يومين"). Understand what they really want and turn it
into concrete specs and searches that find it in Egyptian online stores.

Choose ONE action for the newest message:
- "search": the shopper named a specific product/model, asked for prices/offers, picked one of your suggestions, or
  you already know enough to find good matches. Don't make a shopper who knows what they want answer questions.
- "ask": the request is too open to pick good products and one answer would change what you'd suggest a lot. Ask
  about what matters most for THIS product and this shopper, not always the price. Most shoppers don't know specs:
  ask about their life and use in plain words, and turn the answer into specs yourself. Never ask them to choose a
  spec, part or model number (not "RTX 3060 or 3070?", "16 or 32GB RAM?", "ANC?"); ask "what will you do with it?",
  "do you carry it around a lot?", "who is it for?" and offer tap answers in everyday words (e.g. 'أتعلم وأجرب
  مشاريع صغيرة', 'أدرّب موديلات تقيلة', 'أشغّل ChatGPT-like على جهازي'). If a price range is set below, never ask about
  budget. One short, friendly question with 2-4 tap answers. Don't repeat a question or ask the same thing in other
  words. If the shopper says they're not sure ("مش متأكد", "مش عارف", "اختارلي"), don't ask again: decide for them,
  say in one line what you chose and why, and recommend. After two questions in a row stop asking.
- "recommend": the shopper described a need or asked what to buy ("أنهي موبايل أحسن في حدود 8000؟", "عايز ايربودز
  حلوة"): suggest 3-6 specific products that fit, covering different brands and price levels inside the budget, each
  with a short "tier" label and one line on what it gives THIS shopper and what it lacks, from your own knowledge
  and without writing prices. We then find all of them in the stores right away with today's checked prices, so
  ALSO fill the search fields: need_based true, suggested_products = those products, and web_queries that find them.
- "answer": a QUESTION about results already shown or buying advice ("أول نتيجة دي أصلية؟", "الفرق بين الاتنين إيه؟",
  "أنهي أحسن؟", "عليها ضمان؟", "السعر ده كويس؟") that doesn't ask for different products: reply from the shown results
  (their stores, prices, discounts, trust, price history) and your knowledge, without a new search, and give your
  honest pick. To compare 2-4 products or offers, also fill "table". NEVER write a price, price range or estimate
  that isn't one of the shown results' prices: your memory of prices is old and the shopper must only see prices
  checked on the store page. Asking the price of something not shown, or where to buy it, is a "search". If a price looks too low for that product, say it
  may be a mistake, a different variant or a used/refurbished listing and how to check; if a store is unverified, say so.

Understand the need before you offer anything. Work out what this use really requires (e.g. local AI work needs a
strong NVIDIA GPU with plenty of VRAM and RAM; a gym earbud needs a secure fit and sweat resistance; a gift needs to
know who it's for), and what products that meet it cost in Egypt TODAY (new or imported, as the shopper says). Price
ranges you offer as tap answers and products you suggest must come from that: a range nobody can buy the needed
product in is useless. Prices in Egypt change fast and your memory of them is often old: when your answer depends on
today's prices (budget ranges, what a budget can buy, which models are in reach) and the "Market check" below is
missing, set market_query to ONE short web search, in the words Egyptian stores and shoppers use, that shows those
prices (e.g. 'سعر لابتوب RTX 4070 استيراد', 'اسعار سماعات بلوتوث عزل ضوضاء'); we run it and ask you again with the
results. When a Market check is given, base the prices on it and set market_query to null.

A budget the shopper gave anywhere in the chat (or the price range set below) stays for every later search and
recommendation until they change it: put it in max_price/min_price again each time.
Read the budget the way a shopper means it, and say which kind it is in budget_kind:
- "around": "في حدود 500", "حوالي 500", "ب 500", "around 500", "بـ 500 كده" = roughly that amount, not far below
  or far above: min_price about 70% of it and max_price about 115% (e.g. 350 and 575). Something for 70 is not
  "in the range of 500", and neither is something for 10,000.
- "under": "تحت 8000", "أقل من", "مش أكتر من", "max": max_price only (a gift or need "under X" should still be
  worth that money: suggest things near the top of the range, not the cheapest junk).
- "over": "فوق 3000", "أكتر من": min_price only. "between": "من 400 لـ 600": both.
- "cheaper": the shopper wants cheaper than the results on screen ("أرخص من كده", "فيه أرخص؟"): max_price just
  under the cheapest shown price, and widen the search to other models/brands of the same kind (keep the product
  type and the other requirements). If they asked for ONE exact model, there is nothing cheaper to find (the shown
  offers already are the cheapest checked ones): "answer" that honestly and suggest cheaper similar models instead.
Set budget_cleared true only when the shopper drops their budget ("مش مهم السعر", "أي سعر").

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
The search fields (need_based and everything after "request") are needed for "search" and "recommend": leave them
empty for "ask" and "answer".

{can_do}
{chat}
Reply JSON (in this order):
{
  "need_specs": ["what this shopper's use really requires, in short English, e.g. 'NVIDIA RTX GPU 8GB+ VRAM',
                 '32GB RAM'; [] when they named an exact product"],
  "market_query": "one short web search for today's prices in Egypt, or null (see above)",
  "reply": "what you say to the shopper now, as a salesperson: for ask the question itself; for recommend one short
            intro line; for answer 1-5 short sentences answering; for search one short line saying what you'll look
            for (mention the change if a follow-up). NEVER write a price, price range or 'starts from' amount here
            (not even from the Market check: those are other sites' snippets, not checked prices; the shopper sees
            only prices we check on the store page), and never say how many products you'll show or suggest;
            budget amounts belong only in the tap answers",
  "action": "search" or "ask" or "recommend" or "answer",
  "options": ["if ask: 2-4 short quick answers the shopper can tap, else []"],
  "recommendations": [{"name": "if recommend: a specific product, e.g. 'Samsung Galaxy A16'", "tier": "2-3 words,
     e.g. 'الأوفر' / 'أحسن قيمة' / 'الأعلى جودة' / 'لو تقدر تزود شوية'", "why": "one short Egyptian Arabic line:
     what it gives THIS shopper and what it lacks"}],
  "table": null or, for an answer that compares: {"columns": ["", "product 1", "product 2"],
           "rows": [["السعر", "...", "..."], ["الضمان", "...", "..."]]} (2-6 short rows),
  "suggestions": ["2-4 short Egyptian Arabic next messages the shopper may want to tap, written as the shopper
     would say them; ONLY things we can do (see "Suggestions" above); [] for ask"],
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
  "product": {"kind": "for search/recommend: what the shopper wants in plain English, e.g. 'washing machine',
              'smartphone', 'wireless earbuds'", "brand": "the brand they asked for, or null",
              "size": "the size/capacity/storage they asked for, e.g. '8 kg', '256GB', '55 inch', or null",
              "model": "a model number only when they want ONE exact product, else null"} or null,
  "queries": ["1-2 short English store search queries, most specific first"],
  "requirements": ["hard requirements a matching product must meet, in English, from the need too, e.g.
     'battery 5000 mAh or more', 'black', 'crew (round) neck', 'plain, no print'"],
  "pick_options": ["the options the shopper named that a store page makes them pick for this product (size, color,
     storage, capacity...), in short Egyptian Arabic, e.g. 'مقاس 46', 'لون أسود', '256 جيجا'; [] if none. Each offer's
     own page is checked for these: offered, in stock, and its price"],
  "avoid": ["things that must NOT be returned, e.g. 'cases/covers', 'bundles with other products', 'used items'"],
  "max_price": number or null,   // ONLY when the shopper clearly states a price/budget ("تحت 8000", "بحدود 5000 جنيه",
                                 // "budget 3000"). A bare number is a spec or model ("مكواه 650", "2200 وات"): put it
                                 // in requirements/queries, NOT here. Words like "رخيص" / "مش غالي" -> null.
  "min_price": number or null,
  "budget_kind": "around" | "under" | "over" | "between" | "cheaper" | null,
  "budget_cleared": true only if the shopper dropped their budget, else false
}"""
# What a tapped suggestion can really get done: every suggestion is a button that sends its text as the shopper's
# next message, so one we can't deliver costs the shopper time and us an LLM call for nothing.
CAN_DO = """Suggestions are buttons: tapping one sends its text as the shopper's next message. Offer ONLY what this site
can really do right away:
- change or narrow the search: another brand, size, color or spec, or a price range written with numbers
  ("تحت 3000", "من 400 لـ 600");
- other models of the same kind when the shopper wants a kind of product rather than one exact model: cheaper ones
  ("موديلات أرخص") only if cheaper models exist, a step up ("حاجة أحسن شوية"), other ideas for a need or a gift;
- questions about the offers on screen: compare them, which is the better value, warranty, is it original.
Never offer what it can't do: cheaper offers of the same exact model (the results already are the cheapest checked
offers we found), searching more stores (the page has its own button for that), sorting or filtering the list (the
page does that), price alerts, delivery times, installments, reviews or ratings, used items, or anything outside
these stores. Fewer good suggestions beat one that does nothing."""
UNDERSTAND_PROMPT = UNDERSTAND_PROMPT.replace("{can_do}", CAN_DO)
MAX_HISTORY = 8  # chat messages the planner sees
_PRICE_TAIL = re.compile(r"\s*[-–—:,(]*\s*(?:EGP|LE|ج\.?م|جنيه)?\s*\d[\d,.]*\s*(?:EGP|LE|ج\.?م|جنيه)\)?\s*$"
                         r"|\s*[-–—:,(]+\s*(?:EGP|LE)\s*\d[\d,.]*\)?\s*$", re.I)


_MONEY_NEAR = re.compile(r"(\d[\d,.]*)\s*(?:-|–|لحد|لـ|ل|إلى|الى|to)?\s*(?:\d[\d,.]*\s*)?(?:جنيه|ج\.?\s?م|egp|le\b|ألف|الف)"
                         r"|(?:egp|le|جنيه)\s*(\d[\d,.]*)", re.I)


def unchecked_prices(text: str, shown: list[dict]) -> bool:
    """The reply names a price that none of the shown results has: a guess from memory, never to be shown."""
    known = {round(float(r["price"])) for r in shown if isinstance(r.get("price"), (int, float))}
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,."))
    for m in _MONEY_NEAR.finditer(text):
        for num in re.findall(r"\d[\d,.]*", m.group(0)):
            try:
                n = float(num.replace(",", "").rstrip("."))
            except ValueError:
                continue
            if n >= 50 and not any(abs(n - k) <= 1 for k in known):
                return True
    return False


_NUMBER = re.compile(r"\d")
_THOUSANDS = re.compile(r"^\s*(?:ألف|الف|آلاف|الاف|k\b)", re.I)
_MONEY_WORD = re.compile(r"^\s*\+?\s*(?:ألف|الف|آلاف|الاف|k\b|جنيه|ج\.?\s?م|egp|le\b|pounds?)", re.I)
_MONEY_BEFORE = re.compile(r"(?:egp|le|جنيه|ج\.?\s?م)\s*$", re.I)
_RANGE_JOIN = re.compile(r"^\s*\+?\s*(?:-|–|لحد|لـ|ل|إلى|الى|و|to)\s*$", re.I)
_SENTENCE = re.compile(r"(?<=[.!؟?\n،:])\s+")


def _money_amounts(text: str) -> list[float]:
    """Money amounts a text names: a number with a currency or thousands word next to it ("9000 جنيه", "25 ألف",
    "EGP 4,999"), including both ends of a range ("من 9000 لـ 25000+ جنيه")."""
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,."))
    nums = list(re.finditer(r"\d[\d,.]*", text))
    found: list[float] = []
    for i, m in enumerate(nums):
        try:
            n = float(m.group(0).replace(",", "").rstrip("."))
        except ValueError:
            continue
        after = text[m.end():m.end() + 12]
        money = bool(_MONEY_WORD.match(after)) or bool(_MONEY_BEFORE.search(text[max(0, m.start() - 6):m.start()]))
        # the first end of a range takes the money word of the second ("من 9 لـ 25 ألف")
        nxt = nums[i + 1] if i + 1 < len(nums) else None
        if not money and nxt and _RANGE_JOIN.match(text[m.end():nxt.start()]):
            tail = text[nxt.end():nxt.end() + 12]
            money = bool(_MONEY_WORD.match(tail))
            if _THOUSANDS.match(tail) and n < 1000:
                n *= 1000
        if money:
            if _THOUSANDS.match(after) and n < 1000:
                n *= 1000
            found.append(n)
    return found


def without_unchecked_prices(text: str, shown: list[dict] | None = None, live: bool = False) -> str:
    """The LLM's own words (its reply, question, recommendation lines) with every sentence that names a price none
    of the shown results has taken out: its memory of prices and the web snippets it read aren't prices checked on
    a store page (e.g. "الغسالات دي بتبدأ من 9000 لحد 25000+ جنيه"). `live`: the text is still being written,
    so its unfinished last sentence waits while it has a number (its currency word may come next)."""
    if not text:
        return text
    known = {round(float(r["price"])) for r in shown or [] if isinstance(r.get("price"), (int, float))}
    parts = _SENTENCE.split(text)
    keep = []
    for i, part in enumerate(parts):
        if live and i == len(parts) - 1 and _NUMBER.search(part):
            break
        if any(n >= 50 and not any(abs(n - k) <= 1 for k in known) for n in _money_amounts(part)):
            continue
        keep.append(part)
    return " ".join(keep).strip()


_COUNT_WORDS = (r"\d+|واحد|اتنين|إتنين|اثنين|تلات[ةه]?|ثلاث[ةه]?|تلت|أربع[ةه]?|اربع[ةه]?|خمس[ةه]?|ست[ةه]?|سبع[ةه]?|"
                r"تمان[يي]?[ةه]?|ثماني[ةه]?|تسع[ةه]?|عشر[ةه]?|one|two|three|four|five|six|seven|eight|nine|ten")
_COUNTED = re.compile(r"(?<![\w])(?:" + _COUNT_WORDS + r")\s+(?=(?:من\s+)?(?:ال)?(?:موديلات|موديل|اختيارات|اختيار|خيارات|"
                      r"منتجات|عروض|ترشيحات|أنواع|انواع|options?|models?|products?|offers?|picks?)\b)", re.I)


def without_counts(text: str) -> str:
    """"رشحتلك 3 موديلات" -> "رشحتلك موديلات": the LLM writes how many products it suggests before the list is
    final, and the shopper then sees another number of them. The list itself shows how many."""
    return _COUNTED.sub("", text or "")


def shopper_text(text: str, shown: list[dict] | None = None, live: bool = False) -> str:
    """What the LLM says to the shopper, without prices nobody checked and without counts that may not match."""
    return without_counts(without_unchecked_prices(text, shown, live))


MAX_SHOWN = 10  # results of the last search the planner sees (to answer and compare)
MAX_RECS = 6    # products one recommendation shows


def _money(n) -> str:
    return f"{n:g}" if isinstance(n, (int, float)) else "?"


_DISCOUNT_WORDS = {"real": "REAL deal: lowest price of this page in 30 days",
                   "fake": "FAKE discount: the store's old price was never asked in 30 days, this is its usual price",
                   "higher": "was cheaper at this store in the last 30 days", "usual": "its usual price lately"}


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
        if r.get("real_discount") in _DISCOUNT_WORDS:  # judged on this page's own last 30 days (app/pricechart.py)
            bits.append(_DISCOUNT_WORDS[r["real_discount"]])
        if r.get("note"):
            bits.append(f"note: {str(r['note'])[:120]}")
        lines.append(" | ".join(bits))
    return ("Results shown to the shopper now (cheapest first):\n" + "\n".join(lines) + "\n") if lines else ""


def price_range_text(lo: float | None, hi: float | None) -> str:
    """The price range the shopper set in the search box, in words (also what makes it count as a stated budget)."""
    cur = market.current().currency_ar
    if lo and hi:
        return f"السعر من {lo:g} لحد {hi:g} {cur}"
    if hi:
        return f"السعر تحت {hi:g} {cur}"
    return f"السعر فوق {lo:g} {cur}" if lo else ""


def _chat_block(messages: list[dict], current: str, shown: list[dict] | None = None,
                price: tuple[float | None, float | None] = (None, None)) -> str:
    lines = [("Shopper: " if m.get("role") == "user" else "Assistant: ") + str(m.get("text", ""))[:800]
             for m in messages[-MAX_HISTORY:]]
    rng = (f"Price range the shopper set in the search box: {price_range_text(*price)} (min {price[0] or 'none'}, "
           f"max {price[1] or 'none'}). Respect it in every action and never ask about budget.\n") if any(price) else ""
    return ("Current request (from the previous search): " + (current or "none") + "\n" + rng + _shown_block(shown or [])
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


MARKET_HITS = 8           # search results the planner reads to learn today's prices
MARKET_TTL = 24 * 3600    # the same market check is reused for a day (prices don't move that fast)
_MARKET_ASKED = re.compile(r'"market_query"\s*:\s*"[^"]')  # the planner wants a market check first


async def _market_check(query: str, plan: dict) -> str:
    """One web search for what the needed products cost today (any shop: imported goods are often sold outside
    the big stores), kept for a day so popular needs cost nothing. '' when it can't be done."""
    key = "__market__:" + db.cache_key(query, market.current().code)
    hit = await asyncio.to_thread(_db_safe, db.cached_search, key, MARKET_TTL)
    if hit is not None:
        return hit.get("text", "")
    if not websearch.available():
        return ""
    _progress("بشوف أسعار السوق النهارده عشان أرشحلك صح...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await websearch.search(client, query, max_results=MARKET_HITS)
    except (websearch.SearchError, httpx.HTTPError) as e:
        log.warning("agent: market check failed: %s", e)
        return ""
    text = "\n".join(f"- {str(h.get('title', ''))[:120]} | {str(h.get('snippet', ''))[:240]} | {trust.domain(h.get('url', ''))}"
                     for h in res)
    if text:
        await asyncio.to_thread(_db_safe, db.save_search, key, market.current().code, {"text": text})
    return text


HITS_BLOCK = """Web search results for the shopper's own words "{words}" (title | snippet):
{hits}
For "search" and "recommend", use these results to check what the shopper really wants before we spend searches on
it: the exact product, its product type, the variant they mean, and its official model numbers. Shoppers write
slang, misspellings and mixed Arabic/English, and the same word can mean different things for different products:
read every word in the context of this product, the way stores selling it use it. If the results clearly show the
product, trust them over your first reading; if they are unrelated or unclear, ignore them. Never change the product
type the shopper named. Model numbers only pin down ONE exact product the shopper named; when they want a kind of
product (any model that meets the specs), leave model_codes empty and keep the searches broad enough to find every
fitting model. Take the stores selling it in these results into account for "stores".
"""
HITS_TTL = 1800  # the same shopper words are searched once per half hour (a chat's next turns reuse it)
_HITS: dict[str, tuple[float, list[dict]]] = {}


async def shopper_hits(words: str) -> list[dict]:
    """One web search of the shopper's own words, given to the planner so it plans the right product in the same
    LLM call (this used to be a second "check" call after planning). [] when it can't be done."""
    words = " ".join(words.split())[:200]
    if len(words) < 2 or not websearch.available():
        return []
    key = words.lower()
    if (hit := _HITS.get(key)) and time.time() - hit[0] < HITS_TTL:
        return hit[1]
    try:
        async with httpx.AsyncClient() as client:
            hits = await websearch.search(client, words, max_results=CHECK_HITS)
    except (websearch.SearchError, httpx.HTTPError) as e:
        log.info("agent: shopper words search failed: %s", e)
        return []
    if len(_HITS) > 500:
        _HITS.clear()
    _HITS[key] = (time.time(), hits)
    return hits


def _hits_text(hits: list[dict]) -> str:
    return "\n".join(f"- {str(h.get('title', ''))[:120]} | {str(h.get('snippet', ''))[:200]}" for h in hits)


async def plan_turn(messages: list[dict], current: str = "", shown: list[dict] | None = None,
                    price: tuple[float | None, float | None] = (None, None), hits: list[dict] | None = None,
                    hits_for: str = "") -> dict:
    """Understand the newest message in the context of the chat (and the results on screen), decide what a good
    salesperson does now (search / ask / recommend / answer), and plan the searches. Usually one LLM call; when the
    answer depends on today's prices it doesn't know (market_query), one web search and a second call that sees them.
    `hits`: web search results for the shopper's words (`hits_for`): the plan is then already checked against them
    and run() doesn't spend a second LLM call on that.
    Raises llm.LLMError when no model answers."""
    live = REPLY.get()

    def on_text(text: str) -> None:  # the reply is shown while it's written, unless a market check comes first
        if live and not _MARKET_ASKED.search(text) and (t := shopper_text(reply_so_far(text), shown, live=True)):
            live(t)
    block = _chat_block(messages, current, shown, price)
    if hits:
        block += HITS_BLOCK.format(words=hits_for[:200], hits=_hits_text(hits))
    plan = await llm.ask_json(UNDERSTAND_PROMPT.replace("{chat}", block), timeout=12, prefer="strong", step="plan",
                              on_text=on_text if live else None,
                              temperature=0.5)  # a little variety: a salesperson, not a template
    query = plan.get("market_query")
    if _named_product_search(plan):
        # a search for a product the shopper named: today's prices come from the search itself, checked on each
        # store's page, and the reply line never shows a price, so a second planner call with the market's
        # prices would change nothing the shopper sees (live 2026-11-26: "ريلمي 12 برو بلس 256" asked for one)
        query = None
    if isinstance(query, str) and query.strip() and (found := await _market_check(query.strip()[:150], plan)):
        log.info("agent: market check %r before answering", query)
        block += f"Market check (web search '{query.strip()[:150]}', today's prices in Egypt):\n{found}\n"
        plan = await llm.ask_json(UNDERSTAND_PROMPT.replace("{chat}", block), timeout=12, prefer="strong",
                                  step="plan_market", on_text=on_text if live else None, temperature=0.5)
        plan["market_checked"] = query.strip()[:150]
    last = next((m.get("text", "") for m in reversed(messages) if m.get("role") == "user"), "")
    # names only: a price the LLM guessed would mislead the shopper and the judge
    plan["suggested_products"] = [_PRICE_TAIL.sub("", x).strip() for x in plan.get("suggested_products") or []
                                  if isinstance(x, str) and x.strip()]
    reply = plan.get("reply") if isinstance(plan.get("reply"), str) else ""
    reply = reply.strip() or str(plan.get("question") or plan.get("answer") or "").strip()  # older field names
    if plan.get("action") != "answer":  # (an answer naming unchecked prices becomes a search, in main)
        reply = shopper_text(reply, shown)
    plan["reply"] = without_counts(reply)
    reply = plan["reply"]
    plan["suggestions"] = _strs(plan.get("suggestions"), 4, 60)
    if not isinstance(plan.get("request"), str) or not plan["request"].strip():
        plan["request"] = last
    recs = [{"name": str(r["name"]).strip()[:120], "why": shopper_text(str(r.get("why") or "").strip()[:200], shown),
             "tier": str(r.get("tier") or "").strip()[:40]}
            for r in plan.get("recommendations") or [] if isinstance(r, dict) and str(r.get("name") or "").strip()]
    action = plan.get("action")
    if action == "answer" and reply:
        plan["answer"] = reply
        plan["table"] = _table(plan.get("table"))
        plan["request"] = current or plan["request"]  # a question about the results keeps the current request
    elif action == "recommend" and recs:
        plan["recommendations"] = [{**r, "name": _PRICE_TAIL.sub("", r["name"]).strip()} for r in recs[:MAX_RECS]]
        plan["request"] = current or plan["request"]
    elif action == "ask" and reply:
        plan["question"] = reply
        plan["options"] = _strs(plan.get("options"), 4, 60)
    else:
        plan["action"] = "search"
        plan["understood"] = shopper_text(str(plan.get("understood") or reply).strip(), shown)
    if hits and plan["action"] in ("search", "recommend"):
        plan["checked"], plan["check_hits"] = True, hits
    said = " ".join(m.get("text", "") for m in messages if m.get("role") == "user")
    plan["pick_options"] = [o for o in _strs(plan.get("pick_options"), 4, 60) if shopper_named(o, said)]
    log.info("agent: plan %s -> %r", plan["action"], reply if plan["action"] != "search" else plan["request"])
    return plan


def _named_product_search(plan: dict) -> bool:
    """The planner's first reading is a search for a product the shopper named (not a need to fill, a question to
    ask or products to recommend): it needs no market check before the search."""
    product = plan.get("product") if isinstance(plan.get("product"), dict) else {}
    return (plan.get("action") == "search" and plan.get("need_based") is not True
            and not plan.get("suggested_products") and bool(product.get("brand") or product.get("model")
                                                            or plan.get("model_codes")))


def _db_safe(fn, *args):
    """A database read/write for a nice-to-have: if the database fails, carry on without it."""
    try:
        return fn(*args)
    except Exception as e:
        log.warning("agent: db %s failed: %s", getattr(fn, "__name__", "?"), type(e).__name__)
        return None


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
        hits=_hits_text(hits))
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
Budget (EGP): min {minp}, max {maxp} (every candidate is inside it already; for a need or a gift, things worth
about the budget fit better than the cheapest trinkets)
{known}Candidates (id, store, store trust 0-100 with the signals found on its site, condition if not new,
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
say in the note what to check. Set "sure": true when the title/model/link confirms the exact variant: the brand,
type and size/capacity/spec the shopper asked for all show in it (a model number you don't know is NOT a mismatch:
judge by what the title says). A candidate you keep fits: its note never says the model/variant doesn't match (if
it doesn't, drop it); with "sure": false the note only names the detail to check on its page. And
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
  "suggestions": ["0-4 short Egyptian Arabic next messages the shopper may want to tap after seeing these results,
                   written as the shopper would say them, e.g. 'إيه الفرق بين أول اتنين؟', 'فيه لون تاني؟', 'سامسونج
                   بس'. Only what we can do (below)"]}}
{can_do}"""


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


# First results while the search goes on (the big stores' offers from our own price list, already checked on their
# pages): main.py sets a callback that sends them to the page.
PARTIAL: contextvars.ContextVar = contextvars.ContextVar("partial", default=None)


def _partial(data: dict) -> None:
    if (show := PARTIAL.get()) is not None:
        show(data)


_TIMES: contextvars.ContextVar = contextvars.ContextVar("times", default=None)


class _Times:
    """Seconds per stage of this search (log, /admin and the live check). One dict per shopper's request: searches
    running at the same time used to share one, and one search clearing it made another fail with KeyError 'search'
    (live 2026-11-26, a 500 for "ريلمي 12 برو بلس"). new_search() starts a request's own; tasks it starts share it."""

    def _d(self) -> dict:
        d = _TIMES.get()
        if d is None:
            d = {}
            _TIMES.set(d)
        return d

    def __getitem__(self, k):
        return self._d().get(k, 0.0)

    def __setitem__(self, k, v):
        self._d()[k] = v

    def __iter__(self):
        return iter(self._d())

    def __len__(self):
        return len(self._d())

    def get(self, k, default=None):
        return self._d().get(k, default)

    def keys(self):
        return self._d().keys()

    def items(self):
        return self._d().items()

    def clear(self):
        self._d().clear()

    def update(self, *a, **kw):
        self._d().update(*a, **kw)


TIMES = _Times()


def new_search() -> None:
    """A shopper's request starts: its own stage times, not another request's."""
    _TIMES.set({})
    _LLM_PAGES.set([])
    _PAGE_READS.set([0, 0])


# the store pages the LLM read for this search ("domain listing" / "domain product"), shown by the live check, so
# the pages that still cost an LLM call can be found and read without one
_LLM_PAGES: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_pages", default=None)


def llm_pages() -> list[str]:
    return list(_LLM_PAGES.get() or [])


# store pages read for this search: [read with no LLM call, read by the LLM]
_PAGE_READS: contextvars.ContextVar[list | None] = contextvars.ContextVar("page_reads", default=None)


def _count_read(by_llm: bool, label: str = "") -> None:
    if (reads := _PAGE_READS.get()) is not None:
        reads[1 if by_llm else 0] += 1
    if label and (pages := _LLM_PAGES.get()) is not None and len(pages) < 40:
        pages.append(label)


def page_reads() -> dict:
    """How this search's store pages were read: {"without_llm": n, "llm": m}."""
    reads = _PAGE_READS.get() or [0, 0]
    return {"without_llm": reads[0], "llm": reads[1]}


PAGE_CHARS = 3000  # page text sent to the LLM: the part with the products (most of a page is menus and footers)
_PRICE_NUM = re.compile(r"\d[\d,.]{2,}")


def _page_text(html: str, focus: list[str], limit: int = PAGE_CHARS, base_url: str = "", top: bool = False) -> str:
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
    # top: from where the product is first named (a product page's own title and buy box come before the
    # related items, which can name the same words more often)
    start = max(0, marks[0] - 300) if top else max((max(0, i - 300) for i in marks[::max(1, len(marks) // 40)]), key=score)
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


FOLLOWED_PAGE_PROMPT = """This is the text of one product page of an online store. A shopper wants the current price
of this product: {title}
The option the shopper wants of it: {wanted}
Read the price of THE product this page is for: the one in its title and main buy box at the top. Prices further down
belong to other items (related, sponsored, "customers also bought", bundles, accessories, other sellers' offers,
installments): never use them.
Reply JSON: {{"same_product": true if the page's main product is the followed product (same model; another size or
color is fine), "currency": "currency of the price, e.g. EGP",
"price": number or null (the current price of the main product exactly as written, not the crossed-out old price),
"choice_needed": true only if the page shows a price range ("1,500 - 2,400") or only per-option prices and no single
price for the product as it opens. Sizes or colors to pick don't matter when the page shows one current price (most
pages do): then false. If the page lists a price per size and the product names a size, give that size's price.
"in_stock": true | false | null,
"option": null when no option is wanted above, else {{"offered": true if the page lets you pick that option (or the
  product only comes in it), false if the page lists its choices (sizes, colors...) and it isn't one of them, null if
  the page doesn't show the choices; "available": true if that option can be bought now, false if the page marks it
  unavailable / sold out / greyed out ("(unavailable)" in the text), the VARIANTS data says out of stock, or only
  other options are in stock; null when nothing on the page says this option itself is in stock (a size menu that
  lists every size proves nothing, and neither does "in stock" for the product as a whole when it has options);
  "price": that option's own current price if the page writes one, else null}}}}
VARIANTS (when given) is the store's own data on each option's stock: trust it over the menus.
price is null when the page writes no current price for its main product. Never guess.
PAGE:
{text}"""


NEAR_TITLE_CHARS = 1200  # where a page names its product: its price is shown there too, before related items
MAIN_PRICE_CHARS = 6000  # page text for a main-price read: the top of the page, where the product and its price are


_CHOICE_OFF = re.compile(r"disabled|unavailable|out[-_ ]?of[-_ ]?stock|sold[-_ ]?out|\boos\b|not[-_ ]?available", re.I)
_CHOICE_TAG = re.compile(r"<(option|button|li|a|span|div|label)\b([^>]*)>([^<>]{1,40})</\1>", re.I)


def _mark_choices(html: str) -> tuple[str, str]:
    """A page's choices (sizes, colors...) keep their state in tags and classes that page text loses: mark the ones
    the page greys out as "(unavailable)", and list all of them, so the LLM can tell which options can be bought.
    Returns (the page with marks, "choices: ..." or "")."""
    seen: list[str] = []

    def mark(m: re.Match) -> str:
        tag, attrs, text = m.group(1).lower(), m.group(2), " ".join(unescape(m.group(3)).split())
        off = bool(_CHOICE_OFF.search(attrs))
        if not text or not (tag == "option" or off):
            return m.group(0)
        item = text + (" (unavailable)" if off else "")
        if item not in seen and len(seen) < 80:
            seen.append(item)
        return m.group(0) if not off else f"<{tag}{attrs}>{m.group(3)} (unavailable)</{tag}>"
    marked = _CHOICE_TAG.sub(mark, html)
    return marked, ("CHOICES ON THE PAGE: " + " | ".join(seen))[:1500] if seen else ""


async def _read_main_price(url: str, html: str, title: str, spread: int = 0, wanted: str = "") -> Product | None:
    """A product page whose own product data doesn't settle the price (e.g. Amazon): the LLM reads only the main
    product's price. None when it isn't the followed product, the price depends on a choice (size, color) the page leaves
    open, or the page doesn't show that number: a price from a related item lower down must never become an alert.
    `wanted`: the option the shopper wants (e.g. "مقاس 46"). The page is read for it too: `option_ok` is False when
    the page doesn't offer it or marks it unavailable (price 0 then: never shown), True when it says it's in stock,
    None when the page doesn't say; the price is that option's price when the page writes one."""
    words = [w for w in re.split(r"\W+", title) if len(w) > 2]
    marked, choices = _mark_choices(html) if wanted else (html, "")
    text = _page_text(marked, words, MAIN_PRICE_CHARS, base_url=url, top=True)
    if choices:
        text += "\n" + choices
    variants = extract_mod.variant_stock(html)
    if variants and all(not ok for _, ok, _ in variants):  # every option sold out: the product is
        return Product(store=trust.store_name(url), title=title or url, price=0.0, url=url, trusted=trust.is_known(url),
                       in_stock=False, option=wanted or None, option_ok=False if wanted else None)
    if (not wanted or _size_option_shown(wanted, title, html)) and (got := _by_buybox(url, html, title)):
        _count_read(False)
        return got if not wanted else got.model_copy(update={"option": wanted, "option_ok": got.in_stock})
    if not wanted and (by_recipe := await asyncio.to_thread(recipes.read, url, html)):
        price, stock = by_recipe  # the store's learned recipe read this page: no LLM call
        if shows(_shown_numbers(html), price):
            _count_read(False)
            return Product(store=trust.store_name(url), title=title or url, price=float(price), url=url,
                           trusted=trust.is_known(url), in_stock=stock)
    if variants and wanted and (got := _by_variant(url, html, title, wanted, variants)):
        _count_read(False)
        return got
    if variants and wanted:
        text += "\nVARIANTS: " + "; ".join(
            f"{label}: {'in stock' if ok else 'OUT OF STOCK'}" + (f" ({price:g})" if price else "")
            for label, ok, price in variants)[:1500]
    _count_read(True, f"{trust.domain(url)} check" + ("" if "<" in html[:2000] else " (text copy)") + (" (option)" if wanted else ""))
    data = await llm.ask_json(FOLLOWED_PAGE_PROMPT.format(title=title or url, wanted=wanted or "none", text=text),
                              prefer="bulk", spread=spread, step="read_price")
    if data.get("same_product") is not True:
        return None
    stock = data.get("in_stock")
    stock = stock if isinstance(stock, bool) else None
    option_ok, price, choice_needed = None, data.get("price"), data.get("choice_needed") is True
    if wanted:
        opt = data.get("option") if isinstance(data.get("option"), dict) else {}
        if opt.get("offered") is False or opt.get("available") is False:
            return Product(store=trust.store_name(url), title=title or url, price=0.0, url=url,
                           trusted=trust.is_known(url), in_stock=stock, option=wanted, option_ok=False)
        option_ok = True if opt.get("offered") is True and opt.get("available") is True else None
        own = opt.get("price")
        if isinstance(own, (int, float)) and not isinstance(own, bool) and own >= 5 and shows(_shown_numbers(html), own):
            price, choice_needed = own, False  # the price of the shopper's own option
    cur = str(data.get("currency") or "").strip().upper()
    mk = market.current()
    if (choice_needed or not isinstance(price, (int, float)) or isinstance(price, bool) or price < 5
            or cur and cur not in {c.upper() for c in mk.currency_aliases + (mk.currency,)} | {"ج.م", "جم", "جنيه", "L.E", "L.E."}
            or not shows(_shown_numbers(html), price)):
        return None
    if not wanted:  # the store's recipe is checked against this read, or learned from it (after the answer)
        _later(recipes.after_llm_read(url, html, title or url, float(price)))
    return Product(store=trust.store_name(url), title=title or url, price=float(price), url=url,
                   trusted=trust.is_known(url), in_stock=stock, option=wanted or None, option_ok=option_ok)


_OPTION_WORDS = {"مقاس", "المقاس", "لون", "اللون", "سعة", "size", "color", "colour", "capacity"}


def _by_variant(url: str, html: str, title: str, wanted: str, variants: list) -> Product | None:
    """The shopper's option read from the page's own options data (extract.variant_stock), no LLM call: exactly one
    option is named with every word of it ("46", "XL", "Black"); its stock, and its own price when the page shows it
    (else the product's one price when no option has a price of its own)."""
    words = {w for w in re.split(r"[\s،,/|:-]+", normalize(wanted)) if w and w not in _OPTION_WORDS}
    if not words:
        return None
    hits = [(label, ok, pr) for label, ok, pr in variants
            if words <= set(re.split(r"[\s،,/|:()-]+", normalize(label)))]
    if len(hits) != 1:
        return None
    _, ok, own = hits[0]
    store = trust.store_name(url)
    if not ok:
        return Product(store=store, title=title or url, price=0.0, url=url, trusted=trust.is_known(url),
                       in_stock=False, option=wanted, option_ok=False)
    price = own if own and shows(_shown_numbers(html), own) else None
    if price is None and not any(pr for _, _, pr in variants) and (box := _by_buybox(url, html, title)):
        price = box.price
    if price is None:
        return None
    return Product(store=store, title=title or url, price=float(price), url=url, trusted=trust.is_known(url),
                   in_stock=True, option=wanted, option_ok=True)


def _by_buybox(url: str, html: str, title: str) -> Product | None:
    """The followed product's page read by its buy box (app/buybox.py), no LLM call: only when the page names this
    product, shows one clear price and says whether it can be bought."""
    if "<" not in html[:2000] or not (got := buybox.read(html, url)) or got[2] is None:
        return None
    name, price, stock = got
    words = {normalize(w) for w in re.split(r"\W+", title) if len(w) > 2}
    named = {normalize(w) for w in re.split(r"\W+", name) if len(w) > 2}
    if not words or len(words & named) < max(2, 0.6 * min(len(words), len(named))):
        return None  # the page's name isn't the followed product's
    if not shows(_shown_numbers(html), price):
        return None
    return Product(store=trust.store_name(url), title=title or name, price=float(price), url=url,
                   trusted=trust.is_known(url), in_stock=stock)


_BACKGROUND: set[asyncio.Task] = set()


def _later(coro) -> None:
    """Work nobody waits for (learning a store's page recipe), kept referenced until it ends."""
    task = asyncio.create_task(coro)
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)


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
        return "robots.txt disallows" + (f" ({self.args[0]})" if self.args else "")


async def _fetch_page(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> tuple[list[Product], str]:
    # the store's crawler rules apply to our own requests too, not only to the reader service (cached per site)
    if not await asyncio.wait_for(fetchers.robots_allows(client, url), timeout=PAGE_TIMEOUT):
        raise RobotsDisallowed(fetchers.robots_reason(url))
    async with sem:
        r = await client.get(url, headers=HEADERS, follow_redirects=True, timeout=PAGE_TIMEOUT)
        if r.status_code == 403:
            # some stores (Jumia) refuse browser-looking requests but welcome a crawler that says who it is
            # (their robots.txt asks for that): ask once more under our own crawler name
            r = await client.get(url, headers=catalog.BOT_HEADERS, follow_redirects=True, timeout=PAGE_TIMEOUT)
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


def _listed_products(html: str, url: str, words: str) -> list[Product]:
    """The products of a listing page read from its product cards (app/cards.py, or Amazon's own card layout), no
    LLM call: the ones whose names share a word with the request, best first. [] when the page isn't clearly a
    list of priced products, or none of its products is about the request: the LLM reads it then, as before."""
    store = trust.store_name(url)
    products = cards_mod.listing_cards(html, store, url) or parse_amazon_cards(html, store, url)
    if not products:
        return []
    wanted = {_stem(w) for w in re.findall(r"\w+", _fold(words))
              if w not in _SEARCH_NOISE and w not in FILLER and (len(w) >= 3 or w.isdigit())}
    scored = []
    for p in products:
        title = _fold(p.title)
        hits = sum(1 for w in wanted if (_has_word(title, w) if w.isdigit() else w in title))
        if hits:
            scored.append((hits, p))
    if not scored:
        log.info("web: %s lists %d products, none about the request: the LLM reads it", trust.domain(url),
                 len(products))
        return []
    scored.sort(key=lambda x: -x[0])
    picked = _cheapest_variant([p for _, p in scored])[:MAX_PER_PAGE]
    score, signals = trust.page_signals(html, url, True)
    for p in picked:
        p.store = trust.store_name(p.url or url)
        p.trusted = trust.is_known(p.url or url)
        p.trust_score, p.trust_signals = (100, ["متجر مشهور"]) if p.trusted else (score, signals)
    log.info("web: %s read from its product cards (no LLM): %d of %d products", trust.domain(url), len(picked),
             len(products))
    return picked


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


JUDGE_BUDGET = 20  # seconds for the judge across models: when the free ones are busy, the cheapest fitting offers instead
STORE_RETRY_WAIT = 1.5  # seconds before asking a busy store's search page once more
STORE_SEARCH_DEADLINE = 12  # the big stores' own search pages, read with no LLM call
STORE_SEARCHES = {"EG": ["https://www.amazon.eg/s?k={q}&language=en_AE", "https://btech.com/en/s?q={q}"],
                  "SA": ["https://www.amazon.sa/s?k={q}&language=en_AE"],
                  "AE": ["https://www.amazon.ae/s?k={q}&language=en_AE"]}


async def _store_search(queries: list[str]) -> tuple[list[Product], dict[str, str]]:
    """The big stores' own search pages for the request's English words, read with no LLM call (each store's
    product data or its product cards): every product they sell, not only those on the category pages of our price
    list. robots.txt is read first, as for every page we open. Their prices are checked on each product's own page
    before they're shown, like any other offer."""
    qs = [q for q in queries if re.search(r"[A-Za-z]", q)][:2]
    urls = [t.format(q=quote_plus(q)) for q in qs for t in STORE_SEARCHES.get(market.current().code, [])]
    if not urls:
        return [], {}
    t = time.monotonic()
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(4)

        async def fetch(u: str):
            try:
                return await _fetch_page(client, sem, u)
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in (429, 503):
                    raise
                # a big store answers "busy" now and then (Amazon, live 2026-11-25): once more, a moment later
                await asyncio.sleep(STORE_RETRY_WAIT)
                return await _fetch_page(client, sem, u)
        pages = await _within(STORE_SEARCH_DEADLINE, [fetch(u) for u in urls])
    out, errors, seen = [], {}, set()
    for url, res in zip(urls, pages):
        if isinstance(res, Exception):
            code = getattr(getattr(res, "response", None), "status_code", None)
            errors[f"{trust.domain(url)} search"] = f"{code}" if code else type(res).__name__
            continue
        products, html = res
        if not products:
            products = parse_amazon_cards(html, trust.store_name(url), url)[:MAX_PER_PAGE]
        for p in products:
            if p.url and p.url not in seen:
                seen.add(p.url)
                p.store, p.trusted, p.trust_score, p.trust_signals = (trust.store_name(p.url), True, 100,
                                                                      ["متجر مشهور"])
                out.append(p)
    TIMES["store_search"] = round(time.monotonic() - t, 1)
    log.info("stores: %d products from %d search pages in %.1fs, errors=%s", len(out), len(urls),
             time.monotonic() - t, errors)
    return out, errors


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
        hit_titles: dict[str, str] = {}  # the search service's title of each page (a product page: its product)
        jobs, found = [("check", None)] + jobs, [seed or []] + list(found)  # the plan-check search's hits count too
        for (group, _), res in zip(jobs, found):
            if isinstance(res, Exception):
                errors[f"web search ({group})"] = str(res)[:200]
                continue
            for hit in res:
                u = hit["url"]
                if len(hit.get("content") or "") >= MIN_CRAWLED_TEXT:
                    crawled.setdefault(u, hit["content"])
                if hit.get("title"):
                    hit_titles.setdefault(u, str(hit["title"]))
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
            if not page_products and _page_id(url) and (main := extract_mod.main_product(html)):
                # a clear product page (one product named, one price right by its name): no LLM read. Its price is
                # still checked on the page before it's shown, like any other
                page_products = [Product(store=trust.store_name(url), title=main[0], price=main[1], url=url,
                                         image=extract_mod.page_image(html, url), trusted=trust.is_known(url),
                                         in_stock=False if extract_mod._page_says_sold_out(
                                             html, extract_mod._visible_text(html), main[0]) else None)]
            if not page_products and _page_id(url) and (got := recipes.read(url, html)) \
                    and (name := extract_mod.page_title(html)):
                # a product page of a store whose layout the LLM already learned (app/recipes.py): its recipe reads
                # the main product's price, no LLM call (the price is checked again before it's shown)
                page_products = [Product(store=trust.store_name(url), title=name, price=got[0], url=url,
                                         image=extract_mod.page_image(html, url), trusted=trust.is_known(url),
                                         in_stock=got[1])]
            if not page_products and _page_id(url) and (box := buybox.read(html, url)):
                # a product page whose buy box (its <h1> and the one price beside it) is clear: no LLM read. The
                # price is checked again on the live page before it's shown
                page_products = [Product(store=trust.store_name(url), title=box[0], price=box[1], url=url,
                                         image=extract_mod.page_image(html, url), trusted=trust.is_known(url),
                                         in_stock=box[2])]
            if not page_products and (listed := _listed_products(html, url, f"{want} {focus}")):
                # a listing page whose product cards show their own name, link and price: read with no LLM call
                # (each offer's price is still checked on its own page before it's shown)
                page_products = listed
            if page_products:
                _count_read(False)
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
            _count_read(True, f"{trust.domain(u)} {'product' if _page_id(u) else 'listing'}"
                        + ("" if "<" in h[:2000] else " (text copy)"))
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
        # a product page's text copy that names its product and shows one clear price right after the name: read
        # with no LLM call (live 2026-11-26: nearly every page the LLM still read was such a copy, of noon, Jumia,
        # 2B, Raya...). Its price is checked on the live page before it's shown, like any other
        unclear = []
        for u, text in later:
            got = extract_mod.main_product_text(text, hit_titles.get(u, "")) if _page_id(u) else None
            if not got:
                unclear.append((u, text))
                continue
            _count_read(False)
            p = Product(store=trust.store_name(u), title=got[0], price=got[1], url=u, trusted=trust.is_known(u),
                        in_stock=got[2])
            p.trust_score, p.trust_signals = (100, ["متجر مشهور"]) if p.trusted else trust.page_signals(text, u, False)
            if (p.url, p.title) not in seen:
                seen.add((p.url, p.title))
                products.append(p)
        if len(unclear) < len(later):
            log.info("web: %d text copies of product pages read with no LLM call", len(later) - len(unclear))
        later = worth_reading(unclear, max(0, MAX_LLM_READS - len(first)))
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
    r"حوالي|حوالى|تقريبا|تقريبًا|(?:^|\s)بـ?\s?\d|\d\s*(?:ألف|الف|آلاف|الاف)|"
    r"\begp\b|\ble\b|budget|under|below|less than|max|over|above|between|around|about|\$|£",
    re.I)


def _states_budget(raw: str) -> bool:
    """A price limit counts only if the shopper used price words; '650' alone is usually a model or spec."""
    return bool(BUDGET_WORDS.search(raw))


AROUND_LOW, AROUND_HIGH = 0.7, 1.15  # "في حدود 500": what a shopper still calls "around" that amount


def _amount(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def settle_budget(plan: dict, said: str, shown_prices: list[float] | None = None,
                  earlier: tuple[float | None, float | None] = (None, None)) -> tuple[float | None, float | None]:
    """The price range a search must keep to, from the planner's reading of the budget (its kind: around / under /
    over / between / cheaper). The LLM understands the words; this applies them the same way every time:
    - kept only when the shopper really stated a budget (price words), wants cheaper than the offers on screen, or
      it is the budget from earlier in the chat (a bare "650" is usually a model or a spec, not a budget);
    - "around X" becomes a band near X, so neither a 70 nor a 10,000 item counts as "in the range of 500";
    - "cheaper" stays under the cheapest offer on screen; a budget from earlier carries on until the shopper drops it.
    Sets plan max_price / min_price / budget_ok and returns (min, max)."""
    lo, hi = _amount(plan.get("min_price")), _amount(plan.get("max_price"))
    kind = str(plan.get("budget_kind") or "").lower().strip()
    earlier = (_amount(earlier[0]), _amount(earlier[1]))
    cheapest = min((p for p in (shown_prices or []) if _amount(p)), default=None)
    if kind == "cheaper" and cheapest:
        hi = hi if hi is not None and hi < cheapest else max(cheapest - 1, 1)
        lo = None
        ok = True
    elif lo is None and hi is None:
        ok = False
    else:
        ok = _states_budget(said) or (lo, hi) == earlier
    if not ok:
        if lo is not None or hi is not None:
            log.info("agent: ignoring budget %s-%s the LLM guessed (no price words in the request)", lo, hi)
        lo, hi = (None, None) if plan.get("budget_cleared") else earlier
        kind = ""
    if kind == "around" and (lo is None or hi is None or lo == hi) and (lo or hi):
        x = hi or lo
        lo, hi = round(x * AROUND_LOW), round(x * AROUND_HIGH)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    plan["min_price"], plan["max_price"] = lo, hi
    plan["budget_ok"] = lo is not None or hi is not None
    return lo, hi


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


def _ordered(picks: list[Product], options: str = "") -> list[Product]:
    """Show order; when the shopper named a size/color..., offers whose page says it's in stock come first."""
    if not options:
        return _show_order(picks)
    return _show_order([p for p in picks if p.option_ok]) + _show_order([p for p in picks if not p.option_ok])


def _discount(p: Product) -> int | None:
    """Real discount in %: against the store's own old price, or the checkout coupon."""
    ref = p.old_price if p.old_price and p.old_price > p.price else p.price
    pct = round((ref - _paid(p)) / ref * 100) if ref else 0
    return pct if 3 <= pct <= 90 else None


VERDICT_TTL = 3 * 86400  # a judge verdict on one product page is reused for the same request for 3 days
VERDICT_PRICE_MOVE = 0.15  # ... unless its price moved more than this since (a big move can mean another item)
_VERDICTS: dict[str, dict] = {}  # verdict key -> {page id: verdict}, also kept in the database (see _verdicts)


VERDICTS_VERSION = "__verdicts2__:"  # a new judge prompt: verdicts the older one gave aren't reused


def verdict_key(plan: dict, raw: str, options: str = "") -> str:
    """What a verdict is about: the request as planned and the options the shopper must have. The same request
    (a popular product, a tapped idea, the second round of one search) reuses the verdicts already given."""
    return VERDICTS_VERSION + db.cache_key(f"{plan.get('request') or raw}|{options}", market.current().code)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _names_code(p: Product, codes: list[str]) -> bool:
    """The listing (title or link) carries one of the requested product's model numbers ("FTM-8S" = "ftm8s")."""
    have = f"{p.title} {unquote(p.url)}".lower()
    for code in (_squash(x) for x in codes):
        if len(code) < 4 or not (re.search(r"\d", code) and re.search(r"[a-z]", code)):
            continue  # "a55" or "2024" alone names many products
        # the same letters and digits, with or without - / space between them, and not the start of a longer code
        if re.search(r"(?<![a-z0-9])" + r"[\s_-]?".join(map(re.escape, code)) + r"(?![a-z0-9])", have):
            return True
    return False


def _vid(p: Product) -> str:
    return _page_id(p.url) or p.url


def _verdicts(key: str) -> dict:
    if key not in _VERDICTS:
        saved = _db_safe(db.cached_search, key, VERDICT_TTL) or {}
        if len(_VERDICTS) > 300:
            _VERDICTS.clear()
        _VERDICTS[key] = {k: v for k, v in saved.items() if isinstance(v, dict)}
    now = time.time()
    return {k: v for k, v in _VERDICTS[key].items() if now - v.get("at", 0) < VERDICT_TTL}


def _save_verdicts(key: str, new: dict) -> None:
    if not new:
        return
    have = _VERDICTS.setdefault(key, {})
    have.update(new)
    _db_safe(db.save_search, key, market.current().code, have)


def _known_verdict(v: dict | None, p: Product) -> dict | None:
    if not v or not v.get("price"):
        return None
    return v if abs(_paid(p) - v["price"]) <= VERDICT_PRICE_MOVE * v["price"] else None


async def _judge(products: list[Product], plan: dict, raw: str, minp, maxp, codes: list[str],
                 web_queries: list[str], vkey: str = "", known: list[Product] | None = None,
                 info: dict | None = None, budget: float | None = None
                 ) -> tuple[list[Product], list[Product], str, float | None, list[str]]:
    """Budget, stock and trust filters, then the LLM judge picks the offers that fit.
    `vkey` (verdict_key): candidates judged before for this request keep that verdict and aren't sent again; when
    every candidate has one, no LLM call at all. `known`: offers already checked and shown for this request, given to
    the judge for comparison (prices, summary) but not judged again. `info["failed"]` = True when no model judged.
    Returns (candidates judged, picks, summary, typical price, follow-up suggestions)."""
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
    saved = _verdicts(vkey) if vkey else {}
    fresh = []
    for p in pool:
        v = _known_verdict(saved.get(_vid(p)), p)
        if v is None:
            fresh.append(p)
        elif v.get("fit"):
            p.note, p.unconfirmed = v.get("note"), not v.get("sure")
            picks.append(p)
    if isinstance(saved.get("_typical"), dict):
        typical = saved["_typical"].get("price")
    if len(fresh) < len(pool):
        log.info("agent: %d of %d candidates judged before for this request (%d kept)", len(pool) - len(fresh),
                 len(pool), len(picks))
    if fresh:
        items = "\n".join(
            f"{i}. [{p.store}] trust {p.trust_score if p.trust_score is not None else '?'}"
            + (f" ({', '.join(p.trust_signals[:5])})" if p.trust_signals and not p.trusted else "")
            + (f" | CONDITION: {p.condition}" if p.condition else "") + f" | {p.price:g}"
            + (f" ({p.coupon_price:g} with checkout coupon)" if p.coupon_price else "")
            + (f" | -{p.discount_pct}% off" if p.discount_pct else "") + f" | {p.title[:140]}"
            + (f" | link: {_slug(p.url)[:90]}" if _slug(p.url) else "")
            for i, p in enumerate(fresh))
        before = [q for q in (known or []) + picks]
        known_text = ("Offers for this request already checked on their store pages (for comparison and your "
                      "summary; they are NOT candidates, never return them):\n"
                      + "\n".join(f"- [{q.store}] {_paid(q):g} | {q.title[:100]}" for q in before[:12]) + "\n"
                      ) if before else ""
        log.info("agent: judging %d candidates", len(fresh))
        _progress(f"بقارن {len(fresh)} عرض من {len({p.store for p in fresh})} متجر وبشيل اللي مش مطابق...")
        tj = time.monotonic()
        try:
            prompt = JUDGE_PROMPT.format(
                raw=raw, req=json.dumps(plan.get("requirements", []), ensure_ascii=False),
                avoid=json.dumps(plan.get("avoid", []), ensure_ascii=False), minp=minp, maxp=maxp, items=items,
                codes=json.dumps(codes, ensure_ascii=False), known=known_text, can_do=CAN_DO,
                suggested=json.dumps(plan.get("suggested_products") or [], ensure_ascii=False))
            verdict = await llm.ask_json(prompt, timeout=min(20, budget or 20),
                                         budget=budget or plan.get("judge_budget") or JUDGE_BUDGET, prefer="strong",
                                         step="judge")  # the big call: give it time, it decides quality
            TIMES["judge"] = round(time.monotonic() - tj, 1)
            _save_debug("last_judge.json", {"request": raw, "plan": {k: v for k, v in plan.items() if k != "check_hits"},
                                            "prompt": prompt, "verdict": verdict})
            log.info("agent: judge kept %d of %d (model %s)", len(verdict.get("matches") or []), len(fresh),
                     llm.last_model())
            summary = verdict.get("summary", "")
            follow_ups = _strs(verdict.get("suggestions"), 4, 60)
            if isinstance(verdict.get("typical_price"), (int, float)) and verdict["typical_price"] > 0:
                typical = float(verdict["typical_price"])
            best = verdict.get("best_deal_id")
            if isinstance(best, int) and 0 <= best < len(fresh):
                fresh[best].best_deal = True
            # alternatives only when nothing at all fits (not when earlier offers already do)
            alternatives = not verdict.get("matches") and not picks and not known
            chosen = verdict.get("matches") or (verdict.get("alternatives") if alternatives else None) or []
            now, fits = time.time(), {}
            for m in chosen:
                i = m.get("id") if isinstance(m, dict) else None
                if isinstance(i, int) and 0 <= i < len(fresh):
                    p = fresh[i]
                    p.note = m.get("note")
                    p.unconfirmed = m.get("sure") is False
                    if p.unconfirmed and _names_code(p, codes):
                        # the requested model's own number is on the listing: the exact product, whatever the
                        # judge's note says (live 2026-11-24: Fresh FTM-8S flagged "model doesn't match")
                        p.unconfirmed, p.note = False, None
                    p.alternative = alternatives
                    picks.append(p)
                    if not alternatives:
                        fits[id(p)] = p
            if vkey:  # remembered for the next round / the next shopper asking the same
                new = {_vid(p): {"fit": id(p) in fits, "sure": not p.unconfirmed if id(p) in fits else False,
                                 "note": (p.note or "")[:200] if id(p) in fits else "", "price": _paid(p), "at": now}
                       for p in fresh}
                if typical:
                    new["_typical"] = {"price": typical, "at": now}
                await asyncio.to_thread(_save_verdicts, vkey, new)
        except llm.LLMError as e:
            # the simple word rules miss many titles (Arabic vs English, "2000 وات" vs "2000W"); the candidates
            # all came from searches for this request, so show the cheapest of them rather than nothing
            if info is not None:
                info["failed"] = True
            # the request's own words in English (no model): the offers with most of them or the model's number,
            # and always the brand and size the shopper wrote
            eng = english_words(raw)
            guess = [p for p in _fitting(fresh, eng) if _matches_request(p, eng)] if eng else []
            guess = (guess or [p for p in _fallback_pick(fresh, [raw, plan.get("request") or ""] + web_queries)
                               if _matches_request(p, eng)])[:JUDGE_DOWN_RESULTS]
            for p in guess:
                p.unconfirmed = True
            picks += guess
            log.warning("agent: judge failed: %s", e.details or e)
            summary = ("الموديلات المجانية مشغولة دلوقتي فمقدرتش أراجع النتايج بدقة، فدي مرتبة بالسعر بس "
                       "واتأكد من الموديل في كل واحدة. جرب تاني كمان دقيقة لنتيجة أدق.")
    return pool, picks, summary, typical, follow_ups


async def run(raw: str, plan: dict | None = None, budget_text: str | None = None) -> dict:
    """Search for one request. `plan` comes from plan_turn() when the chat already planned it;
    `budget_text` is what the shopper actually wrote (all their messages), used to check a budget is real."""
    TIMES.clear()
    more: dict | None = None  # what another search round in more stores needs, when the shopper may ask for one
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
            log.warning("agent: planner failed: %s", e.details or e)
            return await run_rules(raw, note=BUSY_NOTE)
    seed: list[dict] = list(plan.get("check_hits") or [])  # the planner already saw them (plan_turn hits)
    if websearch.available() and llm.available() and not plan.get("simple") and not plan.get("checked"):
        plan, seed = await _check_plan(plan, plan.get("request") or raw)
    queries = [q for q in plan.get("queries", []) if isinstance(q, str) and q.strip()][:2] or [raw]
    web_queries = [q for q in plan.get("web_queries", []) if isinstance(q, str) and q.strip()][:MAX_WEB_QUERIES] or queries
    codes = [c for c in plan.get("model_codes", []) if isinstance(c, str) and c.strip()]

    TIMES["understand"] = round(time.monotonic() - t0, 1)
    want = plan.get("request") or raw
    maxp, minp = plan.get("max_price"), plan.get("min_price")
    if not plan.get("budget_ok"):  # (a chat turn settled it already, in main: settle_budget)
        minp, maxp = settle_budget(plan, budget_text or raw)
    options = "، ".join(_strs(plan.get("pick_options"), 4, 60))
    vkey = verdict_key(plan, raw, options)
    # what our own price list (the big stores' category pages, read every day) already has for this request. Its
    # titles are English: a request with no English word gets them from a light model first
    shelf = web_queries + queries
    has_words = lambda texts: any(re.search(r"[a-z]", w) for w in _latin_words(texts))  # numbers alone
    # ("غسالة 7 كيلو") don't find a product among English titles: the planner's English product fields first (no
    # LLM call), a light model only when the planner wrote no English at all
    if not has_words(codes + shelf):
        shelf = shelf + _product_words(plan)
    if not has_words(codes + shelf) and llm.available():
        shelf = shelf + await _catalog_queries(want)
    stocked = await asyncio.to_thread(_catalog_offers, shelf, codes)
    # the index's products with the attributes the shopper asked for: checked and shown in seconds, no LLM call
    indexed = await asyncio.to_thread(_index_offers, plan, minp, maxp)
    have = {p.url for p in stocked}
    stocked = stocked + [p for p in indexed if p.url not in have]
    covered = {trust.domain(p.url) for p in stocked}
    judged: dict = {}
    quick = indexed or _shelf_firsts(stocked, raw, plan, minp, maxp)
    fast = asyncio.create_task(_index_first(quick, options, raw, plan, judged)) if quick and VERIFY_PICKS else None
    # the big stores' own search pages, side by side with the web search: no LLM call, every product they sell
    # (not only the ones on our price list's category pages)
    direct = asyncio.create_task(_store_search(queries + web_queries)) if websearch.available() else None
    found_direct: dict = {"products": [], "errors": {}}

    async def first_round():
        """Our price list's offers and the big stores' own search results: judged and checked on their pages, and
        shown first while the other stores are searched on the web."""
        pool = list(stocked)
        if direct:
            got, found_direct["errors"] = await direct
            found_direct["products"] = got
            pool += [p for p in got if p.url not in have]
        return await _first_results(pool, plan, raw, minp, maxp, codes, web_queries, options, vkey,
                                    judged) if pool else None
    first = asyncio.create_task(first_round()) if stocked or direct else None
    famous = _famous_sites(plan)
    searched = web_queries
    try:
        if websearch.available():
            source = "web"
            if reqs := [r for r in plan.get("requirements") or [] if isinstance(r, str)]:
                want += " (" + "; ".join(reqs[:6]) + ")"
            if len(stocked) >= CATALOG_ENOUGH and len(covered) >= 2:
                # the big stores' offers are already here: one web search round (not 2-3) for the other shops, and
                # the well-known group's places go to the big stores we don't read ourselves
                searched = web_queries[:1]
                famous = [d for d in famous if not any(trust._match(c, d) for c in covered)] or famous
                log.info("agent: %d offers from our price list (%s): searching the web with %d query instead of %d",
                         len(stocked), sorted(covered), len(searched), len(web_queries))
            products, errors = await _web_find(searched, seed, famous, want=want,
                                               named=_domains(plan.get("shopper_stores")))
        else:
            source = "stores"
            products, errors = await _search_all(queries)
        early = await first if first else None
        products = found_direct["products"] + products
        errors = {**errors, **found_direct["errors"]}
        if fast:
            await fast
    except BaseException:
        for task in (first, fast, direct):
            if task:
                task.cancel()
        raise
    if early:  # the price list's offers were judged and checked already: only the rest goes to the judge now
        early_ids = {_vid(p) for p in early["pool"]}
        products = [p for p in _dedupe(products) if _vid(p) not in early_ids]
    else:
        products = _dedupe(stocked + products)
    shown_first = early["shown"] if early else []
    pool, picks, summary, typical, follow_ups = await _judge(products, plan, raw, minp, maxp, codes, web_queries,
                                                             vkey, known=shown_first)
    typical = typical or (early["typical"] if early else None)
    picks = list({id(p): p for p in picks}.values())
    gone, unchecked, pool_size = 0, [], len(pool) + (len(early["pool"]) if early else 0)
    if early:
        gone = early["gone"]
        picks = early["later"] + picks
    # (with nothing found at all, too: the round in other stores below is the shopper's only chance of an offer)
    if (picks or shown_first or source == "web") and VERIFY_PICKS:
        if picks:
            _progress(f"بفتح صفحة كل نتيجة ({len(picks)}) وبتأكد إنها متوفرة وإن سعرها لسه زي ما هو...")
        tv = time.monotonic()
        picks, more_unchecked, more_gone = await checked_only(picks, options)
        TIMES["verify"] = round(time.monotonic() - tv, 1)
        seen_ids = {_vid(p) for p in shown_first}
        picks = shown_first + [p for p in picks if _vid(p) not in seen_ids]
        unchecked += more_unchecked
        gone += more_gone
        first_picks = picks
        few = len(picks) < MIN_RESULTS and source == "web" and llm.available()
        seen_urls = {_page_id(p.url) or p.url for p in pool}
        if few and picks:
            # some checked offers: the answer comes now, and the shopper decides whether another round in other
            # stores (about a minute, 2-6 web searches and ~10 LLM calls) is worth waiting for: a button
            more = {"plan": {k: v for k, v in plan.items() if k not in ("check_hits", "reply", "answer")},
                    "raw": raw, "want": want, "options": options, "minp": minp, "maxp": maxp, "codes": codes,
                    "seen": sorted(seen_urls)[:300], "found": [p.to_dict() for p in picks]}
        if few and not picks:
            # nothing at all to show: a shopper with an empty page leaves, so look in other stores before answering
            tw, times = time.monotonic(), dict(TIMES)
            try:
                extra, more_unchecked, more_gone, judged = await asyncio.wait_for(
                    _widen(plan, raw, picks, seen_urls, want, options, minp, maxp, codes), WIDEN_DEADLINE)
            except Exception as e:  # (a time-out too) the first round's results still stand
                log.warning("agent: widening failed: %s: %s", type(e).__name__, e)
                extra, more_unchecked, more_gone, judged = [], [], 0, 0
            TIMES.clear()
            TIMES.update(times, widen=round(time.monotonic() - tw, 1))
            picks = picks + extra
            unchecked += more_unchecked
            gone += more_gone
            pool_size += judged
        summary = _honest_summary(summary, picks)
        if widened := len(picks) - len(first_picks):
            summary += f" (دورت كمان في متاجر تانية ولقيت {widened} عرض زيادة.)"
        if gone:
            summary += (f" (شلت {gone} عرض لما فتحت صفحته: خلصان أو {options} مش متاح فيه.)" if options
                        else f" (شلت {gone} عرض لقيته خلصان لما فتحت صفحته.)")
        if options and any(p.option_ok is None for p in picks):
            summary += f" اللي مكتوب عليه \"اتأكد من {options}\" صفحته مبتقولش إذا كان متاح ولا لا."
    elif shown_first:
        seen_ids = {_vid(p) for p in shown_first}
        picks = shown_first + [p for p in picks if _vid(p) not in seen_ids]
    _flag_price_outliers(picks, typical)
    picks = _ordered(picks, options)
    # noon doesn't answer our crawler, so its prices can't be checked: the products of its public sitemaps that fit
    # are shown apart, as links without a price (app/noon_index.py)
    noon = await asyncio.to_thread(_db_safe, noon_index.links, plan.get("product"), codes) or []
    used = llm.usage()
    return {
        "mode": "llm", "source": source, "timings": dict(TIMES),
        "noon": noon,
        "more_stores": bool(more),  # the page offers "search more stores" (main keeps `_more` for that, server side)
        "_more": more,
        "llm_usage": {"calls": used["calls"], "tokens": used["tokens"], "by_step": dict(used["by_step"])},
        "llm_pages": llm_pages(),
        "page_reads": page_reads(),
        "first_results": len(shown_first),
        "understood": plan.get("understood", ""),
        "request": plan.get("request") or raw,
        "suggested_products": plan.get("suggested_products") or [],
        "queries": searched if source == "web" else queries,
        "from_catalog": len(stocked),
        "requirements": plan.get("requirements", []),
        "options": options,
        "max_price": maxp, "min_price": minp,
        "summary": summary,
        "suggestions": follow_ups,
        "candidates_checked": pool_size,
        "sold_out_dropped": gone,
        "results": [p.to_dict() for p in picks],
        "unchecked": unchecked,  # offers whose price we couldn't confirm on their own page: never shown (kept for the logs)
        "store_errors": errors,
    }


async def _first_results(stocked: list[Product], plan: dict, raw: str, minp, maxp, codes: list[str],
                         web_queries: list[str], options: str, vkey: str, judged: dict | None = None) -> dict | None:
    """The big stores' offers from our own price list: judged and checked on their own pages right away, and sent
    to the page as the first results (in ~10-20 s) while the web search for the other stores goes on. None when the
    judge couldn't run (then they're judged with the rest)."""
    t = time.monotonic()
    try:
        info: dict = {}
        pool, picks, _, typical, _ = await _judge(stocked, plan, raw, minp, maxp, codes, web_queries, vkey, info=info,
                                                  budget=FIRST_JUDGE_BUDGET)
        if info.get("failed"):
            return None
        # close alternatives are only offered when no store at all has the product: not before the web is searched
        picks = [p for p in {id(p): p for p in picks}.values() if not p.alternative]
        shown, unchecked, gone = await checked_only(picks, options, quick=True) if picks else ([], [], 0)
    except Exception as e:  # the full search still runs
        log.warning("agent: first results failed: %s: %s", type(e).__name__, e)
        return None
    # pages the quick check had no time for are checked again with the full answer (not lost)
    later_urls = {u["url"] for u in unchecked}
    later = [p for p in picks if p.url in later_urls]
    TIMES["first_results"] = round(time.monotonic() - t, 1)
    log.info("agent: %d first results from our price list in %.1fs", len(shown), time.monotonic() - t)
    if judged is not None:
        judged["sent"] = True  # the index-only preview (if still running) must not replace this
    if shown:
        firsts = _ordered(list(shown), options)
        _partial({"results": [p.to_dict() for p in firsts], "count": len(firsts), "request": plan.get("request") or raw,
                  "understood": plan.get("understood", ""),
                  "summary": _honest_summary("", firsts),
                  "note": "دي أول عروض من المتاجر الكبيرة واتأكدنا من سعرها، ولسه بدور في باقي المتاجر..."})
    return {"pool": pool, "shown": shown, "later": later, "gone": gone, "typical": typical}


MIN_RESULTS = 4  # fewer offers than this after the checks: one more search round in other stores
WIDEN_DEADLINE = 40  # the whole extra round: a shopper with nothing on screen won't wait longer (live: 71 s for 0)
WIDEN_JUDGE_BUDGET = 12  # seconds for judging that round's offers (the first round's are on screen meanwhile)

WIDEN_PROMPT = """A shopper in Egypt asked: {request}
We searched with: {queries}
We found only {n} offer(s) still for sale that fit{option_note}: {found}
Write 2 NEW web searches that would find MORE Egyptian online stores selling what the shopper wants: other wording
(Arabic as Egyptian shops write it, and English), the stores that specialise in this kind of product, other models or
colors of the same product that also fit the request. Never change what the shopper asked for, never repeat the
searches above.
Reply JSON: {{"web_queries": ["...", "..."], "stores": ["2-6 domains of Egyptian stores likely to sell it that we may
have missed"]}}"""


async def more_stores(ctx: dict) -> dict:
    """The shopper tapped "search more stores" under an answer with few offers: the round run() didn't start on its
    own (`ctx` is the `_more` it returned). The new offers go through the same judge and page checks."""
    TIMES.clear()
    names = {f.name for f in fields(Product)}
    found = [Product(**{k: v for k, v in r.items() if k in names}) for r in ctx.get("found") or []]
    plan, options = ctx.get("plan") or {}, ctx.get("options") or ""
    t = time.monotonic()
    picks, unchecked, gone, judged = await _widen(plan, ctx.get("raw") or "", found, set(ctx.get("seen") or []),
                                                  ctx.get("want") or "", options, ctx.get("minp"), ctx.get("maxp"),
                                                  ctx.get("codes") or [])
    TIMES["widen"] = round(time.monotonic() - t, 1)
    _flag_price_outliers(found + picks)
    picks = _ordered(picks, options)
    used = llm.usage()
    summary = (f"دورت في متاجر تانية ولقيت {len(picks)} عرض زيادة واتأكدت من سعره." if picks
               else "دورت في متاجر تانية ومالقيتش عروض زيادة نقدر نتأكد من سعرها دلوقتي.")
    if gone:
        summary += f" (شلت {gone} عرض لقيته خلصان لما فتحت صفحته.)"
    return {"results": [p.to_dict() for p in picks], "count": len(picks), "summary": summary,
            "candidates_checked": judged, "sold_out_dropped": gone, "unchecked": unchecked, "timings": dict(TIMES),
            "llm_usage": {"calls": used["calls"], "tokens": used["tokens"], "by_step": dict(used["by_step"])}}


async def _widen(plan: dict, raw: str, found: list[Product], seen: set[str], want: str, options: str,
                 minp, maxp, codes: list[str]) -> tuple[list[Product], list[dict], int, int]:
    """Too few offers survived the checks: the LLM writes new searches, and the new offers go through the same
    judge and page checks. Returns (new offers, unchecked, dropped as sold out, candidates judged)."""
    _progress("لقيت عروض قليلة، بدور كمان في متاجر تانية...")
    prompt = WIDEN_PROMPT.format(
        request=plan.get("request") or raw, n=len(found),
        queries=json.dumps(plan.get("web_queries", []), ensure_ascii=False),
        option_note=f" (with {options} in stock)" if options else "",
        found="; ".join(f"{p.title[:60]} at {p.store}" for p in found) or "none")
    fix = await llm.ask_json(prompt, timeout=10, budget=15, prefer="strong", step="widen")
    queries = [q for q in fix.get("web_queries") or [] if isinstance(q, str) and q.strip()][:2]
    if not queries:
        return [], [], 0, 0
    stores = _domains([d for d in fix.get("stores") or [] if isinstance(d, str)])
    log.info("agent: widening with %s (+%s)", queries, stores)
    products, _ = await _web_find(queries, [], list(dict.fromkeys(stores + _famous_sites(plan))), want=want,
                                  named=_domains(plan.get("shopper_stores")))
    have = {_page_id(p.url) or p.url for p in found}
    products = [p for p in _dedupe(products) if (_page_id(p.url) or p.url) not in seen | have]
    if not products:
        return [], [], 0, 0
    pool, picks, _, _, _ = await _judge(products, plan, raw, minp, maxp, codes, queries,
                                        verdict_key(plan, raw, options), known=found, budget=WIDEN_JUDGE_BUDGET)
    picks = list({id(p): p for p in picks}.values())
    if not picks:
        return [], [], 0, len(pool)
    picks, unchecked, gone = await checked_only(picks, options)
    return picks, unchecked, gone, len(pool)


VERIFY_PICKS = True    # open each result's own page before showing it (tests turn it off unless they check it)
VERIFY_MAX = 25        # results checked per search (the cheapest first); the rest aren't shown
VERIFY_DEADLINE = 8    # opening the result pages ourselves
VERIFY_READER_DEADLINE = 6  # pages we couldn't open, through the reader service
VERIFY_LLM_MAX = 16    # pages without clear product data whose main price the LLM reads, per search
VERIFY_LLM_DEADLINE = 18  # those LLM reads
VERIFY_LLM_PARALLEL = 6   # read at once (each on another model)


_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٬٫", "0123456789,.")


def _price_moved(html: str, before: float, now: float) -> bool:
    """Take the product page's price over the one we had only when the shopper can see it on the page and can't
    see the old one: a page's data can hold another product's or another option's price (related items, the
    cheapest size), and the price we had may be the right one."""
    if not before / 2 <= now <= before * 2:
        return False
    seen = extract_mod._visible_numbers(html.translate(_ARABIC_DIGITS))
    return now in seen and before not in seen


QUICK_DEADLINE = 5          # the first results (shown by SEARCH_CAP_SECONDS): opening their pages
QUICK_LLM_DEADLINE = 9      # ... and the LLM reading those without clear product data
FIRST_JUDGE_BUDGET = 12     # the judge of the first results (our price list's offers): seconds
PATIENT_DEADLINE = 25      # background checks (today's deals list): nobody waits, so pages get more time
PATIENT_LLM_DEADLINE = 75


async def _verify_stock(picks: list[Product], options: str = "", patient: bool = False,
                        quick: bool = False, use_llm: bool = True) -> tuple[list[Product], int]:
    """Before results are shown, open each one's own product page: drop it when the store now says sold out,
    take the page's current price when it changed, and mark it `verified` only when that page, opened just now,
    shows the price we'd show (or its own product data says exactly that price). Anything else - a page we
    can't open, a listing page, a copy kept by a search service - can be old or another option's price.
    `options`: what the shopper must pick on the page (e.g. "مقاس 46"): then the LLM reads every page for that
    option, and an offer whose page doesn't offer it or has it sold out is dropped like a sold-out one.
    Returns (picks still for sale, how many were dropped as sold out or without the shopper's option)."""
    for p in picks:
        p.verified = False
    todo = sorted((p for p in picks if _page_id(p.url)), key=_paid)[:VERIFY_MAX]
    if not todo:
        return picks, 0
    sold: set[int] = set()
    to_read: list[tuple[Product, str]] = []  # pages the LLM reads for their main product's price
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(8)
        pages = await _within(PATIENT_DEADLINE if patient else QUICK_DEADLINE if quick else VERIFY_DEADLINE,
                              [_fetch_page(client, sem, p.url) for p in todo])
        retry = []
        for p, res in zip(todo, pages):
            if isinstance(res, Exception):
                retry.append(p)
                continue
            products, html = res
            pid = _page_id(p.url)
            by_link = [q for q in products if _page_id(q.url) == pid]
            same = by_link or (products if len(products) == 1 else [])
            if not p.image:  # offers read from text (no picture there): the product page's own picture
                p.image = next((q.image for q in same if q.image), None) or extract_mod.page_image(html, p.url)
            if same:
                _count_read(False)
                q = min(same, key=_paid)
                if q.in_stock is False:
                    sold.add(id(p))
                    continue
                if by_link and abs(q.price - p.price) >= 1 and _price_moved(html, p.price, q.price):
                    log.info("agent: %s price now %g (was %g)", p.store, q.price, p.price)
                    p.price, p.old_price, p.coupon_price = q.price, q.old_price, q.coupon_price
                    p.discount_pct = _discount(p)
                p.in_stock = True if q.in_stock is None else q.in_stock
                # the page's own data says this price for this product (a single product on the page must
                # also show it: it may be the page's main item or something else)
                p.verified = abs(q.price - p.price) < 1 and (bool(by_link) or shows(_shown_numbers(html), p.price))
                if options and p.verified and q.in_stock is True and _size_option_shown(options, q.title, html):
                    # the shopper's option is a size (256 GB, 55 inch, 10 kg) that this page's own product is: its
                    # name says so and the page has no choices of its own, so its in-stock price is that option's
                    p.option, p.option_ok = options, True
                elif options:  # the page's data is for the product, not the shopper's size/color: the LLM reads it
                    p.verified = False
                if not p.verified:
                    to_read.append((p, html))
                continue
            if extract_mod._page_says_sold_out(html, extract_mod._visible_text(html), p.title):
                sold.add(id(p))
                continue
            to_read.append((p, html))
        if retry:  # the reader service opens the page now (where robots.txt allows): live, not a stored copy
            texts, _ = await fetchers.read_pages(client, [p.url for p in retry],
                                                 deadline=QUICK_DEADLINE if quick else VERIFY_READER_DEADLINE)
            for p in retry:
                if (text := texts.get(p.url)) is None:
                    continue
                if extract_mod._page_says_sold_out(text, text, p.title):
                    sold.add(id(p))
                else:
                    to_read.append((p, text))
    if not use_llm:  # (today's deals list) no LLM call at all: its free daily quota is kept for shoppers' searches
        _check_without_llm(to_read, sold, options)
        to_read = []
    else:  # a clear page needs no LLM read: only the pages that aren't clear go to the LLM
        to_read = [(p, page) for p, page in to_read if not _settled_without_llm(p, page, sold, options)]
    # a number somewhere on the page isn't enough (related items, other sellers, sizes): the LLM reads the price of
    # the page's own product, and only that price is shown
    if to_read and llm.available():
        llm_sem = asyncio.Semaphore(VERIFY_LLM_PARALLEL)

        async def read(i: int, p: Product, page: str):
            async with llm_sem:  # each read starts on another model, so per-minute limits hold
                return await _read_main_price(p.url, page, p.title, spread=i, wanted=options)
        # well-known stores first: when time runs out, the offers shoppers trust most are the ones checked
        # a size/color to check means every page is read (not only those without product data): more reads, more time
        cap, deadline = (VERIFY_LLM_MAX + 8, VERIFY_LLM_DEADLINE + 8) if options else (VERIFY_LLM_MAX, VERIFY_LLM_DEADLINE)
        if patient:  # every page that needs it is read, with time to spare
            cap, deadline = VERIFY_MAX, PATIENT_LLM_DEADLINE
        elif quick:  # the first results: the shopper is waiting, what isn't read by then comes with the full answer
            deadline = min(deadline, QUICK_LLM_DEADLINE)
        to_read = sorted(to_read, key=lambda x: (not x[0].trusted, -(x[0].trust_score or 0), _paid(x[0])))[:cap]
        reads = await _within(deadline, [read(i, p, page) for i, (p, page) in enumerate(to_read)])
        for (p, _), got in zip(to_read, reads):
            if isinstance(got, Exception) or got is None:
                continue
            if got.in_stock is False or got.option_ok is False:
                if got.option_ok is False:
                    log.info("agent: %s: %s not offered or sold out on its page", p.store, options)
                sold.add(id(p))
                continue
            p.option, p.option_ok = got.option, got.option_ok
            if abs(got.price - p.price) >= 1:
                if not p.price / 2 <= got.price <= p.price * 2:  # far off: another item, not this offer
                    continue
                log.info("agent: %s page price %g (was %g)", p.store, got.price, p.price)
                p.price, p.old_price, p.coupon_price = got.price, None, None
                p.discount_pct = _discount(p)
            p.in_stock = True if got.in_stock is None else got.in_stock
            p.verified = True
    for p in todo:
        if not p.verified and id(p) not in sold:
            log.info("agent: %s: no confirmed price on its page, not shown", p.store)
    if sold:
        log.info("agent: %d results sold out on their own page: %s", len(sold),
                 [p.store for p in picks if id(p) in sold])
    return [p for p in picks if id(p) not in sold], len(sold)


def _check_without_llm(to_read: list[tuple[Product, str]], sold: set[int], options: str) -> None:
    """Today's deals list: every page is checked with no LLM call (see _settled_without_llm). The price was read from
    the store's own list page a few hours ago, so the page showing that exact number anywhere is enough."""
    for p, page in to_read:
        _settled_without_llm(p, page, sold, options, near_title=False)


def _settled_without_llm(p: Product, page: str, sold: set[int], options: str, near_title: bool = True) -> bool:
    """A page without product data for the offer, checked with no LLM call: the store's learned recipe
    (app/recipes.py) reads the main product's price and stock; without one, the page must show exactly the price we'd
    show. `near_title`: (a search, where a price can come from anywhere) that number must sit right where the page
    names the product, before the related items lower down. True when the page settled it (checked or sold out);
    False leaves it to the LLM (a size/color to check, reader text, or a page that isn't clear)."""
    if "<" not in page[:2000] or (options and not _size_option_shown(options, p.title, page)):
        return False
    if got := recipes.read(p.url, page):
        price, stock = got
        _count_read(False)
        if stock is False:
            sold.add(id(p))
            return True
        if abs(price - p.price) < 1 and shows(_shown_numbers(page), price):
            p.verified, p.in_stock = True, True
            if options:
                p.option, p.option_ok = options, True
            return True
        return False
    if got := _by_buybox(p.url, page, p.title):
        _count_read(False)
        if got.in_stock is False:
            sold.add(id(p))
            return True
        if abs(got.price - p.price) < 1:
            p.verified, p.in_stock = True, True
            if options:
                p.option, p.option_ok = options, True
            return True
        return False  # its price moved: the next step reads it (and takes the page's price)
    if near_title:
        words = [w for w in re.split(r"\W+", p.title) if len(w) > 2]
        top = _page_text(page, words, NEAR_TITLE_CHARS, base_url=p.url, top=True)
        named = sum(w.lower() in top.lower() for w in words)
        if not words or named < max(2, len(words) * 0.6) or not shows(_shown_numbers(top), p.price):
            return False
    elif not shows(_shown_numbers(page), p.price):
        return False
    _count_read(False)
    p.verified, p.in_stock = True, True
    if options:
        p.option, p.option_ok = options, True
    return True


def _size_option_shown(options: str, title: str, page: str) -> bool:
    """The options the shopper must pick are all sizes ("256 جيجا", "55 بوصة", "10 كيلو"), the product's own name
    shows each of them, and the page offers no choices of its own (sizes or colors with their own stock or price):
    the page sells exactly that size, no LLM read needed. A color, a clothing size ("مقاس 46") or a page with
    choices still goes to the LLM, which reads the page for that option."""
    sizes = []
    for part in options.split("،"):
        eng = english_words(part)
        got = wanted_sizes(eng)
        if not got or eng.strip() != " ".join(f"{n} {u}" for n, u in got):
            return False  # (a word that isn't a size: a color, a fit...)
        sizes += got
    if not all(shows_size(title, n, u) for n, u in sizes):
        return False
    if extract_mod.variant_stock(page):
        return False
    # the page's name for its product shows the size too (not only our listing's title)
    h1 = re.search(r"<h1\b[^>]*>(.*?)</h1>", page, re.S | re.I)
    name = " ".join(unescape(re.sub(r"<[^>]+>", " ", h1.group(1))).split()) if h1 else ""
    if not name or not all(shows_size(name, n, u) for n, u in sizes):
        return False
    # where the page shows its product (name, price, choices), another size of the same kind its name doesn't have
    # ("128GB | 256GB" buttons) means the shopper picks the size there: the LLM reads that
    top = normalize(_page_text(page, name.split(), NEAR_TITLE_CHARS, base_url="", top=True))
    named = normalize(name)
    for _, unit in sizes:
        ar = AR_UNITS.get(unit, unit)
        for m in re.finditer(rf"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:-\s*)?(?:{unit}|{ar})", top):
            if not shows_size(named, m.group(1), unit):
                return False
    return True


def shows(shown: set[float], price: float) -> bool:
    """The page shows this price: the same number, or its pounds part (stores often write the piasters apart,
    small and raised: "EGP2,413" then "43", so the page text holds 2413 and 43, not 2413.43)."""
    return price in shown or int(price) in shown


def _shown_numbers(page: str) -> set[float]:
    return extract_mod._visible_numbers(page.translate(_ARABIC_DIGITS))


async def checked_only(picks: list[Product], options: str = "", patient: bool = False,
                       quick: bool = False, use_llm: bool = True) -> tuple[list[Product], list[dict], int]:
    """What may be shown with a price: only offers whose own page, opened just now, shows that price (and, when the
    shopper named a size/color..., offers that option). Returns (those offers, the others (never shown to the
    shopper), how many were sold out or without that option)."""
    if not (picks and VERIFY_PICKS):
        return picks, [], 0
    picks, gone = await _verify_stock(picks, options, patient, quick, use_llm)
    unchecked = [{"store": p.store, "title": p.title, "url": p.url} for p in picks if not p.verified]
    now = time.time()
    for p in picks:
        if p.verified:
            p.checked_at = now
    return [p for p in picks if p.verified], unchecked, gone


def _honest_summary(summary: str, picks: list[Product]) -> str:
    """The judge's summary names prices from before the page check; they may now be wrong or not shown at all.
    Keep only its sentences without numbers, and state the cheapest checked price ourselves."""
    keep = without_counts(" ".join(x for x in re.split(r"(?<=[.!؟?])\s+", summary or "") if x and not _NUMBER.search(x)))
    if not picks:
        return keep
    best = min(picks, key=_paid)
    lead = f"أرخص سعر اتأكدنا منه من صفحة المتجر دلوقتي: {_paid(best):,.0f} في {best.store}."
    return f"{lead} {keep}".strip()


async def recheck(payload: dict) -> dict:
    """A saved answer (served again within a few hours): its prices are checked on the store pages again."""
    rows = payload.get("results") or []
    if not rows or not VERIFY_PICKS:
        return payload
    names = {f.name for f in fields(Product)}
    picks = [Product(**{k: v for k, v in r.items() if k in names}) for r in rows]
    extra = {id(p): {k: v for k, v in r.items() if k not in names} for p, r in zip(picks, rows)}
    shown, unchecked, gone = await checked_only(picks, payload.get("options") or "")
    lo, hi = _amount(payload.get("min_price")), _amount(payload.get("max_price"))
    shown = [p for p in shown if (hi is None or _paid(p) <= hi) and (lo is None or p.price >= lo)]  # price moved
    return {**payload, "results": [extra[id(p)] | p.to_dict() for p in _ordered(shown, payload.get("options") or "")],
            "unchecked": unchecked + [u for u in payload.get("unchecked") or [] if u not in unchecked],
            "count": len(shown), "summary": _honest_summary(payload.get("summary", ""), shown),
            "sold_out_dropped": (payload.get("sold_out_dropped") or 0) + gone}


CATALOG_ENOUGH = 6  # offers from our own price list (in 2+ stores) that make one web search round enough
CATALOG_MAX = 15     # at most this many of them go to the judge (the web's shops need room too)
_LATIN_WORD = re.compile(r"[a-z0-9][a-z0-9+-]*[a-z0-9+]|[a-z0-9]")


CATALOG_WORDS_PROMPT = """A shopper in Egypt asked for: {request}
Our price list has the English product titles of big online stores (e.g. "Samsung Galaxy A55 5G 256GB 8GB RAM Navy",
"Fresh Automatic Washing Machine 7 Kg Silver"). Write 1-2 short English searches that would find this product's
titles there: brand and model if named, product type, the size or capacity the shopper gave. Never add a brand or a
spec the shopper didn't give.
Reply JSON: {{"queries": ["...", "..."]}}"""


def _latin_words(texts: list[str]) -> list[str]:
    words: list[str] = []
    for text in texts:
        for w in _LATIN_WORD.findall(str(text).lower()):
            if w not in _SEARCH_NOISE and (len(w) >= 3 or any(c.isdigit() for c in w)) and w not in words:
                words.append(w)
    return words


def _product_words(plan: dict) -> list[str]:
    """The planner's English description of the product ("Toshiba washing machine 8 kg"): words our price list's
    English titles use, with no extra LLM call. [] when it gave none."""
    want = plan.get("product") if isinstance(plan.get("product"), dict) else {}
    text = " ".join(w for k in ("brand", "model", "kind", "size") if (w := str(want.get(k) or "").strip()))
    return [text] if re.search(r"[a-zA-Z]", text) else []


async def _catalog_queries(request: str) -> list[str]:
    """The request only in Arabic (no planner, or it wrote no English): a light model writes the English words our
    price list's titles use, so the big stores' offers aren't skipped. [] when no model answers."""
    try:
        data = await llm.ask_json(CATALOG_WORDS_PROMPT.format(request=request[:300]), timeout=6, budget=8,
                                  prefer="bulk", step="catalog_words")
    except llm.LLMError as e:
        log.info("agent: no English words for the price list: %s", e)
        return []
    return [q for q in data.get("queries") or [] if isinstance(q, str) and q.strip()][:2]


def _catalog_offers(queries: list[str], codes: list[str]) -> list[Product]:
    """Our price list's products that fit the request's words, best fit first (see catalog.offers). The list is
    read from the stores' English pages: the model numbers and the Latin words of the searches find them."""
    words = _latin_words(codes + queries)
    if not words:
        return []
    # the rarest words first: a model number ("a55", "15") or a brand finds the product, "phone" finds everything
    words.sort(key=lambda w: (not any(c.isdigit() for c in w), -len(w)))
    pool = _db_safe(catalog.offers, words[:6]) or []
    if not pool:
        return []
    picked = _fallback_pick(pool, codes + queries)
    return sorted(picked[:CATALOG_MAX], key=_paid)


_SEARCH_NOISE = {"price", "prices", "egypt", "سعر", "اسعار", "أسعار", "مصر", "في", "من", "the", "and", "for", "with",
                 "best", "buy", "online", "جنيه", "تحت", "عايز"}


INDEX_MAX = 12  # offers of the index shown first (checked on their pages) before any judge


def _squash_size(text: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", (text or "").lower().replace("gigabyte", "gb").replace("inches", "inch")
                  .replace("kilo", "kg").replace("kgs", "kg"))


def _index_offers(plan: dict, minp=None, maxp=None) -> list[Product]:
    """The index's products whose attributes (what the LLM made of each title, offline) are what the planner says the
    shopper wants: the same kind, and the brand / size / model when the shopper named them. No LLM call, a database
    query: these are the first offers a shopper sees (each still checked on its own page before it's shown)."""
    want = plan.get("product") if isinstance(plan.get("product"), dict) else {}
    kind = str(want.get("kind") or "").lower().strip()
    if not kind:
        return []
    brand, size, model = (str(want.get(k) or "").lower().strip() for k in ("brand", "size", "model"))
    words = [w for w in ([model] if model else []) + ([brand] if brand else []) + [kind] if w]
    rows = _db_safe(db.catalog_like, words, market.current().code, time.time() - catalog.FRESH_HOURS * 3600,
                    300, time.time() - catalog.FOUND_DAYS * 86400) or []
    kind_words = set(kind.split())
    out = []
    for r in rows:
        try:
            a = json.loads(r.get("attrs") or "{}")
        except ValueError:
            continue
        if not a or a.get("accessory"):
            continue
        theirs = set(str(a.get("kind") or "").lower().split())
        if not theirs or not (kind_words <= theirs or theirs <= kind_words):  # "washing machine" vs "machine": no
            continue
        if brand and str(a.get("brand") or "").lower() != brand:
            continue
        if size and _squash_size(size) not in (_squash_size(a.get("size") or ""), ) \
                and _squash_size(size) not in _squash_size(r["title"]):
            continue
        p = Product(store=r["store"], title=r["title"], price=r["price"], url=r["url"], image=r["image"] or None,
                    old_price=r["old_price"])
        if model and not _names_code(p, [model]):
            continue
        if (maxp is not None and p.price > maxp) or (minp is not None and p.price < minp):
            continue
        if r.get("trust") is None and r.get("category") != db.FOUND:
            p.trusted, p.trust_score, p.trust_signals = True, 100, ["متجر مشهور"]
        else:
            p.trusted, p.trust_score = trust.is_known(r["url"]), r.get("trust")
            if not p.trusted and (p.trust_score or 0) < MIN_TRUST:
                continue
        out.append(p)
    out = sorted({p.url: p for p in out}.values(), key=_paid)
    # a need ("موبايل كويس للتصوير في حدود 12 ألف"): the products the planner picked for that need first, not the
    # cheapest phones of the kind (live 2026-11-26: entry phones shown for a camera request)
    fits = [p for p in out if _suggested(p.title, plan)] if plan.get("need_based") else []
    if len(fits) >= 2:
        out = fits
    elif fits:
        out = fits + [p for p in out if p not in fits]
    # several stores, not the 12 cheapest of one
    by_store: dict[str, int] = {}
    picked = []
    for p in out:
        if by_store.get(p.store, 0) < PER_STORE_FIRST // 2:
            by_store[p.store] = by_store.get(p.store, 0) + 1
            picked.append(p)
    return picked[:INDEX_MAX]


_OPTION_WORDS = {"لون", "مقاس", "موديل", "سعه", "مساحه", "رام", "جيجا", "جيجابايت", "تيرا", "size", "color", "colour",
                 "model", "storage", "gb", "tb", "ram"}


def shopper_named(option: str, said: str) -> bool:
    """A size/color option the planner read is something the shopper wrote ("لون أسود" when they said "اسود"), not
    one it made up: live 2026-11-26 an AC search showed "check white, model AY-A12ZSE, ..." on every offer, words the
    shopper never wrote. A word of the option (not "لون"/"مقاس"...) or its number must be in their messages."""
    have = _fold(said)
    words = [_fold(w) for w in re.split(r"[\s،,/]+", option or "") if w]
    words = [w[2:] if w.startswith("ال") and len(w) > 4 else w for w in words if w not in _OPTION_WORDS]
    for w in words:
        if re.fullmatch(r"\d+(?:\.\d+)?", w):
            if re.search(rf"(?<![\d.]){re.escape(w)}(?![\d.])", have):
                return True
        elif len(w) >= 2 and (w in have or _stem(w) in have):
            return True
    return False


def _shelf_firsts(stocked: list[Product], raw: str, plan: dict, minp=None, maxp=None) -> list[Product]:
    """When the index has nothing with the attributes asked for (most titles aren't read by a model yet), our price
    list's offers that are clearly the brand, kind and size the shopper wrote: shown in seconds like the index's,
    instead of after the whole first round (live 2026-11-26: first offers at 36-62 s). For a need ("good camera
    phone") only the products the planner suggested for it, never just the cheapest of the kind."""
    eng = english_words(raw)
    if not eng or not stocked:
        return []
    out = [p for p in stocked if _matches_request(p, eng) and not _is_part(p.title)
           and (maxp is None or p.price <= maxp) and (minp is None or p.price >= minp)]
    if plan.get("need_based"):
        out = [p for p in out if _suggested(p.title, plan)]
    by_store: dict[str, int] = {}
    picked = []
    for p in sorted(out, key=_paid):
        if by_store.get(p.store, 0) < PER_STORE_FIRST // 2:
            by_store[p.store] = by_store.get(p.store, 0) + 1
            picked.append(p)
    return picked[:INDEX_MAX]


def _suggested(title: str, plan: dict) -> bool:
    """The title is one of the products the planner suggested for the shopper's need: most of that product's words
    (70%, and every model number in it) are in the title."""
    have = _squash_size(title)
    for name in plan.get("suggested_products") or []:
        words = [_squash_size(w) for w in re.split(r"[\s,/()+-]+", str(name).lower()) if _squash_size(w)]
        if not words:
            continue
        codes = [w for w in words if re.search(r"\d", w)]
        if all(c in have for c in codes) and sum(w in have for w in words) >= max(1, round(len(words) * 0.7)):
            return True
    return False


async def _index_first(found: list[Product], options: str, raw: str, plan: dict, judged: dict) -> list[Product]:
    """The index's offers, checked on their own pages, sent to the shopper right away (before any judge). The judge
    still sees them with the rest: the next answer replaces these if it drops one."""
    t = time.monotonic()
    shown, _, _ = await checked_only([Product(**{f.name: getattr(p, f.name) for f in fields(p)}) for p in found],
                                     options, quick=True)
    TIMES["index_first"] = round(time.monotonic() - t, 1)
    log.info("agent: %d of %d index offers checked in %.1fs", len(shown), len(found), time.monotonic() - t)
    if shown and not judged.get("sent"):
        firsts = _ordered(list(shown), options)
        _partial({"results": [p.to_dict() for p in firsts], "count": len(firsts), "request": plan.get("request") or raw,
                  "understood": plan.get("understood", ""), "summary": _honest_summary("", firsts),
                  "note": "دي أول عروض لقيناها في فهرس الأسعار بتاعنا واتأكدنا من سعرها دلوقتي، ولسه بدور في باقي المتاجر..."})
    return shown


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
    # a tenth of what a genuine one costs is not a price: a placeholder ("100" on a call-for-price or sold-out page),
    # a deposit or an accessory. Only the judge's typical price can tell (the others may all carry the same placeholder)
    for p in list(picks) if typical else []:
        if _paid(p) * UNIT_ERROR < typical:
            picks.remove(p)
            log.info("agent: dropped %s at %s (a genuine one costs about %s): not a real price", p.store, _paid(p), typical)
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
            ref, where = median, "المتاجر التانية هنا"  # the other offers' prices were checked on their pages
        elif typical and _paid(p) < TOO_CHEAP * typical:
            ref, where = typical, ""  # the judge's estimate: never shown as a price
        else:
            continue
        p.unconfirmed = True
        p.best_deal = False
        usual = f"من سعره في {where} (حوالي {ref:,.0f} ج.م)" if where else "من سعره المعتاد"
        p.note = (f"السعر أقل بكتير {usual}، ممكن يكون تقليد أو غلط في الإعلان أو "
                  "موديل/حالة مختلفة. اتأكد من صفحة المنتج والمتجر قبل الشراء.")
        log.info("agent: price outlier %s %s (reference %s)", p.store, _paid(p), ref)


BUSY_NOTE = "الموديلات الذكية مشغولة دلوقتي، فدورتلك بالكلام اللي كتبته زي ما هو."
FILLER = {"دورلي", "دورلى", "شوفلي", "شوفلى", "هاتلي", "هاتلى", "ابحثلي", "ابحث", "دور", "طول", "بسرعه", "بسرعة",
          "حاجه", "حاجة", "انا", "عايزه", "عاوزه", "محتاج", "محتاجه", "لو", "سمحت", "please", "search"}
_DIACRITICS = re.compile(r"[\u064b-\u0652\u0640]")  # tashkeel and tatweel: "دوّرلي" -> "دورلي"


def plain_request(texts: list[str]) -> str:
    """What the shopper asked for, without a model: their messages in order (each once), without filler words
    like 'دورلي على طول' (often a tapped option), so a web search finds the product itself."""
    from .query import STOPWORDS
    seen, words = set(), []
    for text in texts:
        for w in _DIACRITICS.sub("", text or "").split():
            k = w.strip(".,!?؟،").lower()
            if k and k not in FILLER and k not in STOPWORDS and k not in seen:
                seen.add(k)
                words.append(w.strip(".,!?؟،"))
    return " ".join(words)[:300] or " ".join(texts)[:300]


async def run_rules(raw: str, note: str) -> dict:
    """No model could plan the search. With a web-search key: the normal web search for the shopper's own words
    (search engines understand them), with a shorter judge; if no model judges either, the word rules pick. Without
    one: the built-in store searches with simple word rules."""
    if websearch.available():
        q = plain_request([raw])
        # every model is resting for a while: the web search's pages couldn't be judged either (live 2026-11-25:
        # 50-80 s, then nothing). Answer in seconds from our own price list instead, each price checked on its page
        wait = None
        if llm.available() and ((wait := await llm.free_in()) is None or wait > OUTAGE_WAIT):
            return await _outage_answer(raw, q, wait, note)
        # no model could plan it, though some aren't resting: they're failing or slow now (live 2026-11-25: 141 s,
        # then nothing). The quick answer first; the slow search only when it found nothing
        quick = await _outage_answer(raw, q, wait, note)
        if quick["results"]:
            return quick
        out = await run(q, plan={"action": "search", "request": q, "queries": [q], "web_queries": [q],
                                 "simple": True, "judge_budget": 20}, budget_text=raw)
        out["understood"] = note
        out["mode"] = "rules"
        return out
    q: ParsedQuery = parse_query(raw)
    products, errors = await _search_all([q.search_text])
    picks, unchecked, _ = await checked_only(rank(products, q))
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
        "unchecked": unchecked,
        "store_errors": errors,
    }


OUTAGE_WAIT = 5  # seconds: a model free again within this is worth the normal search; later, the quick answer
OUTAGE_MAX = 12  # offers of our price list checked on their pages when no model can help
_AR_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})


def _fold(text: str) -> str:
    """Arabic spelling variants made one (أ/ا, ة/ه, ى/ي) and lower case, for matching words."""
    return _DIACRITICS.sub("", (text or "").lower().translate(_AR_FOLD))


def _stem(word: str) -> str:
    """The part of an Arabic word its forms share ("السماعات" / "سماعة" -> "سماع"), so a title's keywords match."""
    w = _fold(word)
    if re.search(r"[\u0600-\u06ff]", w):
        w = w[2:] if w.startswith("ال") and len(w) > 4 else w
        for end in ("ات", "ه", "ين"):
            if w.endswith(end) and len(w) - len(end) >= 3:
                w = w[:-len(end)]
                break
    return w


def _has_word(hay: str, word: str) -> bool:
    """A number the shopper wrote is a whole number in the title ("16" is not in the TV code "FJ16EX"; "16 قدم" or
    "16ft" is); any other word may be part of a longer one (Arabic stems, "galaxy" in "galaxy-a15")."""
    if word.isdigit():
        return bool(re.search(rf"(?<![a-z0-9.]){word}(?![0-9.])", hay))
    return word in hay


def _shelf_offers(raw: str, request: str) -> list[Product]:
    """No model can understand the request: our price list's products whose title or keywords (English and Arabic,
    from the indexing step) have most of the shopper's words, cheapest first. A model number they wrote must be
    there. Each is still checked on its own page before it's shown."""
    from .query import STOPWORDS, AR_VOCAB
    parsed = parse_query(raw)
    # each word of the request, with its English word when we know one ("سماعات" -> "earbuds")
    text = re.sub(r"(?:تحت|اقل من|أقل من|فوق|اكتر من|أكثر من|لحد|under|below|over|above)\s*\d[\d,]*", " ", request)
    groups: list[set[str]] = []
    for w in text.split():
        w = w.strip(".,!?؟،()").lower()
        if not w or w in STOPWORDS or w in _SEARCH_NOISE or w in FILLER or (len(w) < 3 and not any(c.isdigit() for c in w)):
            continue
        if w.isdigit() and len(w) >= 4:  # "2000" with no unit is mostly a price, not a product word
            continue
        alts = {_stem(w)} | ({AR_VOCAB[w]} if w in AR_VOCAB else set()) | ({AR_WORDS[w]} if w in AR_WORDS else set())
        if alts not in groups:
            groups.append(alts)
    if not groups:
        return []
    codes = [next(iter(g)) for g in groups if re.search(r"[a-z]", next(iter(g))) and re.search(r"\d", next(iter(g)))]
    words = sorted({a for g in groups for a in g}, key=lambda w: (not any(c.isdigit() for c in w), -len(w)))
    now = time.time()
    rows = _db_safe(db.catalog_like, words[:8], market.current().code, now - catalog.FRESH_HOURS * 3600, 300,
                    now - catalog.FOUND_DAYS * 86400) or []
    need = max(1, -(-len(groups) * 6 // 10))  # most of the words (60%, rounded up)
    kinds = kinds_in(english_words(request))
    scored = []
    for r in rows:
        if not same_kind(r.get("title") or "", kinds):
            continue  # a "Sharp fridge" request never fills its places with Sharp TVs
        hay = _fold(" ".join(str(r.get(k) or "") for k in ("title", "keywords", "attrs")))
        hits = sum(any(_has_word(hay, a) for a in g) for g in groups)
        if hits < need or any(c not in _squash_size(hay) for c in map(_squash_size, codes)):
            continue
        price = r.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            continue
        if (parsed.max_price and price > parsed.max_price) or (parsed.min_price and price < parsed.min_price):
            continue
        try:
            accessory = bool(json.loads(r.get("attrs") or "{}").get("accessory"))
        except (ValueError, AttributeError):
            accessory = False
        p = Product(store=r["store"], title=r["title"], price=price, url=r["url"], image=r["image"] or None,
                    old_price=r["old_price"])
        if r.get("trust") is None and r.get("category") != db.FOUND:
            p.trusted, p.trust_score, p.trust_signals = True, 100, ["متجر مشهور"]
        else:
            p.trusted, p.trust_score = trust.is_known(r["url"]), r.get("trust")
            if not p.trusted and (p.trust_score or 0) < MIN_TRUST:
                continue
        scored.append((accessory, -hits, price, p))
    if any(not a for a, *_ in scored):  # a phone's cases and screen guards share its words: the product itself first
        scored = [x for x in scored if not x[0]]
    scored.sort(key=lambda x: (x[1], x[2]))
    picked, by_store = [], {}
    for _, _, _, p in scored:
        if p.url not in {q.url for q in picked} and by_store.get(p.store, 0) < 4:
            by_store[p.store] = by_store.get(p.store, 0) + 1
            picked.append(p)
        if len(picked) >= OUTAGE_MAX:
            break
    return sorted(picked, key=_paid)


def _matches_request(p: Product, eng: str) -> bool:
    """With no model to judge it, an offer must at least be the brand, the kind of product and the size the shopper
    wrote: a Samsung 55-inch TV request never shows a 32-inch monitor or a 43-inch TV, a Toshiba 10 kg washer never a
    Fresh 6 kg one, a Sharp fridge never a Sharp TV. The brand may be written in English or Arabic in the title."""
    if not all(names_brand(p.title, brand) for brand in {w for w in eng.split() if w in BRANDS}):
        return False  # as a whole word: an LG request never shows a Levon TV whose model code is "50LG8070Q"
    if not same_kind(p.title, kinds_in(eng)):
        return False  # a "Sharp fridge" request never shows Sharp TVs
    if lacks_wanted(p.title, eng):
        return False  # "بارد ساخن" never shows a "Cooling Only" air conditioner
    return all(shows_size(p.title, n, u) for n, u in wanted_sizes(eng))


async def _nothing() -> tuple[list, dict]:
    return [], {}


# words that make a title an accessory or a part for the product the shopper named ("case cover compatible with
# Xiaomi Redmi Note 14 Pro"): with no model to judge, those are never shown for a search of the product itself
PART_WORDS = {"case", "cases", "cover", "covers", "protector", "tempered", "film", "skin",
              "holder", "mount", "replacement", "compatible", "sticker", "stickers", "pouch", "sleeve", "spare",
              "bracket", "tpu", "bumper", "shockproof"}


def _is_part(title: str) -> bool:
    words = set(re.split(r"[^a-z0-9\u0600-\u06ff]+", (title or "").lower()))
    return bool(words & PART_WORDS)


def _fitting(products: list[Product], eng: str) -> list[Product]:
    """The store search's products that are what the shopper wrote (no model to judge them): most of the request's
    words in the title (60%), or every model number they wrote when there is one (a word with letters and digits,
    e.g. r50i: Arabic titles have little else in English), cheapest first. Accessories that share a phone's words
    (a case "compatible with" it) are left out unless the shopper asked for one, and so are those under a quarter of
    the middle price of the fitting products."""
    words = [_squash_size(w) for w in eng.split() if _squash_size(w)]
    if not words:
        return []
    codes = [w for w in words if re.search(r"[a-z]", w) and re.search(r"\d", w)]
    # a model's own number (r50i, a55, 15irh10) names the product; a size or a chip (512gb, i5) doesn't
    names = [c for c in codes if len(c) >= 4 and not re.fullmatch(r"\d+(gb|tb|inch|kg|hp|mah|w|liter)", c)]
    need = max(1, -(-len(words) * 6 // 10))
    want_part = bool(set(eng.lower().split()) & PART_WORDS)
    out = []
    for p in products:
        title = _squash_size(p.title)
        if not want_part and _is_part(p.title):
            continue
        if all(c in title for c in codes) and (names or sum(w in title for w in words) >= need):
            out.append(p)
    if len(out) >= 3:
        mid = sorted(_paid(p) for p in out)[len(out) // 2]
        out = [p for p in out if _paid(p) >= mid / 4]
    return sorted(out, key=_paid)


def _wait_text(wait: float | None) -> str:
    if wait is None or wait > 3 * 3600:
        return "بعد شوية"
    mins = max(1, round(wait / 60))
    return "كمان دقيقة" if mins == 1 else f"كمان {mins} دقايق" if mins <= 10 else f"كمان {mins} دقيقة"


async def _outage_answer(raw: str, request: str, wait: float | None, note: str) -> dict:
    """Every free model is resting: an answer in seconds, not a minute of searching that ends empty. Our price list's
    offers for the shopper's words, each opened on its store page just now and shown only if that page shows the
    price (no model guesses anything), marked "check the model" since no model compared them to the request; with
    none, an honest message and today's deals."""
    t0 = time.monotonic()
    _progress("الموديلات الذكية مشغولة دلوقتي، فبدورلك في فهرس الأسعار بتاعنا وبتأكد من كل سعر من صفحة المتجر...")
    eng = english_words(request)
    # our price list, and the big stores' own search pages for the request in English words (no model needed)
    shelf, (found, _) = await asyncio.gather(asyncio.to_thread(_shelf_offers, raw, request),
                                             _store_search([eng]) if eng else _nothing())
    have = {p.url for p in shelf}
    picks = shelf + [p for p in _fitting(found, eng) if p.url not in have][:OUTAGE_MAX]
    picks = [p for p in picks if _matches_request(p, eng)]
    shown, unchecked, gone = await checked_only(picks, quick=True) if picks else ([], [], 0)
    for p in shown:
        p.unconfirmed = True
    later = _wait_text(wait)
    if shown:
        summary = (f"الموديلات الذكية مشغولة دلوقتي، فدي عروض من فهرس الأسعار بتاعنا اتأكدنا من سعرها من صفحة المتجر "
                   f"دلوقتي حالاً، بس اتأكد إن الموديل هو اللي عايزه. جرب تاني {later} لبحث أدق في كل المتاجر.")
    else:
        summary = (f"الموديلات الذكية مشغولة دلوقتي ومقدرتش أدور بدقة. جرب تاني {later}، "
                   "ولحد ساعتها شوف عروض النهارده اللي اتأكدنا من أسعارها.")
    TIMES.clear()
    TIMES["outage"] = round(time.monotonic() - t0, 1)
    log.info("agent: every model resting (free in %s s): %d of %d offers from our price list checked in %.1fs",
             None if wait is None else round(wait), len(shown), len(picks), time.monotonic() - t0)
    return {
        "mode": "rules", "source": "catalog", "outage": True, "deals_link": True, "timings": dict(TIMES),
        "understood": note, "request": raw, "suggested_products": [], "queries": [request], "requirements": [],
        "max_price": None, "min_price": None, "summary": summary, "candidates_checked": len(picks),
        "sold_out_dropped": gone, "results": [p.to_dict() for p in _ordered(shown, "")], "unchecked": unchecked,
        "store_errors": {},
    }
