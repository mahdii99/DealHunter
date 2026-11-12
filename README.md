# Easy Found (DealHunter)

[![tests](https://github.com/mahdii99/DealHunter/actions/workflows/tests.yml/badge.svg)](https://github.com/mahdii99/DealHunter/actions/workflows/tests.yml)

**An AI shopping agent that finds the cheapest real offer for what you ask for, across online stores in Egypt, Saudi Arabia and the UAE.**

Egypt is live; Saudi Arabia and the UAE are set up in the country config (`MARKET=SA` / `MARKET=AE`: currency,
local domains, major stores) and being tuned next.

You describe what you want in plain Arabic or English, even if you don't know the exact product name, e.g.
*"an 18 ft silver fridge from a good brand"* or *"earbuds with noise cancelling under 1500"*. The agent works out what
you mean, searches the web for stores that sell it, opens the store pages itself, reads the real prices, throws out
wrong matches (other variants, used items, sold-out items, foreign stores, replicas), and returns the matching offers
from trustworthy shops, cheapest first.

The project runs entirely on free tiers: no paid API is needed.

> Arabic README: [README.ar.md](README.ar.md)

## What it does

- **A salesperson, not a search box.** The agent talks like a good salesperson standing in every online store: it asks a short question when the request is too open, recommends products that fit a need before searching (then finds prices for the one you pick), answers questions about the results with a comparison table (using store trust, stock checks and price history), and offers next steps to tap. Follow-ups refine the previous request ("Samsung only", "cheaper", "in black") instead of starting over. Replies appear while they are written.
- **Saved chats and voice.** Signed-in shoppers find their past conversations on any device; a mic button lets you speak your request (the browser's own speech recognition, free).
- **Need-based search.** "A phone whose battery lasts two days" becomes concrete specs (5000 mAh+) and suggested models to look for.
- **Finds stores by itself.** No hand-picked store list. Each query is searched twice, on well-known retailers and on everyone else, so smaller (often cheaper) shops compete fairly.
- **Prices come from the store, not the LLM.** Products are extracted from each page's own data (JSON-LD, Next.js data, embedded JSON). The LLM only reads a page when it has no structured data.
- **Strict matching.** Variants are different products (knob vs. digital, Hero vs. Hero Plus, 50 L vs. 30 L). When the listing doesn't confirm the exact variant, the result is kept but flagged "check the model".
- **Trust score for unknown shops** (0-100) from signals on their own site: local phone number, cash on delivery, return policy, warranty, address. Replica wording is a red flag.
- **Real discounts only.** Discount % comes from the store's own old price or checkout coupon, and every price is sanity-checked against the market (a 2,000 EGP fridge is a wrong currency or an installment, not a deal).
- **Price history.** Every price shown is saved, so results show the lowest price seen before and when a price dropped. This is the base for price-drop alerts.
- **Accounts with Google sign-in.** Shoppers pick their region at sign-up (Egypt, Saudi Arabia, UAE), save preferences (e.g. trusted shops only) and get a higher daily limit.
- **Price-drop alerts.** "Follow price" on any result; when the product gets cheaper (seen in any search, or in a scheduled re-check every 6 hours) the shopper gets a notification on the site, and optionally an email.
- **Live progress.** The page streams what the agent is doing ("found 16 pages in 11 stores", "comparing 30 offers...").

## How it works

```
request ──► understand (LLM) ──► web search ──► open store pages ──► extract products ──► judge (LLM) ──► cheapest first
             plan queries,        famous stores      in parallel,        structured data;      right product?
             model numbers,       + all other        time-boxed          LLM reads pages       legit store?
             requirements         shops                                  without data          real price?
```

| Stage | Where | Notes |
| --- | --- | --- |
| Understand the chat | `app/agent.py` `plan_turn` | One LLM call: decides to search / ask / recommend / answer, merges follow-ups, plans an Arabic and an English query (with model numbers) |
| Web search | `app/websearch.py` | Tavily, Exa and Firecrawl, rotated to share free quotas, with automatic fallback |
| Open pages | `app/agent.py` `_web_find` | Parallel, with per-stage deadlines so one slow site can't hold up the answer |
| Pages we can't open | `app/fetchers.py` | Retried through a reader service (Tavily Extract, then Jina Reader), only where the site's robots.txt allows it |
| Extract products | `app/extract.py` | Generic: JSON-LD, Next.js flight data, Shopify JSON; fixes prices stored in cents; detects stock and condition |
| Judge | `app/agent.py` `JUDGE_PROMPT` | Keeps exact matches from legit stores, picks the best value deal, offers close alternatives if nothing matches |
| Trust | `app/trust.py` | Known retailers + a page-signal score for everyone else; drops foreign stores and price-comparison sites |
| LLM client | `app/llm.py` | Gemini, Groq, Mistral, NVIDIA, Cloudflare and OpenRouter free tiers; picks live models, rests busy ones, remembers daily quotas across restarts |
| Price DB | `app/db.py` | SQLite: request cache, price history (`GET /api/history?url=`), and sites whose pages never open, which later searches skip (`python -m scripts.blocked_sites`) |
| Country config | `app/market.py` | Currency, local domains and stores per market (`MARKET=EG`, Saudi Arabia and UAE prepared) |

## Tech stack

Python 3.12 · FastAPI · httpx (async) · selectolax · SQLite / Postgres (Supabase) · Google Identity Services · vanilla JS (no build step) · pytest · Docker

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                                 # then add your free keys (see the comments in the file)
python -m scripts.check_keys                         # checks every key with one tiny request
uvicorn app.main:app --reload                        # open http://localhost:8000
```

At minimum you need one LLM key (e.g. a free Gemini key from Google AI Studio) and one search key (e.g. Tavily).
Without keys it still runs in a basic rule-based mode, and `DEMO=1` serves saved sample pages offline.

## Quality checks

```bash
pytest -q                             # unit and integration tests (no network needed)
python -m scripts.benchmark 5         # real requests; writes a CSV with result counts and time per stage
```

The benchmark is how changes are measured: for example, per-stage deadlines and parallel page reading took the
average request from 55 s to 40 s while the average number of matching results went from 6.8 to 9.

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/chat` | One chat turn: `{messages: [{role, text}], request, shown}` → results, a question, recommendations or an answer |
| `POST` | `/api/chat/stream` | Same, as NDJSON: progress lines and the reply while it's written, then the answer |
| `GET` `PUT` `DELETE` | `/api/chats[/{id}]` | A signed-in shopper's saved conversations |
| `GET` | `/api/search?q=` | One-shot search |
| `GET` | `/api/history?url=` | Saved price history of a product page |
| `GET` | `/api/config` | Is Google sign-in on, regions a shopper can pick |
| `POST` | `/api/auth/google` | `{credential}` from the Google button → signed session cookie |
| `GET` / `PATCH` / `DELETE` | `/api/me` | The signed-in shopper; save `{market, email_alerts, prefs}`; delete the account and all its data |
| `GET` / `POST` / `DELETE` | `/api/watches` | Followed products (`POST {url, title, store, price}`) |
| `GET` | `/api/notifications` | The shopper's notifications (`POST /api/notifications/seen` marks them read) |
| `POST` | `/api/cron/check-watches` | Re-check followed prices (header `X-Cron-Secret`), run by `.github/workflows/check-watches.yml` |
| `POST` | `/api/report` | "This result is wrong" from a visitor: `{reason, request, url, store, title, price}`; review with `python -m scripts.reports` |

## Free hosting

The repo includes a `render.yaml` Blueprint for Render's free Docker web service (no card): in Render, choose
New → Blueprint, pick this repo, and paste your API keys when asked. The free service sleeps after 15 idle minutes
and its SQLite data resets on restart. (`python -m scripts.deploy_hf` does the same on Hugging Face Spaces, which now
needs a PRO account for Docker.)
Online, `DAILY_LIMIT` (default 5) caps new searches per visitor per day so the free quotas last; cached repeats
don't count, and visitors are only stored as a salted hash of their IP.

### Accounts (optional, all free)

1. **Google sign-in:** in [Google Cloud Console](https://console.cloud.google.com/apis/credentials) create an OAuth
   client ID (Web application), add your site (and `http://localhost:8000`) to *Authorized JavaScript origins*, and
   set `GOOGLE_CLIENT_ID`. Without it the site works as before, without accounts.
2. **Database that survives restarts:** create a free [Supabase](https://supabase.com) project and set
   `DATABASE_URL` to its *Session pooler* connection string (the tables are created on first use). Without it
   accounts live with the prices in `data/prices.db`, which Render's free plan wipes on restart.
3. **Scheduled price checks:** set the same random `CRON_SECRET` on the server and as a GitHub repository secret,
   plus a `SITE_URL` secret; the workflow wakes the server every 6 hours.
4. **Emails (optional):** a free [Brevo](https://www.brevo.com) key (`BREVO_API_KEY`, `MAIL_FROM`). Render's free
   plan blocks SMTP ports, so an HTTP email API is used.

## Respecting stores

The agent only reads public product pages, honours robots.txt for anything fetched through a reader service, and
doesn't try to get around bot protection. Stores that block automated access (e.g. Jumia's search) are meant to be
added through their official affiliate programs.

## Roadmap

1. ~~Faster, more accurate results: price DB, request cache, live progress, benchmark~~ (done)
2. Free deployment and first users, with a "this result is wrong" button (button and daily limit done)
3. Follow a product and get notified when the price drops (done: Google accounts, in-site notifications, email), then a whole category, a Telegram bot, a price-history chart
4. Affiliate links, then Saudi Arabia and the UAE

## Author

Built by [Mahdy](https://github.com/mahdii99).
