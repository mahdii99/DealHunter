"""Which country the agent shops in. Egypt now; Saudi Arabia and the UAE are ready to switch on later.

Set MARKET=EG | SA | AE in .env. Everything country-specific (currency, local domains, words used in web
searches, famous stores) comes from here, so opening a new country is a config change, not a rewrite.
"""
import os
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Market:
    code: str
    country_en: str
    country_ar: str
    currency: str              # ISO code shown in prices
    currency_ar: str           # how prices are written for the shopper
    currency_aliases: tuple[str, ...]  # other ways stores write it on their pages
    tld: str                   # local domain ending, e.g. "eg" for .com.eg
    famous_stores: dict[str, str] = field(default_factory=dict)  # domain -> display name


MARKETS = {
    "EG": Market("EG", "Egypt", "مصر", "EGP", "ج.م", ("EGP", "LE", "E£", "جنيه"), "eg"),
    "SA": Market("SA", "Saudi Arabia", "السعودية", "SAR", "ر.س", ("SAR", "SR", "ر.س", "ريال"), "sa", {
        "amazon.sa": "Amazon.sa", "noon.com": "noon", "extra.com": "eXtra", "jarir.com": "Jarir",
        "almanea.sa": "Al Manea", "shakersa.com": "Shaker"}),
    "AE": Market("AE", "United Arab Emirates", "الإمارات", "AED", "د.إ", ("AED", "Dhs", "د.إ", "درهم"), "ae", {
        "amazon.ae": "Amazon.ae", "noon.com": "noon", "sharafdg.com": "Sharaf DG", "namshi.com": "Namshi",
        "carrefouruae.com": "Carrefour", "luluhypermarket.com": "Lulu"}),
}


# The country of the search running now: a signed-in shopper's region when the agent already searches there
# (see live()), else MARKET from .env. Set per request, so two shoppers in different countries don't mix.
_CHOSEN: ContextVar[str | None] = ContextVar("market", default=None)


def current() -> Market:
    code = _CHOSEN.get() or os.getenv("MARKET", "EG").upper()
    return MARKETS.get(code, MARKETS["EG"])


def live() -> set[str]:
    """Countries whose stores the agent can search now (LIVE_MARKETS=EG,SA in .env). The prompts and trust signals
    are still written for Egypt, so the others stay off until they're tuned; shoppers can already pick them."""
    return {c.strip().upper() for c in os.getenv("LIVE_MARKETS", "EG").split(",") if c.strip()} & MARKETS.keys()


def use(code: str | None):
    """Search in this country for the rest of this request (when it's live). Returns a token for reset()."""
    return _CHOSEN.set(code if code in live() else None)


def reset(token) -> None:
    _CHOSEN.reset(token)
