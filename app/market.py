"""Which country the agent shops in. Egypt now; Saudi Arabia and the UAE are ready to switch on later.

Set MARKET=EG | SA | AE in .env. Everything country-specific (currency, local domains, words used in web
searches, famous stores) comes from here, so opening a new country is a config change, not a rewrite.
"""
import os
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


def current() -> Market:
    return MARKETS.get(os.getenv("MARKET", "EG").upper(), MARKETS["EG"])
