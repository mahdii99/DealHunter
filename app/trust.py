"""Store trust. Not a list of stores to search (the agent finds those on the web).

Big well-known retailers are trusted by name. Every other shop gets a 0-100 score from signals on its own
page (Egyptian phone, cash on delivery, return policy, warranty, address...), so a small but real shop can
rank next to the big ones, and a sketchy one gets dropped by the LLM judge."""
import re
from urllib.parse import unquote, urlparse

from . import market

KNOWN_STORES = {
    "amazon.eg": "Amazon.eg", "noon.com": "noon", "jumia.com.eg": "Jumia", "btech.com": "B.TECH",
    "2b.com.eg": "2B", "rayashop.com": "Raya Shop", "dream2000.com": "Dream 2000",
    "elarabygroup.com": "El Araby", "carrefouregypt.com": "Carrefour", "sharafdg.com.eg": "Sharaf DG",
    "select.com.eg": "Select", "tradeline.com.eg": "Tradeline", "istyle.com.eg": "iStyle",
    "samsung.com": "Samsung", "xiaomi.com": "Xiaomi", "mi.com": "Xiaomi", "anker.com": "Anker",
    "oraimo.com": "Oraimo", "hyperone.com.eg": "HyperOne", "elghazawy.com": "El Ghazawy",
    "compumarts.com": "CompuMarts", "sigma-computer.com": "Sigma", "elbadrgroupeg.store": "El Badr",
}
# Well-known retail sites: the agent searches them as one group and every other shop as a second group,
# so the big names can't crowd the smaller (often cheaper) shops out of the results.
FAMOUS_SITES = [d for d in KNOWN_STORES if d not in ("samsung.com", "xiaomi.com", "mi.com", "anker.com", "oraimo.com")]
# Never shop results for new items.
CLASSIFIEDS = ["dubizzle.com.eg", "olx.com.eg"]
# Pages that are never a place to buy from.
NOT_STORES = ("youtube.com", "facebook.com", "instagram.com", "tiktok.com", "wikipedia.org", "reddit.com",
              "twitter.com", "x.com", "pinterest.com", "linkedin.com", "gsmarena.com", "quora.com", "tavily.com", "brave.com",
              "google.com", "bing.com", "medium.com", "blogspot.com",
              # price-comparison sites: they list other stores' prices, you can't buy there
              "pricena.com", "yaoota.com", "pricearena.com", "pricecheck.com", "priceoye.com", "compare.eg")


def domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _match(host: str, d: str) -> bool:
    return host == d or host.endswith("." + d)


def store_name(url: str) -> str:
    host = domain(url)
    return next((n for d, n in KNOWN_STORES.items() if _match(host, d)), host)


def is_known(url: str) -> bool:
    host = domain(url)
    return any(_match(host, d) for d in KNOWN_STORES)


# Country domains of other markets (Saudi, UAE, US...): their prices aren't in EGP and they don't ship here.
_GENERIC_TLDS = {"com", "net", "org", "store", "shop", "co", "online", "app", "io", "info", "biz", "site", "me",
                 "tech", "market", "deals", "xyz"}
_FOREIGN_SUBDOMAINS = ("us.", "uk.", "ae.", "sa.", "ksa.", "kw.", "qa.", "en-us.", "en-ae.", "en-sa.")


# Shops named after another country on a generic domain (pointiraq.com, bahraingoods.com, dubaiphone.net):
# they sell there, not here. The current market's own words are allowed (dubai.. is fine when MARKET=AE).
_COUNTRY_WORDS = {
    "EG": ("egypt", "masr", "misr", "cairo"),
    "SA": ("saudi", "ksa", "riyadh", "jeddah"),
    "AE": ("uae", "dubai", "emirates", "abudhabi", "sharjah"),
    "OTHER": ("iraq", "bahrain", "kuwait", "qatar", "jordan", "lebanon", "morocco", "algeria", "tunisia",
              "libya", "syria", "yemen", "pakistan", "india", "turkey"),
}


def _named_for_other_country(host: str) -> bool:
    name = host.rsplit(".", 1)[0].replace("-", "")
    mine = market.current().code
    return any(w in name for code, words in _COUNTRY_WORDS.items() if code != mine for w in words)


def is_local_market(url: str) -> bool:
    """A shop in the country we're searching (MARKET in .env): local domain (.com.eg) or a generic one (.com)."""
    host = domain(url)
    tld = host.rsplit(".", 1)[-1]
    if tld == market.current().tld:
        return True
    if _named_for_other_country(host):
        return False
    if host.startswith(_FOREIGN_SUBDOMAINS):
        return False
    return tld in _GENERIC_TLDS


# Pages that tell from their address that they are an article, not a product or a shop listing: news, blogs,
# reviews, guides (they mention prices, but you can't buy there). Any site, any product.
_ARTICLE_HOST = re.compile(r"^(?:blog|blogs|news|magazine|mag|articles?)\.", re.I)
_ARTICLE_PATH = re.compile(r"/(?:news|blogs?|articles?|posts?|story|stories|magazine|buying-guides?|reviews?|"
                           r"مقالات|اخبار|أخبار)(?:/|$)|/(?:19|20)\d\d/[01]?\d/", re.I)


def is_article_url(url: str) -> bool:
    parts = urlparse(url)
    return bool(_ARTICLE_HOST.match(domain(url)) or _ARTICLE_PATH.search(unquote(parts.path)))


def is_shop_candidate(url: str) -> bool:
    host = domain(url)
    return (bool(host) and is_local_market(url) and not any(_match(host, d) for d in NOT_STORES)
            and not is_article_url(url))


# (label, points, regex) checked against the page's visible text + links.
_SIGNALS = [
    ("رقم تليفون مصري", 15, r"(?<!\d)(?:\+?20\s?|0)1[0125](?:[\s-]?\d){8}(?!\d)|(?<!\d)1\d{4}(?!\d)\s*(?:hotline|الخط الساخن)|(?:hotline|الخط الساخن)\D{0,15}1\d{4}"),
    ("دفع عند الاستلام", 10, r"cash on delivery|الدفع عند الاستلام|الدفع عند التسليم|\bCOD\b"),
    ("سياسة استرجاع", 15, r"return policy|refund policy|returns? (?:&|and) exchange|سياسة (?:الاسترجاع|الاستبدال)|استرجاع|استبدال"),
    ("ضمان", 10, r"warranty|ضمان"),
    ("عنوان في مصر", 10, r"cairo|giza|alexandria|nasr city|maadi|القاهرة|الجيزة|الإسكندرية|الاسكندرية|مدينة نصر|المعادي|شارع"),
    ("صفحات سوشيال", 5, r"facebook\.com/|instagram\.com/"),
]
_RED_FLAGS = [("كوبي/تقليد", 40, r"replica|first copy|high copy|copy 1:1|\bmaster copy\b|هاي كوبي|كوبي ون|تقليد")]


def page_signals(html: str, url: str, has_product_data: bool) -> tuple[int, list[str]]:
    """Trust score 0-100 for the shop that owns this page, with the reasons."""
    if is_known(url):
        return 100, ["متجر مشهور"]
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    links = " ".join(re.findall(r"href=[\"']([^\"']+)", text))
    text = re.sub(r"<[^>]+>", " ", text) + " " + links
    score, found = 0, []
    if url.startswith("https://"):
        score += 5
    if domain(url).endswith(".eg"):
        score, found = score + 15, found + ["دومين مصري"]
    if has_product_data:
        score, found = score + 10, found + ["صفحة منتجات منظمة"]
    for label, pts, rx in _SIGNALS:
        if re.search(rx, text, re.I):
            score, found = score + pts, found + [label]
    for label, pts, rx in _RED_FLAGS:
        if re.search(rx, text, re.I):
            score, found = score - pts, found + ["⚠ " + label]
    return max(0, min(100, score)), found
