"""Store affiliate codes: the site earns a commission when a shopper buys after opening a store link from us.

AFFILIATE_TAGS (env, never in the repo) holds one rule per store domain, split by ';'. A rule is one of:
  - query codes added to the store link:   "amazon.eg: tag=mycode-21"   "noon.com: utm_source=x&utm_medium=y"
  - a network's tracking link (Admitad, ArabClicks, DCMnetwork...), with {url} where the product page goes:
        "jumia.com.eg: https://ad.admitad.com/g/abc123/?ulp={url}"
A domain also matches its subdomains. The same rules are used by the page (static/index.html, tagged()) and by the
price-drop emails, so every store link the shopper opens carries our code. Empty = links stay as the stores gave them.
"""

import os
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


def rules() -> dict[str, dict]:
    """{"amazon.eg": {"params": {"tag": "mycode-21"}}, "jumia.com.eg": {"template": "https://...{url}"}}.
    Broken rules are skipped (a template must be https and hold {url}), never break the links."""
    out = {}
    for part in os.getenv("AFFILIATE_TAGS", "").split(";"):
        domain, _, value = part.partition(":")
        domain, value = domain.strip().lower().removeprefix("www."), value.strip()
        if not domain or not value:
            continue
        if value.lower().startswith("http"):
            if value.lower().startswith("https://") and "{url}" in value:
                out[domain] = {"template": value}
            continue
        pairs = dict(p.split("=", 1) for p in value.split("&") if "=" in p)
        pairs = {k.strip(): v.strip() for k, v in pairs.items() if k.strip()}
        if pairs:
            out[domain] = {"params": pairs}
    return out


def _rule_for(host: str, table: dict[str, dict]) -> dict | None:
    host = host.lower().removeprefix("www.")
    match = [d for d in table if host == d or host.endswith("." + d)]
    return table[max(match, key=len)] if match else None  # the most specific domain wins


def store_has_code(store: str) -> bool:
    return _rule_for(store or "", rules()) is not None


def link(url: str) -> str:
    """The store link with our code on it (the page does the same in tagged())."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return url
    rule = _rule_for(parts.hostname, rules())
    if not rule:
        return url
    if "template" in rule:
        return rule["template"].replace("{url}", quote(url, safe=""))
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(rule["params"])
    return urlunsplit(parts._replace(query=urlencode(query)))
