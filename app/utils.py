import re

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def parse_price(text: str | None) -> float | None:
    """'EGP 1,299.00' / '1,299 ج.م' -> 1299.0"""
    if not text:
        return None
    text = text.translate(_AR_DIGITS)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def normalize(text: str) -> str:
    text = text.translate(_AR_DIGITS).lower()
    # "10,000 mAh" / "10000 mah" / "10k mah" -> "10000mah"
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = re.sub(r"(\d+)\s*k\s*mah", lambda m: f"{int(m.group(1)) * 1000}mah", text)
    text = re.sub(r"(\d+)\s*(mah|w|gb|tb|mp|hz|inch)\b", r"\1\2", text)
    return text


# Page text helpers that stay linear on any page. Patterns like `<script.*?</script>` or `<[^<>]+>` scan to the end
# of the page for every opening that never closes (a cut page, a reader copy with stray "<" or "[" in it): on a big
# page that is minutes of CPU inside one regex call, which holds the interpreter lock, so the whole server (and
# Render's health check) waits. Live 2026-11-28.
_BLOCK_OPEN = {}
TAG = re.compile(r"<[^<>]+>")  # a tag: never crosses another "<"


def drop_blocks(html: str, tags: tuple[str, ...] = ("script", "style"), repl: str = " ") -> str:
    """The page without <tag ...>...</tag> blocks (case-insensitive); an unclosed one is dropped to the end."""
    key = tags
    if key not in _BLOCK_OPEN:
        _BLOCK_OPEN[key] = re.compile(r"<(" + "|".join(tags) + r")\b", re.I)
    opener = _BLOCK_OPEN[key]
    low = None
    out, pos = [], 0
    while (m := opener.search(html, pos)) is not None:
        if low is None:
            low = html.lower()
        out.append(html[pos:m.start()])
        out.append(repl)
        close = low.find("</" + m.group(1).lower(), m.end())
        if close < 0:
            return "".join(out)
        end = html.find(">", close)
        pos = len(html) if end < 0 else end + 1
    out.append(html[pos:])
    return "".join(out)


def strip_tags(html: str, repl: str = " ") -> str:
    return TAG.sub(repl, html)


def script_bodies(html: str):
    """The text inside each <script ...>...</script> (an unclosed one at the end of a cut page: to the end)."""
    low = html.lower()
    pos = 0
    while (start := low.find("<script", pos)) >= 0:
        open_end = html.find(">", start)
        if open_end < 0:
            return
        close = low.find("</script", open_end)
        yield html[open_end + 1:close if close >= 0 else len(html)]
        if close < 0:
            return
        pos = close + 8


class _Block:
    def __init__(self, inner: str):
        self._inner = inner

    def group(self, _n: int = 1) -> str:
        return self._inner


def first_block(html: str, tag: str, limit: int = 3000) -> "_Block | None":
    """The first <tag ...>...</tag> of the page whose inside is at most `limit` characters (like re.search's match:
    .group(1) is the inside), found without scanning the page once per opening."""
    low, opener = html.lower(), "<" + tag
    pos = 0
    while (start := low.find(opener, pos)) >= 0:
        nxt = low[start + len(opener):start + len(opener) + 1]
        if nxt and (nxt.isalnum() or nxt in "-_:"):  # <h10>, <h1x>: another tag
            pos = start + 1
            continue
        end = html.find(">", start)
        if end < 0:
            return None
        close = low.find("</" + tag, end, end + 1 + limit + len(tag) + 2)
        if close >= 0:
            return _Block(html[end + 1:close])
        pos = end + 1
    return None
